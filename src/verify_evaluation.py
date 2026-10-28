"""Recompute every error/rank from CSV and sample frozen inference independently."""

from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-7), (
        actual,
        expected,
    )


def scalar_prediction(row, model):
    values = []
    for harmonic in (1, 2):
        angle = int(row["hr"]) * (2 * math.pi * harmonic / 24)
        values += [round(math.sin(angle), 12), round(math.cos(angle), 12)]
    angle = int(row["mnth"]) * (2 * math.pi / 12)
    values += [round(math.sin(angle), 12), round(math.cos(angle), 12)]
    values += [int(row["weekday"]) == k for k in range(7)]
    values += [int(row["weathersit"]) == k for k in range(1, 5)]
    values += [int(row["workingday"]), int(row["holiday"])]
    transform = model["transform"]
    for name in ["temp", "atemp", "hum", "windspeed"]:
        number = float(row[name])
        if transform["family"] == "additive":
            spec = transform["splines"][name]
            spline = BSpline(spec["t"], spec["c"], spec["k"], extrapolate=False)
            low, high = spec["t"][3], spec["t"][-4]
            boundary = max(low, min(number, high))
            basis = spline(boundary) + (number - boundary) * spline(boundary, nu=1)
            values += basis[:-1].tolist()
        else:
            values.append(number)
    if "mean" in transform:
        values = [
            (v - mean) / sd
            for v, mean, sd in zip(values, transform["mean"], transform["scale"])
        ]
        return (
            math.fsum(v * b for v, b in zip(values, model["coefficient"]))
            + model["intercept"]
        )
    if model["candidate"]["family"] == "cart":
        values = np.array(values, dtype=np.float32).tolist()
    total = model["baseline"]
    for tree in model["trees"]:
        node = 0
        while not tree["leaf"][node]:
            branch = (
                "left"
                if values[tree["feature"][node]] <= tree["threshold"][node]
                else "right"
            )
            node = tree[branch][node]
        total += tree["value"][node]
    return total


def ranked(values):
    return sorted(values, key=lambda name: (-values[name], name))


def main():
    results = ROOT / "results"
    source = list(csv_rows(ROOT / "data/hour.csv.gz"))
    training = [r for r in source if r["yr"] == "0"]
    test = [r for r in source if r["yr"] == "1"]
    targets = {r["timestamp"]: int(r["cnt"]) for r in test}
    source_by_key = {r["timestamp"]: r for r in test}
    mean = math.fsum(int(r["cnt"]) for r in training) / len(training)
    variance = math.fsum((int(r["cnt"]) - mean) ** 2 for r in training) / len(training)
    history = defaultdict(list)
    for r in training:
        history[(r["hr"], r["workingday"])].append(int(r["cnt"]))
    reference = {k: math.fsum(v) / len(v) for k, v in history.items()}
    grouped, negatives, observed, base_predictions = (
        defaultdict(list),
        defaultdict(int),
        defaultdict(set),
        {},
    )
    for r in csv_rows(results / "test_predictions.csv.gz"):
        name, key = r["model"], r["timestamp"]
        assert key not in observed[name] and int(r["cnt"]) == targets[key]
        observed[name].add(key)
        raw, prediction = float(r["raw_prediction"]), float(r["prediction"])
        assert prediction == max(raw, 0) and math.isfinite(raw)
        if name == "overall_mean":
            close(raw, mean)
        if name == "hour_workday_mean":
            row = source_by_key[key]
            close(raw, reference.get((row["hr"], row["workingday"]), mean))
        grouped[name].append(prediction - targets[key])
        negatives[name] += raw < 0
        base_predictions[(name, key)] = prediction
    assert len(grouped) == 54 and all(
        keys == set(targets) for keys in observed.values()
    )
    for r in csv_rows(results / "test_metrics.csv"):
        error = grouped[r["model"]]
        mse = math.fsum(e * e for e in error) / len(error)
        for key, actual in {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": math.fsum(abs(e) for e in error) / len(error),
            "mean_error": math.fsum(error) / len(error),
            "normalized_mse": mse / variance,
        }.items():
            close(actual, r[key])
        assert int(r["negative_raw_predictions"]) == negatives[r["model"]]
    permutations = json.loads(
        (results / "permutations.json").read_text(encoding="utf-8")
    )
    keys = sorted(targets, key=lambda key: hashlib.sha256(key.encode()).hexdigest())[
        :1500
    ]
    key_set = set(keys)
    assert keys == permutations["evaluation_timestamp"]
    evaluation = [source_by_key[key] for key in keys]
    strata = defaultdict(list)
    for i, r in enumerate(evaluation):
        strata[(int(r["mnth"]), int(r["hr"]), int(r["workingday"]))].append(i)
    assert sum(len(v) == 1 for v in strata.values()) == permutations["singleton_rows"]
    for repeat, indices in enumerate(permutations["permutation_indices"]):
        assert sorted(indices) == list(range(len(keys)))
        reconstructed = list(range(len(keys)))
        rng = np.random.default_rng(20261028 + repeat)
        for stratum in sorted(strata):
            members = strata[stratum]
            for destination, origin in zip(members, rng.permutation(members)):
                reconstructed[destination] = int(origin)
        assert reconstructed == indices
        for i, j in enumerate(indices):
            assert all(
                evaluation[i][name] == evaluation[j][name]
                for name in permutations["strata"]
            )
    importance = {}
    total_exported, sampled_inference = 0, 0
    repeats_from_export = {}
    for path in sorted((results / "importance_predictions").glob("*.csv.gz")):
        name = path.name.removesuffix(".csv.gz")
        model = json.loads(
            (results / "models/final" / f"{name}.json").read_text(encoding="utf-8")
        )
        losses, seen = defaultdict(list), defaultdict(set)
        for r in csv_rows(path):
            group, repeat, key = r["group"], int(r["repeat"]), r["timestamp"]
            pair = (group, repeat)
            assert (
                key not in seen[pair]
                and key in key_set
                and int(r["cnt"]) == targets[key]
            )
            seen[pair].add(key)
            raw, prediction = float(r["raw_prediction"]), float(r["prediction"])
            assert prediction == max(raw, 0) and math.isfinite(raw)
            losses[pair].append((prediction - targets[key]) ** 2)
            if key in keys[:2]:
                i = keys.index(key)
                j = permutations["permutation_indices"][repeat][i]
                changed = dict(evaluation[i])
                for column in permutations["raw_columns_by_group"][group]:
                    changed[column] = evaluation[j][column]
                close(scalar_prediction(changed, model), raw)
                sampled_inference += 1
            total_exported += 1
        assert len(losses) == 55 and all(k == set(keys) for k in seen.values())
        base_mse = math.fsum(
            (base_predictions[(name, key)] - targets[key]) ** 2 for key in keys
        ) / len(keys)
        for pair, squared_errors in losses.items():
            repeat_mse = math.fsum(squared_errors) / len(keys)
            repeats_from_export[(name, *pair)] = (
                base_mse,
                repeat_mse,
                repeat_mse - base_mse,
            )
        for group in permutations["raw_columns_by_group"]:
            importance[(name, group)] = (
                math.fsum(repeats_from_export[(name, group, i)][2] for i in range(5))
                / 5
            )
    assert total_exported == 4290000 and sampled_inference == 5720
    for r in csv_rows(results / "importance_repeats.csv"):
        actual = repeats_from_export[(r["model"], r["group"], int(r["repeat"]))]
        for number, field in zip(actual, ["base_mse", "permuted_mse", "importance"]):
            close(number, r[field])
    for r in csv_rows(results / "importance.csv"):
        close(importance[(r["model"], r["group"])], r["importance"])
    weather = ["weather", "temp", "atemp", "humidity", "windspeed"]
    main_groups = [
        name
        for name in permutations["raw_columns_by_group"]
        if name != "joint_temperature"
    ]
    for r in csv_rows(results / "stability.csv"):
        family, variant = r["family"], r["variant"]
        a = {g: importance[(family + "_full", g)] for g in main_groups}
        b = {g: importance[(family + "_" + variant, g)] for g in main_groups}
        order_a, order_b = ranked(a), ranked(b)
        union, overlap = set(order_a[:3]) | set(order_b[:3]), set(order_a[:3]) & set(
            order_b[:3]
        )
        close(len(overlap) / len(union), r["top3_jaccard"])
        ranks = {name: i for i, name in enumerate(order_a)}
        close(
            1
            - 6
            * sum((ranks[name] - i) ** 2 for i, name in enumerate(order_b))
            / (10 * 99),
            r["spearman"],
        )
        assert ranked({g: b[g] for g in weather})[0] == r["top_weather_group"]
        reversal = ranked({g: a[g] for g in ["temp", "atemp"]}) != ranked(
            {g: b[g] for g in ["temp", "atemp"]}
        )
        assert str(reversal) == r["temperature_order_reversed"]
        joint = ranked(
            {
                g: importance[(family + "_" + variant, g)]
                for g in ["joint_temperature", "weather", "humidity", "windspeed"]
            }
        )
        assert joint.index("joint_temperature") + 1 == int(
            r["joint_temperature_weather_rank"]
        )
    summary = {
        "status": "PASS",
        "test_rows_per_model": len(test),
        "model_and_baseline_count": len(grouped),
        "test_predictions_checked": sum(len(v) for v in grouped.values()),
        "permuted_predictions_checked": total_exported,
        "independent_scalar_inference_spot_checks": sampled_inference,
        "importance_repeat_rows_checked": len(repeats_from_export),
        "mean_importance_rows_checked": len(importance),
        "rank_rows_checked": 52,
        "source_target_joins_all_rows": True,
        "baselines_all_rows": True,
        "common_stratified_permutations": True,
        "scope": "All exported error/importance/rank arithmetic; scalar inference sampled at2fixed evaluation keys per group/repeat/model; not every inference independently rerun",
    }
    (results / "INDEPENDENT_EVALUATION_CHECK.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
