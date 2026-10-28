"""Protected-year scoring and common, stratified model-reliance perturbations."""

import hashlib
import json

import numpy as np
import pandas as pd

from modeling import GROUPS, ROOT, predict, source, write_json


def metrics(target, raw, variance):
    prediction = np.maximum(raw, 0)
    error = prediction - target
    return {
        "rows": len(target),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs(error))),
        "mean_error": float(error.mean()),
        "normalized_mse": float(np.mean(error**2) / variance),
        "negative_raw_predictions": int(np.sum(raw < 0)),
    }


def ordered_names(values):
    return sorted(values, key=lambda name: (-values[name], name))


def ranking_comparison(reference, values):
    first, second = ordered_names(reference), ordered_names(values)
    a, b = set(first[:3]), set(second[:3])
    rank = {name: i for i, name in enumerate(first)}
    squared = sum((rank[name] - i) ** 2 for i, name in enumerate(second))
    n = len(first)
    return {
        "top3_jaccard": len(a & b) / len(a | b),
        "spearman": 1 - 6 * squared / (n * (n * n - 1)),
    }


def main():
    results = ROOT / "results"
    freeze = json.loads((results / "freeze.json").read_text(encoding="utf-8"))
    assert (
        json.loads(
            (results / "INDEPENDENT_TRAINING_CHECK.json").read_text(encoding="utf-8")
        )["status"]
        == "PASS"
    )
    for name, digest in freeze["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    training, test = source(2011), source(2012)
    target = test.cnt.to_numpy(float)
    variance = float(training.cnt.var(ddof=0))
    keys = sorted(
        range(len(test)),
        key=lambda i: hashlib.sha256(test.timestamp[i].encode()).hexdigest(),
    )[:1500]
    evaluation = test.iloc[keys].reset_index(drop=True)
    groups = evaluation.groupby(["mnth", "hr", "workingday"], sort=True).indices
    permutations = []
    for repeat in range(5):
        indices = np.arange(len(evaluation))
        rng = np.random.default_rng(20261028 + repeat)
        for members in groups.values():
            indices[members] = rng.permutation(members)
        permutations.append(indices)
    write_json(
        results / "permutations.json",
        {
            "evaluation_timestamp": evaluation.timestamp.tolist(),
            "source_test_row_indices": keys,
            "strata": ["mnth", "hr", "workingday"],
            "seed_by_repeat": [20261028 + i for i in range(5)],
            "permutation_indices": [i.tolist() for i in permutations],
            "singleton_rows": sum(len(v) for v in groups.values() if len(v) == 1),
            "stratum_count": len(groups),
            "raw_columns_by_group": GROUPS,
            "unchanged_row_fractions": {
                name: [
                    float(
                        np.mean(
                            np.all(
                                evaluation[columns].to_numpy()
                                == evaluation.iloc[index][columns].to_numpy(),
                                axis=1,
                            )
                        )
                    )
                    for index in permutations
                ]
                for name, columns in GROUPS.items()
            },
        },
    )
    all_predictions, all_metrics, importance_rows = [], [], []
    for path in sorted((results / "models/final").glob("*.json")):
        model = json.loads(path.read_text(encoding="utf-8"))
        raw = predict(test, model)
        frame = test[["timestamp", "cnt"]].copy()
        frame["model"] = model["identity"]
        frame["raw_prediction"], frame["prediction"] = raw, np.maximum(raw, 0)
        all_predictions.append(frame)
        all_metrics.append(
            {
                "model": model["identity"],
                "family": model["candidate"]["family"],
                "variant": model["variant"],
                **metrics(target, raw, variance),
            }
        )
        base = predict(evaluation, model)
        base_mse = float(
            np.mean((np.maximum(base, 0) - evaluation.cnt.to_numpy()) ** 2)
        )
        exports = []
        for group, columns in GROUPS.items():
            for repeat, indices in enumerate(permutations):
                changed = evaluation.copy()
                changed[columns] = evaluation.iloc[indices][columns].to_numpy()
                raw = predict(changed, model)
                clipped = np.maximum(raw, 0)
                loss = float(np.mean((clipped - evaluation.cnt.to_numpy()) ** 2))
                importance_rows.append(
                    {
                        "model": model["identity"],
                        "family": model["candidate"]["family"],
                        "variant": model["variant"],
                        "group": group,
                        "repeat": repeat,
                        "base_mse": base_mse,
                        "permuted_mse": loss,
                        "importance": loss - base_mse,
                    }
                )
                export = evaluation[["timestamp", "cnt"]].copy()
                export["group"], export["repeat"] = group, repeat
                export["raw_prediction"], export["prediction"] = raw, clipped
                exports.append(export)
        pd.concat(exports, ignore_index=True).to_csv(
            results / "importance_predictions" / f"{model['identity']}.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        print(f"evaluated {model['identity']}", flush=True)
    mean = float(training.cnt.mean())
    strata = training.groupby(["hr", "workingday"]).cnt.mean().to_dict()
    baselines = {
        "overall_mean": np.full(len(test), mean),
        "hour_workday_mean": np.array(
            [strata.get((row.hr, row.workingday), mean) for row in test.itertuples()]
        ),
    }
    for name, raw in baselines.items():
        frame = test[["timestamp", "cnt"]].copy()
        frame["model"], frame["raw_prediction"], frame["prediction"] = name, raw, raw
        all_predictions.append(frame)
        all_metrics.append(
            {
                "model": name,
                "family": "baseline",
                "variant": "full",
                **metrics(target, raw, variance),
            }
        )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        results / "test_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(all_metrics).to_csv(results / "test_metrics.csv", index=False)
    repeats = pd.DataFrame(importance_rows)
    repeats.to_csv(results / "importance_repeats.csv", index=False)
    values = (
        repeats.groupby(["model", "family", "variant", "group"], sort=True)
        .importance.mean()
        .reset_index()
    )
    values.to_csv(results / "importance.csv", index=False)
    weather = ["weather", "temp", "atemp", "humidity", "windspeed"]
    stability = []
    for family in ["linear", "additive", "cart", "boost"]:
        full = (
            values[(values.family == family) & (values.variant == "full")]
            .set_index("group")
            .importance.to_dict()
        )
        reference = {name: full[name] for name in GROUPS if name != "joint_temperature"}
        for variant in ["full", *[f"week{i:02}" for i in range(12)]]:
            raw_values = (
                values[(values.family == family) & (values.variant == variant)]
                .set_index("group")
                .importance.to_dict()
            )
            current = {name: raw_values[name] for name in reference}
            weather_rank = ordered_names({name: current[name] for name in weather})
            joint_rank = ordered_names(
                {
                    name: raw_values[name]
                    for name in [
                        "joint_temperature",
                        "weather",
                        "humidity",
                        "windspeed",
                    ]
                }
            )
            stability.append(
                {
                    "family": family,
                    "variant": variant,
                    **ranking_comparison(reference, current),
                    "top_weather_group": weather_rank[0],
                    "temperature_order_reversed": (
                        ordered_names(
                            {name: current[name] for name in ["temp", "atemp"]}
                        )
                        != ordered_names(
                            {name: reference[name] for name in ["temp", "atemp"]}
                        )
                    ),
                    "joint_temperature_weather_rank": joint_rank.index(
                        "joint_temperature"
                    )
                    + 1,
                }
            )
    pd.DataFrame(stability).to_csv(results / "stability.csv", index=False)
    write_json(
        results / "evaluation_summary.json",
        {
            "status": "EXECUTED_AWAITING_INDEPENDENT_CHECKS",
            "training_year": 2011,
            "test_year": 2012,
            "test_rows": len(test),
            "evaluation_rows": len(evaluation),
            "model_count": 52,
            "baseline_count": 2,
            "permutation_prediction_rows": 52 * 11 * 5 * len(evaluation),
            "training_target_variance": variance,
            "importance_unit": "increase in count-squared MSE",
            "structurally_unchanged_groups": ["hour", "month", "workingday"],
            "interpretation": "Retrospective model reliance with observed weather; conditional resampling sensitivity, not causal effects or deployment trials",
        },
    )


if __name__ == "__main__":
    (ROOT / "results/importance_predictions").mkdir(parents=True, exist_ok=True)
    main()
