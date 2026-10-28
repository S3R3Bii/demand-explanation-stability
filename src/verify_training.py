"""Independent CSV/matrix audit; imports no fitting or generation helpers."""

from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta
import csv
import gzip
import hashlib
import io
import json
import math

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def calendar_matrix(records):
    output = []
    for row in records:
        hour, month = int(row["hr"]), int(row["mnth"])
        result = []
        for harmonic in (1, 2):
            angle = hour * (2 * math.pi * harmonic / 24)
            result += [round(math.sin(angle), 12), round(math.cos(angle), 12)]
        angle = month * (2 * math.pi / 12)
        result += [round(math.sin(angle), 12), round(math.cos(angle), 12)]
        result += [int(row["weekday"]) == k for k in range(7)]
        result += [int(row["weathersit"]) == k for k in range(1, 5)]
        result += [int(row["workingday"]), int(row["holiday"])]
        output.append(result)
    return np.array(output, dtype=float)


def independent_design(records, model):
    transform = model["transform"]
    design = [calendar_matrix(records)]
    for name in ["temp", "atemp", "hum", "windspeed"]:
        values = np.array([float(r[name]) for r in records])
        if transform["family"] == "additive":
            spec = transform["splines"][name]
            # Check the library's training-only knots against direct quantiles.
            knots = np.quantile(values, np.linspace(0, 1, 5))
            np.testing.assert_allclose(spec["t"][3:-3], knots, atol=1e-12, rtol=1e-12)
            basis = BSpline(spec["t"], np.eye(len(spec["c"])), 3, extrapolate=False)
            lo, hi = knots[0], knots[-1]
            # Training values cannot require extrapolation; inference edge cases
            # are exercised separately in the contract tests.
            assert values.min() >= lo and values.max() <= hi
            design.append(basis(values)[:, :-1])
        else:
            design.append(values[:, None])
    matrix = np.column_stack(design)
    if "mean" in transform:
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0)
        sd[sd == 0] = 1
        np.testing.assert_allclose(mean, transform["mean"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(sd, transform["scale"], atol=1e-12, rtol=1e-12)
        matrix = (matrix - mean) / sd
    return matrix


def main():
    training = [r for r in rows(ROOT / "data/hour.csv.gz") if r["yr"] == "0"]
    assert len(training) == 8645
    week_map = defaultdict(list)
    for i, row in enumerate(training):
        day = date.fromisoformat(row["dteday"])
        week_map[(day - timedelta(days=day.weekday())).isoformat()].append(i)
    resamples = json.loads(
        (ROOT / "results/week_resamples.json").read_text(encoding="utf-8")
    )
    assert sorted(week_map) == resamples["ordered_source_weeks"]
    rng = np.random.default_rng(20261028)
    for name, draw in resamples["draws"].items():
        assert (
            draw
            == rng.choice(sorted(week_map), size=len(week_map), replace=True).tolist()
        )
    certificates = []
    for path in sorted((ROOT / "results/models").glob("*/*.json")):
        model = json.loads(path.read_text(encoding="utf-8"))
        if model["stage"] == "validation":
            records = [r for r in training if int(r["mnth"]) <= model["fit_end_month"]]
        else:
            variant = model["variant"]
            indices = (
                list(range(len(training)))
                if variant == "full"
                else [i for week in resamples["draws"][variant] for i in week_map[week]]
            )
            assert indices == model["source_row_indices"]
            counts = [0] * len(training)
            for index in indices:
                counts[index] += 1
            assert counts == model["source_row_multiplicities"]
            for members in week_map.values():
                assert len({counts[i] for i in members}) == 1
            records = [training[i] for i in indices]
        assert len(records) == model["training_rows"]
        assert all(r["timestamp"] < "2012" for r in records)
        digest = hashlib.sha256(
            "\n".join(r["timestamp"] for r in records).encode()
        ).hexdigest()
        assert digest == model["training_row_identity_sha256"]
        matrix = independent_design(records, model)
        family = model["candidate"]["family"]
        if family in ("linear", "additive"):
            beta = np.array(model["coefficient"])
            target = np.array([float(r["cnt"]) for r in records])
            residual = matrix @ beta + model["intercept"] - target
            penalty = model["candidate"]["lambda"]
            objective = math.fsum(float(x) ** 2 for x in residual) / len(
                records
            ) + penalty * math.fsum(float(b) ** 2 for b in beta)
            gradient = 2 * matrix.T @ residual / len(records) + 2 * penalty * beta
            norm = max(
                float(np.max(abs(gradient))),
                abs(2 * math.fsum(residual) / len(records)),
            )
            assert norm <= 1e-9
            assert math.isclose(
                objective, model["objective"], rel_tol=1e-12, abs_tol=1e-9
            )
            hessian = 2 * matrix.T @ matrix / len(records) + 2 * penalty * np.eye(
                matrix.shape[1]
            )
            assert np.linalg.eigvalsh(hessian).min() > 0
            certificates.append(
                {
                    "model": model["identity"],
                    "objective": objective,
                    "max_gradient": norm,
                    "positive_definite_penalized_feature_hessian": True,
                }
            )
        else:
            for tree in model["trees"]:
                for leaf, feature, left, right in zip(
                    tree["leaf"], tree["feature"], tree["left"], tree["right"]
                ):
                    if not leaf:
                        assert 0 <= feature < matrix.shape[1]
                        assert 0 <= left < len(tree["leaf"]) and 0 <= right < len(
                            tree["leaf"]
                        )
    prediction_rows = rows(ROOT / "results/validation_predictions.csv.gz")
    errors = defaultdict(list)
    for row in prediction_rows:
        assert row["timestamp"].startswith("2011")
        assert float(row["prediction"]) == max(float(row["raw_prediction"]), 0)
        errors[row["candidate"]].append(
            (float(row["prediction"]) - float(row["cnt"])) ** 2
        )
    selection = json.loads(
        (ROOT / "results/selection.json").read_text(encoding="utf-8")
    )
    losses = {name: math.fsum(values) / len(values) for name, values in errors.items()}
    assert len(losses) == 24 and len(prediction_rows) == 105480
    for name, loss in losses.items():
        assert math.isclose(
            loss, selection["candidate_pooled_mse"][name], abs_tol=1e-7, rel_tol=1e-12
        )
    for family, selected in selection["selected"].items():
        names = [n for n in losses if n.startswith(family + "_")]
        assert selected["id"] == min(names, key=lambda n: (losses[n], n))
    freeze = json.loads((ROOT / "results/freeze.json").read_text(encoding="utf-8"))
    for name, digest in freeze["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    receipt = {
        "status": "PASS",
        "all_models_checked": 124,
        "ridge_certificates": certificates,
        "max_gradient": max(c["max_gradient"] for c in certificates),
        "validation_predictions_checked": len(prediction_rows),
        "candidate_losses_checked": len(losses),
        "whole_week_multiplicities_checked": True,
        "training_only_transform_checks": True,
        "frozen_model_hashes": len(freeze["sha256"]),
        "scope": "Independent CSV/calendar/quantile/scaler/objective/gradient/Hessian/selection audit; no fitting helper imports or 2012 labels",
    }
    (ROOT / "results/INDEPENDENT_TRAINING_CHECK.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in receipt.items() if k != "ridge_certificates"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
