"""Independent figure binding checks, using already audited CSV exports."""

from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "results" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    values = json.loads(
        (ROOT / "results/figure_values.json").read_text(encoding="utf-8")
    )
    metrics = {r["model"]: r for r in read("test_metrics.csv")}
    for row in values["full_models"]:
        for name in ["mse", "rmse", "mae"]:
            assert math.isclose(
                row[name],
                float(metrics[row["model"]][name]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
    stability, importance = read("stability.csv"), read("importance.csv")
    for family, summary in values["resamples"].items():
        subset = [
            r for r in stability if r["family"] == family and r["variant"] != "full"
        ]
        full = next(
            r for r in stability if r["family"] == family and r["variant"] == "full"
        )
        assert (
            len(subset) == 12
            and summary["reference_weather_leader"] == full["top_weather_group"]
        )
        frequencies = {}
        for row in subset:
            frequencies[row["top_weather_group"]] = (
                frequencies.get(row["top_weather_group"], 0) + 1
            )
        assert frequencies == summary["weather_leader_counts"]
        assert summary["headline_changes"] == sum(
            r["top_weather_group"] != full["top_weather_group"] for r in subset
        )
        assert summary["temperature_order_reversals"] == sum(
            r["temperature_order_reversed"] == "True" for r in subset
        )
        for name in ["top3_jaccard", "spearman"]:
            assert summary[name] == [float(r[name]) for r in subset]
        assert summary["joint_temperature_first"] == sum(
            int(r["joint_temperature_weather_rank"]) == 1 for r in subset
        )
    for family, groups in values["temperature_importance"].items():
        for group, summary in groups.items():
            subset = [
                r for r in importance if r["family"] == family and r["group"] == group
            ]
            expected = float(
                next(r for r in subset if r["variant"] == "full")["importance"]
            )
            assert math.isclose(summary["full"], expected, rel_tol=1e-12, abs_tol=1e-9)
            for a, b in zip(
                summary["resampled"],
                [float(r["importance"]) for r in subset if r["variant"] != "full"],
            ):
                assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    for name, digest in values["input_sha256"].items():
        assert (
            hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest() == digest
        )
    receipt = {
        "status": "PASS",
        "full_model_rows": 6,
        "resampled_rank_rows": 48,
        "temperature_importance_series": 12,
        "numeric_bindings": "Independently reconciled to checked source CSV",
        "visual_layout": "Separate rendered inspection required; this script checks values, not pixels",
    }
    (ROOT / "results/INDEPENDENT_PRESENTATION_CHECK.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
