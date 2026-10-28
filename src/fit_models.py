"""Run the declared 2011-only search and freeze week-perturbed models."""

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from modeling import ROOT, candidates, fit, predict, source, write_json


def main():
    training = source(2011)
    assert len(training) == 8645 and training.timestamp.max() < "2012"
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    attempts, predictions, objectives = [], [], {}
    for candidate in candidates():
        squared_error = 0.0
        count = 0
        for end, start, finish in [(6, 7, 8), (8, 9, 10), (10, 11, 12)]:
            fitting = training[training.mnth <= end].reset_index(drop=True)
            validation = training[training.mnth.between(start, finish)].reset_index(
                drop=True
            )
            identity = f"{candidate['id']}_through{end:02}"
            started = time.perf_counter()
            try:
                model = fit(fitting, candidate)
                model.update(identity=identity, stage="validation", fit_end_month=end)
                raw = predict(validation, model)
                output = np.maximum(raw, 0)
                error = float(np.sum((output - validation.cnt.to_numpy()) ** 2))
                squared_error += error
                count += len(validation)
                write_json(results / "models/validation" / f"{identity}.json", model)
                frame = validation[["timestamp", "cnt"]].copy()
                frame["candidate"] = candidate["id"]
                frame["fold_end_month"] = end
                frame["raw_prediction"], frame["prediction"] = raw, output
                predictions.append(frame)
                attempts.append(
                    {
                        "candidate": candidate["id"],
                        "family": candidate["family"],
                        "fold_end_month": end,
                        "status": "success",
                        "training_rows": len(fitting),
                        "validation_rows": len(validation),
                        "sse": error,
                        "mse": error / len(validation),
                        "fit_seconds": model["fit_seconds"],
                        "warnings": " | ".join(model["warnings"]),
                    }
                )
            except Exception as exc:
                attempts.append(
                    {
                        "candidate": candidate["id"],
                        "family": candidate["family"],
                        "fold_end_month": end,
                        "status": "failed",
                        "error": repr(exc),
                        "fit_seconds": time.perf_counter() - started,
                    }
                )
                pd.DataFrame(attempts).to_csv(
                    results / "candidate_attempts.csv", index=False
                )
                raise
            pd.DataFrame(attempts).to_csv(
                results / "candidate_attempts.csv", index=False
            )
        objectives[candidate["id"]] = squared_error / count
        print(
            f"validation {candidate['id']}: pooled MSE={objectives[candidate['id']]:.6f}",
            flush=True,
        )
    pd.concat(predictions, ignore_index=True).to_csv(
        results / "validation_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    selected = {}
    for family in ("linear", "additive", "cart", "boost"):
        eligible = [c for c in candidates() if c["family"] == family]
        selected[family] = min(eligible, key=lambda c: (objectives[c["id"]], c["id"]))
    write_json(
        results / "selection.json",
        {
            "criterion": "Pooled 2011 forward validation MSE after fixed zero clipping",
            "candidate_pooled_mse": objectives,
            "selected": selected,
            "test_used_for_selection": False,
        },
    )
    dates = pd.to_datetime(training.dteday)
    week = (
        (dates - pd.to_timedelta(dates.dt.weekday, unit="D"))
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    weeks = sorted(set(week))
    rng = np.random.default_rng(20261028)
    draws = {
        f"week{repeat:02}": rng.choice(weeks, size=len(weeks), replace=True).tolist()
        for repeat in range(12)
    }
    write_json(
        results / "week_resamples.json",
        {
            "seed": 20261028,
            "ordered_source_weeks": weeks,
            "draws": draws,
            "source_week_by_timestamp": dict(zip(training.timestamp, week.tolist())),
        },
    )
    fits = []
    for family, candidate in selected.items():
        for variant in ["full", *draws]:
            indices = (
                np.arange(len(training))
                if variant == "full"
                else np.concatenate([np.flatnonzero(week == w) for w in draws[variant]])
            )
            sample = training.iloc[indices].reset_index(drop=True)
            model = fit(sample, candidate)
            identity = f"{family}_{variant}"
            model.update(
                identity=identity,
                stage="frozen_2011",
                variant=variant,
                source_row_indices=indices.tolist(),
                source_unique_rows=int(len(set(indices))),
                source_row_multiplicities=np.bincount(
                    indices, minlength=len(training)
                ).tolist(),
            )
            path = results / "models/final" / f"{identity}.json"
            write_json(path, model)
            fits.append(
                {
                    "identity": identity,
                    "family": family,
                    "variant": variant,
                    "training_rows": len(sample),
                    "unique_training_rows": len(set(indices)),
                    "fit_seconds": model["fit_seconds"],
                    "warnings": " | ".join(model["warnings"]),
                }
            )
            print(f"frozen {identity}: {len(sample)} training rows", flush=True)
    pd.DataFrame(fits).to_csv(results / "final_fit_attempts.csv", index=False)
    files = [
        results / "selection.json",
        results / "week_resamples.json",
        *sorted((results / "models/final").glob("*.json")),
    ]
    write_json(
        results / "freeze.json",
        {
            "status": "52_MODELS_FROZEN_BEFORE_2012_INFERENCE",
            "training_year": 2011,
            "final_model_count": len(fits),
            "candidate_fold_attempts": len(attempts),
            "test_scored": False,
            "sha256": {
                str(p.relative_to(ROOT))
                .replace("\\", "/"): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in files
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
