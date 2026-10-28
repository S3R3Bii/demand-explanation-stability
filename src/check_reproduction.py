"""Fresh fit/score/independent replay at the original declared numeric tolerances."""

from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE, RELATIVE = 1e-7, 1e-9


def compare_json(expected, actual, location="root"):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), location
        for name in expected:
            if name in {
                "fit_seconds",
                "max_gradient",
                "training_inference_max_difference",
                "input_sha256",
            }:
                continue
            compare_json(expected[name], actual[name], location + "." + name)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), location
        for i, (first, second) in enumerate(zip(expected, actual)):
            compare_json(first, second, location + f"[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(expected, actual, abs_tol=ABSOLUTE, rel_tol=RELATIVE), (
            location,
            expected,
            actual,
        )
    else:
        assert expected == actual, location


def main():
    started = time.perf_counter()
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    for script in [
        "verify_training.py",
        "verify_evaluation.py",
        "verify_presentation.py",
    ]:
        subprocess.run(
            [sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env, check=True
        )
    with tempfile.TemporaryDirectory(prefix="demand-replay-") as temporary:
        work = Path(temporary)
        shutil.copytree(ROOT / "src", work / "src")
        shutil.copytree(ROOT / "data", work / "data")
        for script in [
            "verify_sources.py",
            "fit_models.py",
            "verify_training.py",
            "evaluate.py",
            "verify_evaluation.py",
            "figures.py",
            "verify_presentation.py",
        ]:
            (work / "results/importance_predictions").mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(work / "src" / script)],
                cwd=work,
                env=env,
                check=True,
            )
        for name in [
            "selection.json",
            "week_resamples.json",
            "permutations.json",
            "evaluation_summary.json",
            "figure_values.json",
        ]:
            compare_json(
                json.loads((ROOT / "results" / name).read_text(encoding="utf-8")),
                json.loads((work / "results" / name).read_text(encoding="utf-8")),
                name,
            )
        for path in sorted((ROOT / "results/models").glob("*/*.json")):
            compare_json(
                json.loads(path.read_text(encoding="utf-8")),
                json.loads((work / path.relative_to(ROOT)).read_text(encoding="utf-8")),
                path.name,
            )
        frames = [
            "candidate_attempts.csv",
            "final_fit_attempts.csv",
            "validation_predictions.csv.gz",
            "test_predictions.csv.gz",
            "test_metrics.csv",
            "importance_repeats.csv",
            "importance.csv",
            "stability.csv",
        ]
        frames += [
            str(path.relative_to(ROOT / "results"))
            for path in sorted(
                (ROOT / "results/importance_predictions").glob("*.csv.gz")
            )
        ]
        for name in frames:
            expected = pd.read_csv(
                ROOT / "results" / name, float_precision="round_trip"
            )
            actual = pd.read_csv(work / "results" / name, float_precision="round_trip")
            if "fit_seconds" in expected:
                expected, actual = expected.drop(columns="fit_seconds"), actual.drop(
                    columns="fit_seconds"
                )
            pd.testing.assert_frame_equal(
                expected, actual, check_exact=False, atol=ABSOLUTE, rtol=RELATIVE
            )
    receipt = {
        "status": "PASS",
        "scope": "Fresh124fits,52frozen variants,all original/source/optimization/error/rank checks and prediction exports",
        "absolute_tolerance": ABSOLUTE,
        "relative_tolerance": RELATIVE,
        "end_to_end_seconds": time.perf_counter() - started,
        "runtime_and_gradient_diagnostics": "Retained and independently bounded, not compared as identical platform timings/residuals",
    }
    (ROOT / "results/reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
