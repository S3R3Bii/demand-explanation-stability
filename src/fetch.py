"""Preserve the exact UCI hourly count source before any fitting."""

from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"


def main():
    path = ROOT / "data/raw/bike-sharing.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    manifest_path = ROOT / "data/manifest.json"
    expected_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    raw = urllib.request.urlopen(URL, timeout=90).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("hour.csv"))
        content = archive.read(member)
    if expected_manifest is not None:
        assert (
            hashlib.sha256(raw).hexdigest() == expected_manifest["zip_sha256"]
        ), "Published archive snapshot changed"
        assert (
            hashlib.sha256(content).hexdigest() == expected_manifest["member_sha256"]
        ), "Published hourly source changed"
        assert (
            hashlib.sha256((ROOT / expected_manifest["panel"]).read_bytes()).hexdigest()
            == expected_manifest["panel_sha256"]
        )
        path.write_bytes(raw)
        print(
            "Verified pinned archive/member/panel hashes; retained original retrieval metadata",
            flush=True,
        )
        return
    path.write_bytes(raw)
    frame = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
    frame["timestamp"] = pd.to_datetime(frame.dteday) + pd.to_timedelta(
        frame.hr, unit="h"
    )
    assert frame.timestamp.is_unique and not frame.isna().any().any()
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
    assert (frame.cnt == frame.casual + frame.registered).all()
    assert set(frame.timestamp.dt.year) == {2011, 2012}
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    expected = pd.date_range(frame.timestamp.min(), frame.timestamp.max(), freq="h")
    missing = expected.difference(frame.timestamp)
    output = ROOT / "data/hour.csv.gz"
    output.write_bytes(
        gzip.compress(
            frame.to_csv(
                index=False, float_format="%.17g", lineterminator="\n"
            ).encode(),
            mtime=0,
        )
    )
    manifest = {
        "source_url": URL,
        "landing_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "creator": "Hadi Fanaee-T",
        "year": 2013,
        "doi": "10.24432/C5W894",
        "license": "CC BY4.0",
        "zip_sha256": hashlib.sha256(raw).hexdigest(),
        "zip_bytes": len(raw),
        "member": member,
        "member_sha256": hashlib.sha256(content).hexdigest(),
        "panel": "data/hour.csv.gz",
        "panel_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "landing_page_instances": 17389,
        "actual_csv_rows": len(frame),
        "landing_count_matches_csv": len(frame) == 17389,
        "missing_hours_within_span": len(missing),
        "missing_timestamps": [x.isoformat() for x in missing],
        "source_changes": "Added timestamp, sorted chronologically, lossless17digit numeric serialization; missing hours remain absent, not zero",
        "prediction_availability": "Contemporaneously observed weather; retrospective explanation, not an original-vintage forecast",
        "private_record_scope": "Public hourly aggregate counts; no individual rider records",
    }
    (ROOT / "data/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    quality = {
        "status": "SOURCE_READY_FOR_INDEPENDENT_CHECK",
        "rows": len(frame),
        "date_min": str(frame.timestamp.min()),
        "date_max": str(frame.timestamp.max()),
        "year_rows": {
            str(k): int(v)
            for k, v in frame.groupby(frame.timestamp.dt.year).size().items()
        },
        "duplicate_timestamp_keys": int(frame.timestamp.duplicated().sum()),
        "target_component_identity": True,
        "missing_hours": len(missing),
        "official_metadata_mismatch_retained": len(frame) != 17389,
    }
    (ROOT / "results/source_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(quality), flush=True)


if __name__ == "__main__":
    main()
