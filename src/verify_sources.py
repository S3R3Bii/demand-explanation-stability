"""Independent CSV, calendar and SQLite audit of every original hourly row."""
from pathlib import Path
from datetime import datetime, timedelta
import csv
import gzip
import hashlib
import io
import json
import math
import sqlite3
import zipfile

ROOT = Path(__file__).resolve().parents[1]
def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    raw = (ROOT / "data/raw/bike-sharing.zip").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["zip_sha256"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        member = archive.read(manifest["member"])
    assert hashlib.sha256(member).hexdigest() == manifest["member_sha256"]
    content = (ROOT / manifest["panel"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == manifest["panel_sha256"]
    source = list(csv.DictReader(io.StringIO(member.decode("utf-8-sig"))))
    panel = list(csv.DictReader(io.StringIO(gzip.decompress(content).decode())))
    source.sort(key=lambda row: (row["dteday"], int(row["hr"])))
    assert len(source) == len(panel) == 17379
    keys = set(); database_rows = []
    for original, actual in zip(source, panel):
        timestamp = datetime.fromisoformat(original["dteday"]) + timedelta(hours=int(original["hr"]))
        assert datetime.fromisoformat(actual["timestamp"]) == timestamp
        assert timestamp not in keys; keys.add(timestamp)
        for key, value in original.items():
            if key == "dteday": assert value == actual[key]
            else:
                assert math.isfinite(float(value)) and float(value) == float(actual[key]), (key, value, actual[key])
        assert int(original["weekday"]) == (timestamp.weekday() + 1) % 7
        assert int(original["mnth"]) == timestamp.month and int(original["yr"]) == timestamp.year - 2011
        assert int(original["workingday"]) == int(timestamp.weekday() < 5 and not int(original["holiday"]))
        assert int(original["holiday"]) in (0, 1) and 1 <= int(original["weathersit"]) <= 4
        assert all(0 <= float(original[k]) <= 1 for k in ["temp", "atemp", "hum", "windspeed"])
        assert int(original["cnt"]) == int(original["casual"]) + int(original["registered"])
        week = (timestamp.date() - timedelta(days=timestamp.weekday())).isoformat()
        database_rows.append((timestamp.isoformat(), timestamp.year, timestamp.month, int(original["hr"]), int(original["cnt"]), week))
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE source(timestamp TEXT PRIMARY KEY,year INTEGER,month INTEGER,hour INTEGER,cnt INTEGER,week TEXT)")
    connection.executemany("INSERT INTO source VALUES(?,?,?,?,?,?)", database_rows)
    year_counts = connection.execute("SELECT year,COUNT(*),SUM(cnt),COUNT(DISTINCT week) FROM source GROUP BY year ORDER BY year").fetchall()
    missing = []
    cursor = min(keys)
    while cursor <= max(keys):
        if cursor not in keys: missing.append(cursor.isoformat())
        cursor += timedelta(hours=1)
    assert missing == manifest["missing_timestamps"] and len(missing) == 165
    folds = []
    for end, start, finish in [(6, 7, 8), (8, 9, 10), (10, 11, 12)]:
        fit = connection.execute("SELECT COUNT(*) FROM source WHERE year=2011 AND month<=?", (end,)).fetchone()[0]
        validation = connection.execute("SELECT COUNT(*) FROM source WHERE year=2011 AND month BETWEEN ? AND ?", (start, finish)).fetchone()[0]
        folds.append({"fit_end_month": end, "fit_rows": fit, "validation_rows": validation})
    receipt = {"status": "PASS", "all_source_values_exact": True, "source_rows": len(source), "sql_year_count_total_and_weeks": year_counts, "forward_folds": folds, "missing_hours_retained": len(missing), "landing_page_mismatch": "17389 listed versus17379 actual hourly CSV rows; no invented repair", "target_component_identity_all_rows": True}
    (ROOT / "results/INDEPENDENT_SOURCE_CHECK.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)

if __name__ == "__main__": main()
