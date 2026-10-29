#!/usr/bin/env python3
"""
Фильтрует abuseipdb_blacklist.csv: оставляет только CN и TW,
затем прореживает результат так, чтобы между соседними записями
(по lastReportedAt) было не меньше 1 минуты.
"""

import csv
import sys
from datetime import datetime, timedelta

def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def filter_and_thin(input_path: str, output_path: str, countries=("CN", "TW", 'RU', 'US', 'UK'), interval_seconds=1):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("countryCode") in countries]

    rows.sort(key=lambda r: parse_ts(r["lastReportedAt"]))

    kept = []
    last_kept_ts = None
    for row in rows:
        ts = parse_ts(row["lastReportedAt"])
        if last_kept_ts is None or (ts - last_kept_ts) >= timedelta(seconds=interval_seconds):
            kept.append(row)
            last_kept_ts = ts

    if not kept:
        print("Пусто. Либо Китай с Тайванем взяли выходной, либо ты дал не тот файл.")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kept[0].keys())
        writer.writeheader()
        writer.writerows(kept)

    print(f"CN/TW всего: {len(rows)} -> после прореживания в {interval_seconds} сек: {len(kept)}")
    print(f"Результат: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python filter_abuseipdb.py input.csv [output.csv] [interval_seconds]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "filtered_cn_tw.csv"
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    filter_and_thin(input_csv, output_csv, interval_seconds=interval)