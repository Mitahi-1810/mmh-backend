"""
Stage 1 — prepare
Reads the raw merged medicines CSV, cleans it, deduplicates, and writes
a clean CSV ready for analysis and OCR evaluation.

Usage:
    python src/prepare_data.py data/raw/medicines_v2.csv data/processed/medicines_clean.csv
"""
import csv
import sys
import yaml
from pathlib import Path
from collections import defaultdict

params = yaml.safe_load(open("params.yaml"))["prepare"]
MIN_PRICE       = float(params["min_price"])
DEDUP_STRATEGY  = params["dedup_strategy"]   # keep_first | keep_cheapest

input_path  = Path(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Reading {input_path} ...")
with open(input_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    raw_rows = list(reader)
    fieldnames = reader.fieldnames

total_raw = len(raw_rows)
print(f"  Raw rows: {total_raw:,}")

# ── 1. Drop rows missing brand_name or generic_name ──────────────────────────
rows = [r for r in raw_rows
        if r.get("brand_name", "").strip() and r.get("generic_name", "").strip()]
after_missing = len(rows)
print(f"  After dropping missing brand/generic: {after_missing:,} "
      f"(removed {total_raw - after_missing:,})")

# ── 2. Drop rows with price below minimum ────────────────────────────────────
def to_float(val):
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None

rows_price_ok = []
for r in rows:
    price = to_float(r.get("unit_price"))
    if price is None or price >= MIN_PRICE:
        rows_price_ok.append(r)

after_price = len(rows_price_ok)
print(f"  After price filter (≥ ৳{MIN_PRICE}): {after_price:,} "
      f"(removed {after_missing - after_price:,})")

# ── 3. Deduplicate by (brand_name, strength, dosage_form) ───────────────────
seen: dict[tuple, list] = defaultdict(list)
for r in rows_price_ok:
    key = (
        r.get("brand_name", "").strip().lower(),
        r.get("strength", "").strip().lower(),
        r.get("dosage_form", "").strip().lower(),
    )
    seen[key].append(r)

deduped = []
for key, group in seen.items():
    if DEDUP_STRATEGY == "keep_cheapest":
        group.sort(key=lambda r: to_float(r.get("unit_price")) or float("inf"))
    deduped.append(group[0])

after_dedup = len(deduped)
print(f"  After dedup ({DEDUP_STRATEGY}): {after_dedup:,} "
      f"(removed {after_price - after_dedup:,} duplicates)")

# ── 4. Write clean CSV ───────────────────────────────────────────────────────
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped)

print(f"\n✓ Wrote {after_dedup:,} clean medicines → {output_path}")
print(f"  Cleaning removed {total_raw - after_dedup:,} rows "
      f"({(total_raw - after_dedup)/total_raw*100:.1f}%)")
