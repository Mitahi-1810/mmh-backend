"""
Stage 2 — analyze
Computes coverage statistics from the clean medicine database.
Writes two files:
  • metrics/data_metrics.json  (DVC metrics — key numbers only)
  • data/processed/stats.json  (full breakdown for reporting)

Usage:
    python src/analyze.py data/processed/medicines_clean.csv \
                          data/processed/stats.json \
                          metrics/data_metrics.json
"""
import csv
import json
import sys
import yaml
from collections import Counter
from pathlib import Path

params      = yaml.safe_load(open("params.yaml"))["analyze"]
TOP_N_COS   = int(params["top_n_companies"])
TOP_N_GENS  = int(params["top_n_generics"])

input_path   = Path(sys.argv[1])
stats_path   = Path(sys.argv[2])
metrics_path = Path(sys.argv[3])
stats_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.parent.mkdir(parents=True, exist_ok=True)

def to_float(val):
    try: return float(str(val).strip())
    except: return None

print(f"Analyzing {input_path} ...")
with open(input_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

total = len(rows)

# Counters
companies     = Counter(r.get("manufacturer", "").strip() or r.get("company_name","").strip() for r in rows)
generics      = Counter(r.get("generic_name", "").strip().lower() for r in rows)
dosage_forms  = Counter(r.get("dosage_form", "").strip().lower() for r in rows)
ther_classes  = Counter(r.get("therapeutic_class", "").strip() for r in rows)

# Price stats
prices = [p for r in rows if (p := to_float(r.get("unit_price"))) is not None and p > 0]
avg_price = sum(prices) / len(prices) if prices else 0
min_price = min(prices) if prices else 0
max_price = max(prices) if prices else 0

# Unique generics (non-empty)
unique_generics = len({k for k in generics if k})

print(f"  Total medicines    : {total:,}")
print(f"  Unique companies   : {len(companies):,}")
print(f"  Unique generics    : {unique_generics:,}")
print(f"  Dosage forms       : {len(dosage_forms):,}")
print(f"  Price range        : ৳{min_price:.2f} – ৳{max_price:.2f}  (avg ৳{avg_price:.2f})")

# ── DVC metrics (key numbers the teacher will see with `dvc metrics show`) ───
metrics = {
    "total_medicines":   total,
    "unique_companies":  len(companies),
    "unique_generics":   unique_generics,
    "dosage_forms":      len(dosage_forms),
    "avg_unit_price_bdt": round(avg_price, 2),
    "min_unit_price_bdt": round(min_price, 2),
    "max_unit_price_bdt": round(max_price, 2),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# ── Full stats (used by pipeline_report.py for visualisation) ────────────────
stats = {
    **metrics,
    "top_companies": [
        {"name": name, "count": count}
        for name, count in companies.most_common(TOP_N_COS)
        if name
    ],
    "top_generics": [
        {"name": name, "brand_count": count}
        for name, count in generics.most_common(TOP_N_GENS + 1)
        if name
    ][: TOP_N_GENS],
    "dosage_form_distribution": {k: v for k, v in dosage_forms.most_common(10) if k},
    "top_therapeutic_classes": {k: v for k, v in ther_classes.most_common(15) if k},
    "price_buckets": {
        "under_10":   sum(1 for p in prices if p < 10),
        "10_to_50":   sum(1 for p in prices if 10 <= p < 50),
        "50_to_200":  sum(1 for p in prices if 50 <= p < 200),
        "over_200":   sum(1 for p in prices if p >= 200),
    },
}

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n✓ Metrics → {metrics_path}")
print(f"✓ Full stats → {stats_path}")
print("\nKey metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
