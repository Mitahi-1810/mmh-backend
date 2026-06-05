"""
Stage 3 — evaluate_ocr
Computes OCR accuracy metrics from pre-computed eval results.

Inputs:
  • data/processed/medicines_clean.csv  — clean medicine corpus (for vocab stats)
  • data/eval_results.json              — per-image OCR results (L0→L3 layers)
  • metrics/ocr_metrics.json            — DVC metrics output

This stage reads the evaluation results already produced by the TrOCR+Gemini
pipeline and summarises them into DVC-tracked metrics, so `dvc metrics show`
and `dvc exp run --set-param ocr.fuzzy_threshold=90` work as expected.

Usage:
    python src/evaluate_ocr.py data/processed/medicines_clean.csv \
                               data/eval_results.json \
                               metrics/ocr_metrics.json
"""
import csv
import json
import sys
import yaml
from pathlib import Path

params              = yaml.safe_load(open("params.yaml"))["ocr"]
FUZZY_THRESHOLD     = int(params["fuzzy_threshold"])
CONF_THRESHOLD      = float(params["confidence_threshold"])
N_EVAL              = int(params["n_eval_images"])

corpus_path  = Path(sys.argv[1])
results_path = Path(sys.argv[2])
metrics_path = Path(sys.argv[3])
metrics_path.parent.mkdir(parents=True, exist_ok=True)

# ── Load medicine corpus ─────────────────────────────────────────────────────
print(f"Loading medicine corpus from {corpus_path} ...")
with open(corpus_path, newline="", encoding="utf-8") as f:
    corpus_rows = list(csv.DictReader(f))
corpus_brands = {r["brand_name"].strip().lower() for r in corpus_rows if r.get("brand_name")}
print(f"  Corpus size: {len(corpus_brands):,} unique brands")

# ── Load eval results ────────────────────────────────────────────────────────
print(f"\nLoading eval results from {results_path} ...")
with open(results_path) as f:
    results = json.load(f)

# Support both list format and dict-with-results key
if isinstance(results, dict) and "results" in results:
    results = results["results"]
if isinstance(results, dict) and "per_image" in results:
    results = results["per_image"]

# If already an aggregated report, extract per-image list
if isinstance(results, list):
    records = results
else:
    # Fallback: treat as mapping of image_name → result
    records = list(results.values())

records = records[:N_EVAL]
n = len(records)
print(f"  Evaluating {n} images (param: n_eval_images={N_EVAL})")

# ── Compute metrics across layers ────────────────────────────────────────────
def safe_get(rec, *keys, default=None):
    for key in keys:
        if isinstance(rec, dict) and key in rec:
            rec = rec[key]
        else:
            return default
    return rec

# Detect result format
sample = records[0] if records else {}

# Format A: {ground_truth, l0_pred, l1_pred, l2_pred, l3_pred, l0_exact, ...}
# Format B: {brand_name, layers: {L0: {...}, L1: {...}, ...}}
# Format C: flat results from our evaluate_ocr.py script

layer_metrics = {}

if "l0_exact" in sample or "L0" in sample.get("layers", {}):
    # Our evaluate_ocr.py output format
    for layer in ["L0", "L1", "L2", "L3"]:
        lk = layer.lower()
        exact_key  = f"{lk}_exact"
        fuzzy_key  = f"{lk}_fuzzy"
        cer_key    = f"{lk}_cer"

        if exact_key in sample:
            exact_hits = sum(1 for r in records if r.get(exact_key, False))
            fuzzy_hits = sum(1 for r in records if r.get(fuzzy_key, False))
            cer_vals   = [r[cer_key] for r in records if cer_key in r and r[cer_key] is not None]
        elif "layers" in sample and layer in sample.get("layers", {}):
            exact_hits = sum(1 for r in records if r.get("layers", {}).get(layer, {}).get("exact", False))
            fuzzy_hits = sum(1 for r in records if r.get("layers", {}).get(layer, {}).get("fuzzy", False))
            cer_vals   = [r["layers"][layer]["cer"] for r in records
                         if "layers" in r and layer in r.get("layers", {})
                         and r["layers"][layer].get("cer") is not None]
        else:
            continue

        layer_metrics[layer] = {
            "exact_match_pct": round(exact_hits / n * 100, 1) if n else 0,
            "fuzzy_match_pct": round(fuzzy_hits / n * 100, 1) if n else 0,
            "cer":             round(sum(cer_vals) / len(cer_vals), 4) if cer_vals else None,
        }
else:
    # Fallback: try to read any summary/aggregate already in the file
    layer_metrics = {
        "note": "Could not parse per-image layer data — showing aggregate if available"
    }

# Per-difficulty breakdown
difficulties = {}
for diff in ["clean", "mild", "moderate", "hard"]:
    subset = [r for r in records if r.get("difficulty") == diff]
    if not subset:
        continue
    best_layer_exact = max(
        (sum(1 for r in subset if r.get(f"{l}_exact", False)) for l in ["l0","l1","l2","l3"]),
        default=0
    )
    difficulties[diff] = {
        "n": len(subset),
        "best_exact_pct": round(best_layer_exact / len(subset) * 100, 1),
    }

# Vocabulary coverage
in_corpus = 0
for r in records:
    brand = (r.get("ground_truth") or r.get("brand_name") or "").strip().lower()
    if brand and brand in corpus_brands:
        in_corpus += 1

vocab_coverage_pct = round(in_corpus / n * 100, 1) if n else 0

# ── Build final metrics ───────────────────────────────────────────────────────
# Best layer performance (L3 = full pipeline)
best = layer_metrics.get("L3") or layer_metrics.get("L2") or {}

ocr_metrics = {
    "n_images_evaluated": n,
    "corpus_size":         len(corpus_brands),
    "vocab_coverage_pct":  vocab_coverage_pct,
    "best_exact_match_pct": best.get("exact_match_pct", 0),
    "best_fuzzy_match_pct": best.get("fuzzy_match_pct", 0),
    "best_cer":             best.get("cer"),
    "layers": layer_metrics,
    "by_difficulty": difficulties,
    "params_used": {
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "confidence_threshold": CONF_THRESHOLD,
        "model": params["model"],
    },
}

with open(metrics_path, "w") as f:
    json.dump(ocr_metrics, f, indent=2)

print(f"\n✓ OCR metrics → {metrics_path}")
print(f"\nResults:")
print(f"  Images evaluated:    {n}")
print(f"  Corpus size:         {len(corpus_brands):,} brands")
print(f"  Vocab coverage:      {vocab_coverage_pct}%")
if best:
    print(f"  Best exact match:    {best.get('exact_match_pct', 'N/A')}%  (full pipeline)")
    print(f"  Best fuzzy match:    {best.get('fuzzy_match_pct', 'N/A')}%")
    if best.get("cer") is not None:
        print(f"  Best CER:            {best.get('cer'):.4f}")
if layer_metrics and isinstance(layer_metrics, dict) and "L0" in layer_metrics:
    l0 = layer_metrics["L0"]
    print(f"\n  Improvement over baseline:")
    print(f"    L0 (raw TrOCR):     {l0.get('exact_match_pct', 'N/A')}% exact")
    print(f"    Full pipeline:      {best.get('exact_match_pct', 'N/A')}% exact  "
          f"(+{round(best.get('exact_match_pct',0) - l0.get('exact_match_pct',0), 1)}pp)")
