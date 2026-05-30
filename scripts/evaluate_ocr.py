"""
Medicine-Aware OCR Pipeline — Accuracy Evaluator
=================================================
Measures the accuracy improvement from each pipeline layer.

LAYER DEFINITIONS
-----------------
Layer 0  Raw TrOCR (no medicine awareness)
Layer 1  Layer 0 + in-memory fuzzy post-correction using medicine names from CSV
Layer 2  TrOCR + MedicineVocabLogitsProcessor (constrained beam search)
Layer 3  Layer 2 + strength-based disambiguation from CSV (no DB needed)

METRICS
-------
  brand_exact   Brand name exact match (case-insensitive, normalised)
  brand_fuzzy   Brand name fuzzy match: WRatio ≥ 85 (acceptable read)
  cer           Character Error Rate on the brand-name token
  speed_s       Average seconds per image

MODES
-----
  --synthetic   Use generated test images from data/synthetic_eval/
  --real        Use real prescription images; requires --images-dir + --ground-truth

Usage
-----
# Run on synthetic data (no real images needed)
python3 scripts/evaluate_ocr.py --synthetic

# Run on real images with ground truth
python3 scripts/evaluate_ocr.py --real \\
    --images-dir /path/to/rx_photos/ \\
    --ground-truth /path/to/ground_truth.json

Ground-truth JSON format for --real mode:
[
  {"image": "rx001.jpg", "brand_name": "Napa"},
  {"image": "rx002.jpg", "brand_name": "Seclo"},
  ...
]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz import fuzz, process as rf_process
from rapidfuzz.distance import Levenshtein

log = logging.getLogger("eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

BASE_DIR      = Path(__file__).parent.parent
EVAL_DIR      = BASE_DIR / "data" / "synthetic_eval"
MEDICINES_CSV = BASE_DIR / "data" / "medicines_top20.csv"

# ── Medicine corpus (loaded from CSV, no DB needed) ───────────────────────────

class MedicineCorpus:
    """In-memory medicine database built from the scraped CSV."""

    def __init__(self, csv_path: Path):
        self.brands: list[str] = []           # canonical brand names
        self.brand_lower: list[str] = []      # lower-cased (for fuzzy match)
        self.strength_map: dict[str, list[str]] = {}  # brand_lower → [strengths]
        self._load(csv_path)

    def _load(self, csv_path: Path):
        if not csv_path.exists():
            log.warning("CSV not found: %s", csv_path)
            return
        seen: set[str] = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand = (row.get("brand_name") or "").strip()
                strength = (row.get("strength") or "").strip()
                if not brand:
                    continue
                key = brand.lower()
                if key not in seen:
                    seen.add(key)
                    self.brands.append(brand)
                    self.brand_lower.append(key)
                if strength:
                    self.strength_map.setdefault(key, []).append(strength.lower())
        log.info("MedicineCorpus: %d unique brands loaded", len(self.brands))

    def fuzzy_correct(self, ocr_name: str, cutoff: float = 70.0) -> str | None:
        """Return closest known brand name (WRatio ≥ cutoff), or None."""
        if not ocr_name or not self.brand_lower:
            return None
        result = rf_process.extractOne(
            ocr_name.lower().strip(),
            self.brand_lower,
            scorer=fuzz.WRatio,
            score_cutoff=cutoff,
        )
        if result:
            return self.brands[result[2]]
        return None

    def disambiguate(self, brand: str, strength: str | None) -> str:
        """
        If `brand` is in the corpus but `strength` doesn't match its known strengths,
        look for a brand with a similar name that DOES have this strength.
        Returns the corrected brand name, or the original if no better match.
        """
        if not strength:
            return brand

        # Check if exact brand+strength matches
        key = brand.lower()
        num = "".join(c for c in strength if c.isdigit())
        if key in self.strength_map:
            for s in self.strength_map[key]:
                if num and num in s:
                    return brand  # Already correct

        # Try fuzzy neighbours with matching strength
        candidates = rf_process.extract(
            key, self.brand_lower,
            scorer=fuzz.WRatio,
            score_cutoff=72,
            limit=5,
        )
        for cand_key, score, idx in candidates:
            cand_strengths = self.strength_map.get(cand_key, [])
            if num and any(num in s for s in cand_strengths):
                return self.brands[idx]

        return brand


# ── TrOCR model interface (reuse ocr_engine internals) ────────────────────────

_trocr_ready = False
_processor   = None
_model       = None
_vocab_proc  = None   # Layer 2 logits processor


_device = "cpu"  # Set after torch import


async def _load_model(corpus: MedicineCorpus):
    global _processor, _model, _vocab_proc, _trocr_ready, _device
    if _trocr_ready:
        return

    log.info("Loading TrOCR model (microsoft/trocr-large-handwritten)…")
    t0 = time.monotonic()

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        _device = "mps"
    elif torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"
    log.info("  Inference device: %s", _device)

    _processor = await asyncio.to_thread(
        TrOCRProcessor.from_pretrained, "microsoft/trocr-large-handwritten"
    )
    _model = await asyncio.to_thread(
        VisionEncoderDecoderModel.from_pretrained, "microsoft/trocr-large-handwritten"
    )
    _model = _model.to(_device)
    _model.eval()

    # Build vocab logits processor from corpus
    try:
        from transformers import LogitsProcessorList
        # Import from project code if available, otherwise inline
        sys.path.insert(0, str(BASE_DIR))
        try:
            from app.services.medicine_vocab import MedicineVocabLogitsProcessor
        except ImportError:
            # Inline fallback — same logic as medicine_vocab.py
            from transformers import LogitsProcessor

            def _build_trie(names, tokenizer):
                trie = {}
                for name in names:
                    try:
                        ids = tokenizer(name, add_special_tokens=False).input_ids
                    except Exception:
                        continue
                    node = trie
                    for tid in ids:
                        node = node.setdefault(tid, {})
                    node["__end__"] = True
                return trie

            class MedicineVocabLogitsProcessor(LogitsProcessor):
                def __init__(self, names, tokenizer, boost=3.0):
                    self.trie = _build_trie(names, tokenizer)
                    self.boost = boost
                    self.bos_id = getattr(tokenizer, "bos_token_id", 0) or 0
                    self.pad_id = getattr(tokenizer, "pad_token_id", 1) or 1

                def __call__(self, input_ids, scores):
                    for bi in range(input_ids.shape[0]):
                        seq = input_ids[bi].tolist()
                        eff = [t for t in seq if t not in (self.bos_id, self.pad_id) and t > 3]
                        if len(eff) >= 10:
                            continue
                        node, on_path = self.trie, True
                        for tok in eff:
                            if tok in node:
                                node = node[tok]
                            else:
                                on_path = False
                                break
                        if not on_path or not isinstance(node, dict):
                            continue
                        for nxt, _ in node.items():
                            if nxt != "__end__" and nxt < scores.shape[-1]:
                                scores[bi, nxt] += self.boost
                    return scores

        _vocab_proc = MedicineVocabLogitsProcessor(
            corpus.brands, _processor.tokenizer, boost=3.0
        )
        log.info("  MedicineVocabLogitsProcessor: %d brand names in trie", len(corpus.brands))
    except Exception as exc:
        log.warning("  Vocab processor build failed: %s — Layer 2 will equal Layer 0", exc)
        _vocab_proc = None

    elapsed = time.monotonic() - t0
    log.info("TrOCR loaded in %.1fs", elapsed)
    _trocr_ready = True


def _infer_line(img: Image.Image, use_vocab: bool) -> tuple[str, float]:
    """
    Run TrOCR on a single-line image.
    use_vocab=True → Layer 2 (constrained decoding)
    use_vocab=False → Layer 0 (raw TrOCR)
    """
    import torch
    from transformers import LogitsProcessorList

    pv = _processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(_device)

    kwargs: dict = dict(
        num_beams=4,
        max_new_tokens=64,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if use_vocab and _vocab_proc is not None:
        kwargs["logits_processor"] = LogitsProcessorList([_vocab_proc])

    with torch.no_grad():
        out = _model.generate(pv, **kwargs)

    text = _processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
        lp = out.sequences_scores[0].item()
        conf = float(min(1.0, max(0.0, 1.0 + lp / 10.0)))
    else:
        conf = 0.6
    return text, conf


# ── Brand name extraction ──────────────────────────────────────────────────────

import re
_BRAND_RE = re.compile(r'^([A-Z][a-zA-Z0-9\-]+)')


def _extract_brand(text: str) -> str:
    """Pull the first capitalised token (brand name position)."""
    m = _BRAND_RE.match(text.strip())
    return m.group(1) if m else text.split()[0] if text.split() else text


def _extract_strength(text: str) -> str | None:
    m = re.search(r'(\d+\.?\d*)\s*(mg|ml|mcg|g\b|iu)', text, re.I)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def brand_cer(predicted: str, ground_truth: str) -> float:
    """Character Error Rate on brand name (0.0 = perfect)."""
    p, g = predicted.lower().strip(), ground_truth.lower().strip()
    if not g:
        return 0.0
    return Levenshtein.normalized_distance(p, g)


def brand_exact(predicted: str, ground_truth: str) -> bool:
    return predicted.lower().strip() == ground_truth.lower().strip()


def brand_fuzzy_ok(predicted: str, ground_truth: str, threshold: float = 85.0) -> bool:
    return fuzz.WRatio(predicted.lower(), ground_truth.lower()) >= threshold


# ── Evaluation loop ───────────────────────────────────────────────────────────

async def run_evaluation(
    test_cases: list[dict],
    images_dir: Path,
    corpus: MedicineCorpus,
) -> dict:
    """
    Run all 4 layers on every test case.
    Returns aggregated and per-case results.
    """
    await _load_model(corpus)

    results: list[dict] = []
    n = len(test_cases)
    log.info("Evaluating %d test cases…", n)

    for i, tc in enumerate(test_cases):
        img_path = images_dir / tc["image"]
        if not img_path.exists():
            log.warning("Image not found: %s", img_path)
            continue

        gt_brand  = tc["brand_name"]
        difficulty = tc.get("difficulty", "unknown")
        img = Image.open(img_path).convert("RGB")

        t0 = time.monotonic()

        # ── Layer 0: raw TrOCR ──────────────────────────────────────────────
        l0_text, l0_conf = await asyncio.to_thread(_infer_line, img, False)
        l0_brand = _extract_brand(l0_text)

        # ── Layer 1: Layer 0 + fuzzy correction ─────────────────────────────
        l1_brand_raw = corpus.fuzzy_correct(l0_brand)
        l1_brand = l1_brand_raw if l1_brand_raw else l0_brand

        # ── Layer 2: TrOCR + vocab logits processor ──────────────────────────
        l2_text, l2_conf = await asyncio.to_thread(_infer_line, img, True)
        l2_brand = _extract_brand(l2_text)

        # ── Layer 3: Layer 2 + strength disambiguation ───────────────────────
        strength = _extract_strength(l2_text) or _extract_strength(tc.get("full_line", ""))
        l3_brand = corpus.disambiguate(l2_brand, strength)

        elapsed = time.monotonic() - t0

        rec = {
            "image":      tc["image"],
            "ground_truth": gt_brand,
            "difficulty": difficulty,
            "font":       tc.get("font", ""),
            "l0_brand":   l0_brand,  "l0_conf": round(l0_conf, 3),
            "l1_brand":   l1_brand,
            "l2_brand":   l2_brand,  "l2_conf": round(l2_conf, 3),
            "l3_brand":   l3_brand,
            # Per-layer correct flags
            "l0_exact": brand_exact(l0_brand, gt_brand),
            "l1_exact": brand_exact(l1_brand, gt_brand),
            "l2_exact": brand_exact(l2_brand, gt_brand),
            "l3_exact": brand_exact(l3_brand, gt_brand),
            "l0_fuzzy": brand_fuzzy_ok(l0_brand, gt_brand),
            "l1_fuzzy": brand_fuzzy_ok(l1_brand, gt_brand),
            "l2_fuzzy": brand_fuzzy_ok(l2_brand, gt_brand),
            "l3_fuzzy": brand_fuzzy_ok(l3_brand, gt_brand),
            "l0_cer": round(brand_cer(l0_brand, gt_brand), 4),
            "l2_cer": round(brand_cer(l2_brand, gt_brand), 4),
            "l3_cer": round(brand_cer(l3_brand, gt_brand), 4),
            "elapsed_s": round(elapsed, 2),
        }
        results.append(rec)

        if (i + 1) % 20 == 0 or (i + 1) == n:
            # Live accuracy progress
            done = results
            l0e  = sum(r["l0_exact"] for r in done) / len(done) * 100
            l2e  = sum(r["l2_exact"] for r in done) / len(done) * 100
            l3e  = sum(r["l3_exact"] for r in done) / len(done) * 100
            log.info(
                "[%d/%d] L0=%.1f%% L2=%.1f%% L3=%.1f%%",
                i + 1, n, l0e, l2e, l3e,
            )

    return {"results": results, "n": len(results)}


# ── Reporting ──────────────────────────────────────────────────────────────────

def _agg(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def print_report(data: dict):
    results = data["results"]
    n = len(results)
    if n == 0:
        print("No results.")
        return

    sep = "─" * 70
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        Medicine-Aware OCR Pipeline — Accuracy Report           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Test cases : {n}")
    avg_s = sum(r["elapsed_s"] for r in results) / n
    print(f"  Avg speed  : {avg_s:.2f}s / image")
    print()

    # Overall table
    print(f"  {'Layer':<40} {'Exact':>8} {'Fuzzy':>8} {'CER':>8}")
    print(f"  {sep[:64]}")

    layers = [
        ("Layer 0  Raw TrOCR (no medicine awareness)",     "l0_exact", "l0_fuzzy", "l0_cer"),
        ("Layer 1  L0 + Fuzzy post-correction",            "l1_exact", "l1_fuzzy", None),
        ("Layer 2  TrOCR + Vocab constrained decoding",    "l2_exact", "l2_fuzzy", "l2_cer"),
        ("Layer 3  L2 + Strength disambiguation (full)",   "l3_exact", "l3_fuzzy", "l3_cer"),
    ]

    for label, ek, fk, ck in layers:
        exact = _agg(results, ek)
        fuzzy = _agg(results, fk)
        cer_v = _agg(results, ck) / 100 if ck else None  # _agg returns %, cer is fraction
        cer_str = f"{cer_v:.4f}" if cer_v is not None else "  —   "
        print(f"  {label:<40} {exact:>7.1f}% {fuzzy:>7.1f}% {cer_str:>8}")

    print()
    print("  Improvement from medicine-awareness:")
    l0_e = _agg(results, "l0_exact")
    l2_e = _agg(results, "l2_exact")
    l3_e = _agg(results, "l3_exact")
    l0_f = _agg(results, "l0_fuzzy")
    l3_f = _agg(results, "l3_fuzzy")
    print(f"    Exact match:  Layer 0 → Full pipeline  {l0_e:.1f}% → {l3_e:.1f}%  "
          f"(+{l3_e - l0_e:.1f} pp)")
    print(f"    Fuzzy match:  Layer 0 → Full pipeline  {l0_f:.1f}% → {l3_f:.1f}%  "
          f"(+{l3_f - l0_f:.1f} pp)")
    print()

    # Per-difficulty breakdown
    diffs = sorted(set(r["difficulty"] for r in results))
    if len(diffs) > 1:
        print(f"  {'Difficulty':<12} {'N':>5} {'L0 Exact':>10} {'L2 Exact':>10} {'L3 Exact':>10}")
        print(f"  {sep[:55]}")
        for d in diffs:
            sub = [r for r in results if r["difficulty"] == d]
            print(f"  {d:<12} {len(sub):>5} "
                  f"{_agg(sub, 'l0_exact'):>9.1f}% "
                  f"{_agg(sub, 'l2_exact'):>9.1f}% "
                  f"{_agg(sub, 'l3_exact'):>9.1f}%")
        print()

    # Error analysis — cases Layer 0 got wrong but Layer 2 got right
    l2_saves = [r for r in results if not r["l0_exact"] and r["l2_exact"]]
    l2_hurts = [r for r in results if r["l0_exact"] and not r["l2_exact"]]

    print(f"  Vocab constrained decoding:")
    print(f"    Corrected by Layer 2  : {len(l2_saves)} cases")
    print(f"    Broken by  Layer 2    : {len(l2_hurts)} cases")
    if l2_saves:
        ex = l2_saves[:3]
        print(f"    Examples corrected:")
        for r in ex:
            print(f"      GT: {r['ground_truth']:<20} L0: {r['l0_brand']:<20} → L2: {r['l2_brand']}")

    print()
    print(f"  ⚠  Note: synthetic images ≠ real handwriting.")
    print(f"     Provide real prescription photos for production-grade accuracy numbers.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args):
    corpus = MedicineCorpus(MEDICINES_CSV)

    if args.synthetic:
        gt_path = EVAL_DIR / "ground_truth.json"
        if not gt_path.exists():
            log.info("Synthetic data not found — generating %d images…", args.n_eval)
            # Run generator inline
            gen_script = BASE_DIR / "scripts" / "generate_synthetic_data.py"
            import subprocess
            result = subprocess.run(
                [sys.executable, str(gen_script),
                 "--mode", "eval", "--n-eval", str(args.n_eval)],
                capture_output=False,
            )
            if result.returncode != 0:
                sys.exit("Data generation failed.")

        with open(gt_path) as f:
            test_cases = json.load(f)

        images_dir = EVAL_DIR / "images"

    elif args.real:
        if not args.ground_truth or not args.images_dir:
            sys.exit("--real requires --ground-truth and --images-dir")
        with open(args.ground_truth) as f:
            test_cases = json.load(f)
        images_dir = Path(args.images_dir)

    else:
        sys.exit("Specify --synthetic or --real")

    # Sub-sample if requested
    if args.limit and args.limit < len(test_cases):
        import random
        random.seed(42)
        test_cases = random.sample(test_cases, args.limit)
        log.info("Sub-sampled to %d test cases", len(test_cases))

    data = await run_evaluation(test_cases, images_dir, corpus)

    # Save raw results
    out_path = BASE_DIR / "data" / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Raw results saved to %s", out_path)

    print_report(data)


def main():
    ap = argparse.ArgumentParser(description="Evaluate medicine-aware OCR pipeline")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true",
                      help="Evaluate on synthetic test images")
    mode.add_argument("--real",      action="store_true",
                      help="Evaluate on real prescription images")
    ap.add_argument("--images-dir",    help="Directory containing prescription images (--real mode)")
    ap.add_argument("--ground-truth",  help="Path to ground_truth.json (--real mode)")
    ap.add_argument("--n-eval", type=int, default=200,
                    help="Number of synthetic images to generate if not already present")
    ap.add_argument("--limit",  type=int, default=None,
                    help="Limit evaluation to N test cases (faster)")

    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
