"""
Synthetic Prescription Image Generator
=======================================
Creates labeled prescription-line images from the BD medicine database.
Used for:
  (a) Evaluation: hold-out test set to measure layer-by-layer accuracy
  (b) Fine-tuning: training set for TrOCR adaptation (upload to Colab)

For each medicine name we render it as a prescription line in several fonts
and augmentation levels, simulating the range of doctor handwriting quality.

Usage
-----
# Generate evaluation test set only (fast, ~200 images)
python3 scripts/generate_synthetic_data.py --mode eval

# Generate full training + eval set
python3 scripts/generate_synthetic_data.py --mode both

# Custom count
python3 scripts/generate_synthetic_data.py --mode both --n-train 5000 --n-eval 300

Output
------
data/synthetic_eval/
  images/           ← PNG images, one per test case
  ground_truth.json ← [{image, brand_name, full_line, strength, difficulty}, ...]

data/synthetic_train/
  images/           ← training images
  labels.csv        ← image_path, text (HuggingFace dataset format)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

log = logging.getLogger("synth")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
EVAL_DIR    = DATA_DIR / "synthetic_eval"
TRAIN_DIR   = DATA_DIR / "synthetic_train"
MEDICINES_CSV = DATA_DIR / "medicines_top20.csv"

# ── Fonts available on macOS (fallback chain) ──────────────────────────────────
FONT_PATHS = {
    "bradley_hand": "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "brush_script":  "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "comic_sans":    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "georgia":       "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "courier":       "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "arial":         "/Library/Fonts/Arial.ttf",
    "arial_italic":  "/Library/Fonts/Arial Italic.ttf",
}

FREQ_TOKENS   = ["BD", "TDS", "OD", "BID", "TID", "1+0+1", "0+0+1", "1+1+1"]
TIMING_TOKENS = ["AC", "PC", "after meals", "HS", ""]
DURATION_TMPL = ["× {} days", "for {} days", "× {} wks", "{} days", ""]

DIFFICULTIES = ["clean", "mild", "moderate", "hard"]


# ── Image rendering ────────────────────────────────────────────────────────────

def _load_font(font_key: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_PATHS.get(font_key)
    if path and os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # Try any TTF in the dict
    for p in FONT_PATHS.values():
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _augment(img: Image.Image, difficulty: str) -> Image.Image:
    """Apply progressive augmentation based on difficulty level."""
    if difficulty == "clean":
        return img

    arr = np.array(img, dtype=np.float32)

    # Gaussian noise
    std = {"mild": 8, "moderate": 20, "hard": 40}[difficulty]
    arr = arr + np.random.randn(*arr.shape) * std
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Blur
    radius = {"mild": 0.5, "moderate": 1.2, "hard": 2.0}[difficulty]
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Rotation
    max_angle = {"mild": 2, "moderate": 5, "hard": 8}[difficulty]
    angle = random.uniform(-max_angle, max_angle)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    return img


def render_line_image(
    text: str,
    font_key: str,
    font_size: int,
    difficulty: str,
    canvas_w: int = 600,
) -> Image.Image:
    """
    Render a text string as a prescription-line image.

    Returns an RGB PIL image with white background.
    """
    font = _load_font(font_key, font_size)

    # Measure text to set canvas height
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 20, 12
    w = max(canvas_w, text_w + 2 * pad_x)
    h = text_h + 2 * pad_y

    img  = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Slight ink-colour variation (dark grey instead of pure black, occasionally)
    ink = random.choice([(0, 0, 0), (10, 10, 30), (20, 15, 10), (5, 5, 5)])
    draw.text((pad_x, pad_y), text, font=font, fill=ink)

    return _augment(img, difficulty)


# ── Prescription-line text builder ────────────────────────────────────────────

def build_prescription_line(brand: str, strength: str) -> str:
    """Produce a realistic handwritten prescription line."""
    freq = random.choice(FREQ_TOKENS)
    timing = random.choice(TIMING_TOKENS)
    dur_tmpl = random.choice(DURATION_TMPL)
    n_days = random.choice([3, 5, 7, 10, 14])

    parts = [brand]
    if strength:
        parts.append(strength)
    parts.append(freq)
    if timing:
        parts.append(timing)
    if dur_tmpl:
        parts.append(dur_tmpl.format(n_days))

    return " ".join(parts)


# ── Medicine loader ────────────────────────────────────────────────────────────

def load_medicines(csv_path: Path) -> list[dict]:
    """
    Load unique (brand_name, strength) pairs from the medicines CSV.
    Returns list of dicts with: brand_name, strength, generic_name.
    """
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return []

    seen: set[str] = set()
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = (row.get("brand_name") or "").strip()
            strength = (row.get("strength") or "").strip()
            generic = (row.get("generic_name") or "").strip()
            if not brand or brand in seen:
                continue
            seen.add(brand)
            rows.append({"brand_name": brand, "strength": strength, "generic_name": generic})

    log.info("Loaded %d unique brand names from %s", len(rows), csv_path.name)
    return rows


# ── Generation ─────────────────────────────────────────────────────────────────

FONT_POOL = ["bradley_hand", "comic_sans", "arial_italic", "arial", "courier"]
FONT_POOL = [f for f in FONT_POOL if os.path.exists(FONT_PATHS.get(f, ""))]

if not FONT_POOL:
    FONT_POOL = ["arial"]  # ultimate fallback


def generate_dataset(
    medicines: list[dict],
    out_dir: Path,
    n_images: int,
    difficulties: list[str],
    seed: int = 42,
) -> list[dict]:
    """
    Generate n_images synthetic prescription line images.

    Returns list of ground-truth records:
      {image_path, brand_name, full_line, strength, difficulty, font}
    """
    random.seed(seed)
    np.random.seed(seed)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    gt: list[dict] = []
    font_sizes = [26, 28, 30, 32, 34]

    for i in range(n_images):
        med = random.choice(medicines)
        brand    = med["brand_name"]
        strength = med["strength"]
        diff     = random.choice(difficulties)
        font_key = random.choice(FONT_POOL)
        font_sz  = random.choice(font_sizes)

        line_text = build_prescription_line(brand, strength)
        try:
            img = render_line_image(line_text, font_key, font_sz, diff)
        except Exception as exc:
            log.warning("Render error for %r: %s", line_text, exc)
            continue

        img_name = f"img_{i:05d}.png"
        img_path = out_dir / "images" / img_name
        img.save(img_path)

        gt.append({
            "image":      img_name,
            "brand_name": brand,
            "full_line":  line_text,
            "strength":   strength,
            "difficulty": diff,
            "font":       font_key,
        })

        if (i + 1) % 100 == 0:
            log.info("  Generated %d / %d images", i + 1, n_images)

    # Save ground truth
    gt_path = out_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    log.info("Saved %d images to %s", len(gt), out_dir)
    return gt


def generate_train_csv(medicines: list[dict], out_dir: Path, n_images: int):
    """Generate training data with labels.csv in HuggingFace TrOCR format."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_images):
        med      = random.choice(medicines)
        brand    = med["brand_name"]
        strength = med["strength"]
        diff     = random.choice(["mild", "moderate", "hard"])
        font_key = random.choice(FONT_POOL)
        font_sz  = random.choice([24, 28, 32, 36])

        line_text = build_prescription_line(brand, strength)
        try:
            img = render_line_image(line_text, font_key, font_sz, diff)
        except Exception:
            continue

        img_name = f"train_{i:06d}.png"
        img.save(out_dir / "images" / img_name)
        rows.append({"file_name": f"images/{img_name}", "text": line_text})

        if (i + 1) % 500 == 0:
            log.info("  Train: %d / %d", i + 1, n_images)

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text"])
        w.writeheader()
        w.writerows(rows)

    log.info("Training set: %d images → %s", len(rows), labels_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic prescription images")
    ap.add_argument("--mode",    choices=["eval", "train", "both"], default="eval")
    ap.add_argument("--n-eval",  type=int, default=300,  help="Number of eval images")
    ap.add_argument("--n-train", type=int, default=5000, help="Number of training images")
    ap.add_argument("--csv",     default=str(MEDICINES_CSV), help="Medicines CSV path")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    meds = load_medicines(Path(args.csv))
    if not meds:
        sys.exit("No medicines loaded — check CSV path")

    if args.mode in ("eval", "both"):
        log.info("Generating EVAL set (%d images)…", args.n_eval)
        generate_dataset(
            meds, EVAL_DIR,
            n_images=args.n_eval,
            difficulties=DIFFICULTIES,
            seed=args.seed,
        )

    if args.mode in ("train", "both"):
        log.info("Generating TRAIN set (%d images)…", args.n_train)
        generate_train_csv(meds, TRAIN_DIR, args.n_train)

    log.info("Done.")


if __name__ == "__main__":
    main()
