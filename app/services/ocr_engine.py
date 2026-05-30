"""
Medicine-Aware OCR Engine — V2  (TrOCR backbone)

Model: microsoft/trocr-large-handwritten
  • Pre-trained on IAM handwriting dataset: 2.89% CER (~97% character accuracy)
  • Open-weight, runs on CPU (~2-4s per prescription), no API cost
  • Fine-tunable later on BD prescription data for ≥99% on medicine names

Pipeline stages:
  1. Preprocess  — deskew, denoise, CLAHE contrast, upscale if needed
  2. Segment     — OpenCV row-projection histogram → individual text line crops
  3. HTR         — TrOCR-large per line + MedicineVocabLogitsProcessor
                   (constrained beam search prefers known BD medicine names)
  4. NER         — Rule-based: brand, strength, dose, frequency, timing, duration
  5. Disambiguate — Strength-based DB lookup to resolve similar-named drugs
  6. Calibrate   — Per-medicine confidence from OCR score + DB match quality

The returned dict is 100% compatible with the previous ocr_prescription_v7() shape:
  {medicines: [...], overall_confidence: float, legibility: int}

Gemini (ocr_prescription_v7) remains as a graceful fallback in prescription.py.
"""
import asyncio
import io
import logging
import re
import time
from typing import Any

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None          # type: ignore
    _NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

log = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
_MODEL_ID      = "microsoft/trocr-large-handwritten"
_BEAM_WIDTH    = 4        # Beam search width (accuracy/speed)
_MAX_NEW_TOKENS = 64      # Max tokens to generate per line
_MED_BOOST     = 1.5      # Log-prob boost for medicine-vocab paths (tuned: 3.0 overcorrects)
_MIN_LINE_H    = 18       # Pixels — lines shorter than this are noise
_MIN_LINE_W    = 40       # Pixels — very narrow crops are noise
_IMG_MIN_H     = 500      # Upscale if image height is below this

# ── Model singleton ────────────────────────────────────────────────────────────
_processor   = None   # TrOCRProcessor
_model       = None   # VisionEncoderDecoderModel
_vocab_proc  = None   # MedicineVocabLogitsProcessor (built once per process)
_model_lock  = asyncio.Lock()
_load_error: str | None = None


async def _ensure_model() -> bool:
    """
    Load TrOCR once.  Subsequent calls return immediately.
    Downloads ~1.3 GB to HF cache on first run (one-time cost).
    Returns True if model is ready, False on load failure.
    """
    global _processor, _model, _vocab_proc, _load_error

    if _model is not None:
        return True
    if _load_error is not None:
        return False

    async with _model_lock:
        if _model is not None:
            return True
        try:
            log.info("Loading TrOCR model: %s (this downloads ~1.3GB on first run)", _MODEL_ID)
            t0 = time.monotonic()
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LogitsProcessorList

            _processor = await asyncio.to_thread(
                TrOCRProcessor.from_pretrained, _MODEL_ID
            )
            _model = await asyncio.to_thread(
                VisionEncoderDecoderModel.from_pretrained, _MODEL_ID
            )
            # Move model to best available device (MPS → CUDA → CPU)
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            _model = _model.to(device)
            log.info("Model on device: %s", device)
            _model.eval()

            # Build medicine-aware logits processor from full BD vocab
            try:
                from app.services.semantic_search import get_medicine_names_for_vocab
                from app.services.medicine_vocab import MedicineVocabLogitsProcessor
                names = get_medicine_names_for_vocab()
                if names:
                    _vocab_proc = MedicineVocabLogitsProcessor(
                        names, _processor.tokenizer, boost=_MED_BOOST
                    )
                    log.info("MedicineVocabLogitsProcessor built with %d brand names", len(names))
            except Exception as exc:
                log.warning("Medicine vocab processor skipped: %s", exc)
                _vocab_proc = None

            elapsed = time.monotonic() - t0
            log.info("TrOCR ready in %.1fs", elapsed)
            return True

        except Exception as exc:
            _load_error = str(exc)
            log.error("TrOCR load failed: %s", exc, exc_info=True)
            return False


# ── Stage 1: Image preprocessing ──────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> Image.Image:
    """
    Load and prepare a prescription image for best TrOCR accuracy.

    Steps:
      • Decode bytes → RGB
      • Upscale if too small (TrOCR needs sufficient pixel detail)
      • Deskew: detect dominant text angle, rotate to horizontal
      • CLAHE contrast enhancement
      • Mild sharpening
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Upscale small images
    if img.height < _IMG_MIN_H:
        scale = _IMG_MIN_H / img.height
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    # Deskew
    try:
        img = _deskew(img)
    except Exception as exc:
        log.debug("Deskew skipped: %s", exc)

    # CLAHE contrast on L channel (keeps colour, enhances local contrast)
    try:
        img = _clahe_enhance(img)
    except Exception:
        # Fallback: simple contrast boost
        img = ImageEnhance.Contrast(img).enhance(1.4)

    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    return img


def _deskew(img: Image.Image) -> Image.Image:
    """
    Detect dominant text rotation via Hough lines and correct it.
    Only corrects small tilts (< ±15°) to avoid over-rotation.
    """
    import cv2

    gray = np.array(img.convert("L"))
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return img

    angles = []
    for rho, theta in lines[:50, 0]:  # Use top 50 lines
        angle = (theta - np.pi / 2) * 180 / np.pi
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.4:  # Skip trivial correction
        return img

    return img.rotate(-median_angle, expand=True, fillcolor=(255, 255, 255))


def _clahe_enhance(img: Image.Image) -> Image.Image:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel."""
    import cv2

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    merged = cv2.merge((l_ch, a_ch, b_ch))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


# ── Stage 2: Line segmentation ─────────────────────────────────────────────────

def _segment_lines(img: Image.Image) -> list[Image.Image]:
    """
    Segment a prescription image into individual text-line crops using
    OpenCV horizontal projection profile analysis.

    Algorithm:
      1. Threshold to binary (Otsu's method)
      2. Compute row-wise sum of foreground pixels (projection profile)
      3. Find transitions between "text rows" and "gap rows"
      4. Crop each text band with small vertical padding
    """
    import cv2

    gray = np.array(img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Mild horizontal dilation to close gaps within a line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Row projection
    row_proj = dilated.sum(axis=1)
    threshold = row_proj.max() * 0.04  # rows with < 4% max density = gap

    lines: list[Image.Image] = []
    in_line = False
    start = 0
    pad = 5  # extra pixels above and below each line crop

    for row_idx, val in enumerate(row_proj):
        if val > threshold and not in_line:
            start = max(0, row_idx - pad)
            in_line = True
        elif val <= threshold and in_line:
            end = min(img.height, row_idx + pad)
            h = end - start
            w = img.width
            if h >= _MIN_LINE_H and w >= _MIN_LINE_W:
                crop = img.crop((0, start, w, end))
                lines.append(crop)
            in_line = False

    # Handle text running to the bottom
    if in_line:
        end = img.height
        h = end - start
        if h >= _MIN_LINE_H:
            lines.append(img.crop((0, start, img.width, end)))

    # If segmentation failed (no lines found), return whole image
    if not lines:
        log.warning("Line segmentation found 0 lines — using full image")
        return [img]

    log.debug("Line segmentation: %d lines from %dx%d image", len(lines), img.width, img.height)
    return lines


# ── Stage 3: TrOCR inference ───────────────────────────────────────────────────

def _trocr_infer_line(line_img: Image.Image) -> tuple[str, float]:
    """
    Run TrOCR on a single line image.

    Returns:
        (decoded_text, confidence)
        confidence is exp(mean_log_prob), mapped to [0.0, 1.0].
    """
    import torch
    from transformers import LogitsProcessorList

    pixel_values = _processor(images=line_img, return_tensors="pt").pixel_values.to(_model.device)

    logits_processors = LogitsProcessorList([_vocab_proc]) if _vocab_proc else None

    with torch.no_grad():
        kwargs: dict[str, Any] = dict(
            num_beams=_BEAM_WIDTH,
            max_new_tokens=_MAX_NEW_TOKENS,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if logits_processors:
            kwargs["logits_processor"] = logits_processors

        outputs = _model.generate(pixel_values, **kwargs)

    text = _processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Sequence score → probability
    # outputs.sequences_scores is a tensor of shape (batch_size,) with log-probs
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        log_prob = outputs.sequences_scores[0].item()
        # Map log-prob (typically -10 to 0) to [0, 1] confidence
        # A log-prob of 0 = perfect sequence (prob 1.0); -5 ≈ 0.5; -10 ≈ 0.0
        confidence = float(min(1.0, max(0.0, 1.0 + log_prob / 10.0)))
    else:
        confidence = 0.6  # Default if scores unavailable

    return text, confidence


# ── Stage 4: Rule-based NER ────────────────────────────────────────────────────

# Pre-compiled patterns
_DOSE_PATTERN = re.compile(r'(\d)\s*\+\s*(\d)\s*\+\s*(\d)')
_STRENGTH_RE  = re.compile(r'(\d+\.?\d*)\s*(mg|ml|mcg|µg|g\b|iu|mEq|mmol)', re.IGNORECASE)
_DURATION_RE  = re.compile(r'(?:for\s+)?(\d+)\s*(days?|weeks?|months?)', re.IGNORECASE)
_BRAND_RE     = re.compile(r'^([A-Z][a-zA-Z0-9\-]+)')  # First capitalised token

_FREQ_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bQID\b|\bQDS\b|\b4\s*times?\b',     re.I), "four_times_daily"),
    (re.compile(r'\bTDS\b|\bTID\b|\b3\s*times?\b',     re.I), "three_times_daily"),
    (re.compile(r'\bBD\b|\bBID\b|\b2\s*times?\b|twice', re.I), "twice_daily"),
    (re.compile(r'\bOD\b|\bonce\s*(a\s*)?(day|daily)\b', re.I), "once_daily"),
    (re.compile(r'\bHS\b|\bbed\s*time\b',               re.I), "once_daily"),
    (re.compile(r'\bSOS\b|\bPRN\b|\bas\s+needed\b',     re.I), "as_needed"),
    (re.compile(r'\bweekly\b|\bper\s+week\b',           re.I), "weekly"),
]

_TIMING_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bAC\b|\bbefore\s+meal|\bpre-?\s*meal|\bon\s+empty\s+stomach|\bfasting', re.I), "before_meals"),
    (re.compile(r'\bPC\b|\bafter\s+meal|\bpost-?\s*meal',           re.I), "after_meals"),
    (re.compile(r'\bwith\s+(meal|food)',                            re.I), "with_meals"),
    (re.compile(r'\bHS\b|\bat\s+bed\s*time|\bnightly',             re.I), "at_bedtime"),
    (re.compile(r'\bempty\s+stomach|\bfasting',                    re.I), "on_empty_stomach"),
]

_FORM_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\btab(let)?s?\b',         re.I), "tablet"),
    (re.compile(r'\bcap(sule)?s?\b',        re.I), "capsule"),
    (re.compile(r'\bsyrup\b|\bsusp\b|\bsolution\b', re.I), "syrup"),
    (re.compile(r'\binj(ection)?\b',        re.I), "injection"),
    (re.compile(r'\bcream\b|\boint\b|\bgel\b', re.I), "cream"),
    (re.compile(r'\bdrops?\b',              re.I), "drops"),
    (re.compile(r'\binhaler?\b|\bpuffer\b', re.I), "inhaler"),
]


def _parse_line(raw_text: str) -> dict[str, Any]:
    """
    Extract structured prescription fields from one OCR'd line.

    Returns a dict with keys compatible with ExtractedMedicine.
    Only includes keys where a value was found.
    """
    text = raw_text.strip()
    result: dict[str, Any] = {"raw_text": text}

    if not text:
        return result

    # Brand name: first capitalised word
    bm = _BRAND_RE.match(text)
    result["brand_name"] = bm.group(1) if bm else text.split()[0] or "Unknown"

    # Strength — primary: number + explicit unit (e.g. "500mg", "20 ml")
    sm = _STRENGTH_RE.search(text)
    if sm:
        result["strength"] = f"{sm.group(1)}{sm.group(2).lower()}"
    else:
        # Fallback: bare number after brand name that looks like a dosage
        # e.g. "Metformin 500 BD" → "500mg" (unit inferred as mg)
        bare = re.search(r'\b(\d{2,4})\b(?!\s*[+%])', text)
        if bare:
            result["strength"] = f"{bare.group(1)}mg"

    # Dosage form
    for pat, form in _FORM_MAP:
        if pat.search(text):
            result["dosage_form"] = form
            break

    # Dose instruction (e.g. 1+0+1)
    dm = _DOSE_PATTERN.search(text)
    if dm:
        result["dose_instruction"] = f"{dm.group(1)}+{dm.group(2)}+{dm.group(3)}"

    # Frequency
    for pat, freq in _FREQ_MAP:
        if pat.search(text):
            result["frequency"] = freq
            break
    # Infer from dose instruction if no explicit frequency found
    if "frequency" not in result and "dose_instruction" in result:
        active = sum(1 for p in result["dose_instruction"].split("+") if p != "0")
        result["frequency"] = {1: "once_daily", 2: "twice_daily",
                                3: "three_times_daily", 4: "four_times_daily"}.get(active, "other")

    # Timing
    for pat, timing in _TIMING_MAP:
        if pat.search(text):
            result["timing"] = timing
            break

    # Duration
    durm = _DURATION_RE.search(text)
    if durm:
        unit = durm.group(2).lower().rstrip("s")  # normalise to singular: "days" → "day"
        n    = int(durm.group(1))
        result["duration"] = f"{n} {unit}{'s' if n != 1 else ''}"  # "5 days", "1 day"

    return result


# ── Stage 5 + 6: Disambiguation + confidence calibration ──────────────────────

def _enrich_and_calibrate(
    parsed: dict[str, Any],
    ocr_score: float,
) -> dict[str, Any]:
    """
    For a single parsed medicine line:
      1. Run strength-based DB disambiguation to resolve ambiguous brand names
      2. Compute calibrated confidence from OCR score + DB match quality

    Returns a dict compatible with ExtractedMedicine (plus confidence).
    """
    from app.database import get_supabase
    from app.services.medicine_vocab import disambiguate_by_strength, extract_strength

    brand  = parsed.get("brand_name", "Unknown")
    strength = parsed.get("strength") or extract_strength(parsed.get("raw_text", ""))

    # Strength-based disambiguation
    try:
        dis = disambiguate_by_strength(brand, strength, get_supabase())
    except Exception as exc:
        log.debug("Disambiguation error for '%s': %s", brand, exc)
        dis = {"resolved_brand": brand, "generic_name": None, "strength": strength,
               "confidence_bonus": 0.0, "disambiguation": "error",
               "medicine_id": None, "unit_price": None}

    # Confidence calibration
    # ocr_score: 0.0–1.0 from TrOCR sequence score
    # confidence_bonus: 0.0–0.20 from DB match quality
    base_confidence = 0.5 * ocr_score + 0.5 * min(1.0, ocr_score + dis["confidence_bonus"])
    confidence = round(min(1.0, max(0.0, base_confidence)), 3)

    return {
        "raw_text":        parsed.get("raw_text"),
        "brand_name":      dis["resolved_brand"] or brand,
        "generic_name":    dis["generic_name"] or parsed.get("generic_name"),
        "strength":        dis["strength"] or parsed.get("strength"),
        "dosage_form":     parsed.get("dosage_form"),
        "dose_instruction": parsed.get("dose_instruction"),
        "frequency":       parsed.get("frequency"),
        "timing":          parsed.get("timing"),
        "duration":        parsed.get("duration"),
        "confidence":      confidence,
        "medicine_id":     dis.get("medicine_id"),
        "unit_price":      dis.get("unit_price"),
        "_disambiguation": dis["disambiguation"],  # Internal — stripped before returning
    }


def _estimate_legibility(ocr_scores: list[float]) -> int:
    """Map average OCR confidence to a 1–10 legibility score."""
    if not ocr_scores:
        return 5
    avg = sum(ocr_scores) / len(ocr_scores)
    return max(1, min(10, round(avg * 10)))


# ── Sync pipeline (runs in thread pool) ───────────────────────────────────────

def _run_sync(
    image_bytes: bytes,
    mime_type: str,
    rag_medicines: list[dict[str, Any]] | None,
    patient_conditions: list[str] | None,
    current_medicines: list[str] | None,
) -> dict[str, Any]:
    """
    Full synchronous OCR pipeline.  Called via asyncio.to_thread() to avoid
    blocking the FastAPI event loop during CPU-bound inference.
    """
    t_start = time.monotonic()

    # Stage 1: preprocess
    img = _preprocess(image_bytes)

    # Stage 2: segment into lines
    lines = _segment_lines(img)

    # Stage 3: TrOCR inference per line
    ocr_results: list[tuple[str, float]] = []
    for i, line_img in enumerate(lines):
        try:
            text, score = _trocr_infer_line(line_img)
            if text.strip():
                ocr_results.append((text, score))
                log.debug("Line %d: %r (conf=%.2f)", i, text[:60], score)
        except Exception as exc:
            log.warning("TrOCR error on line %d: %s", i, exc)

    if not ocr_results:
        return {
            "legibility": 2,
            "overall_confidence": 0.0,
            "medicines": [],
            "_engine": "trocr",
            "_error": "no_text_detected",
        }

    # Stage 4: NER per line
    parsed_lines = [_parse_line(text) for text, _ in ocr_results]
    ocr_scores   = [score for _, score in ocr_results]

    # Filter lines that don't look like medicine lines (no capitalized brand-like token)
    med_lines = [
        (p, s) for p, s in zip(parsed_lines, ocr_scores)
        if p.get("brand_name") and p["brand_name"] != "Unknown"
        and p.get("brand_name", "")[0].isupper()
    ]

    if not med_lines:
        # Fall back to all lines if filter was too aggressive
        med_lines = list(zip(parsed_lines, ocr_scores))

    # Stage 5 + 6: Disambiguation + confidence calibration
    medicines: list[dict[str, Any]] = []
    for parsed, score in med_lines:
        enriched = _enrich_and_calibrate(parsed, score)
        # Remove internal-only key before returning
        enriched.pop("_disambiguation", None)
        medicines.append(enriched)

    overall_conf = round(
        sum(m["confidence"] for m in medicines) / len(medicines), 3
    ) if medicines else 0.0

    elapsed = time.monotonic() - t_start
    log.info(
        "TrOCR pipeline: %d lines → %d medicines, conf=%.2f, leg=%d, %.1fs",
        len(lines), len(medicines), overall_conf,
        _estimate_legibility(ocr_scores), elapsed,
    )

    return {
        "legibility":         _estimate_legibility(ocr_scores),
        "overall_confidence": overall_conf,
        "medicines":          medicines,
        "_engine":            "trocr",
        "_elapsed_s":         round(elapsed, 2),
    }


# ── Public async entry point ──────────────────────────────────────────────────

async def run_ocr_pipeline(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    rag_medicines: list[dict[str, Any]] | None = None,
    patient_conditions: list[str] | None = None,
    current_medicines: list[str] | None = None,
) -> dict[str, Any]:
    """
    Main entry point for the TrOCR OCR pipeline.

    Signature is compatible with ocr_prescription_v7() so prescription.py
    can swap between engines with a single try/except.

    Raises RuntimeError if the model failed to load.
    Returns same dict shape as ocr_prescription_v7:
      {medicines: [...], overall_confidence: float, legibility: int}
    """
    if not _NUMPY_AVAILABLE or not _PIL_AVAILABLE:
        raise RuntimeError("TrOCR unavailable: numpy/pillow not installed (cloud deploy uses Gemini)")
    ready = await _ensure_model()
    if not ready:
        raise RuntimeError(f"TrOCR engine unavailable: {_load_error}")

    return await asyncio.to_thread(
        _run_sync,
        image_bytes, mime_type,
        rag_medicines, patient_conditions, current_medicines,
    )
