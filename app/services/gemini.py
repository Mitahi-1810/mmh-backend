"""
Gemini Vision — prescription OCR pipeline (V8).

Model strategy (confidence-gated dual-pass):
  Pass 1:  gemini-2.5-flash  — fast, highly capable; handles ~85% of cases
  Pass 2:  gemini-2.5-pro    — maximum accuracy; triggered automatically when
             overall_confidence < OCR_GATE  OR  any medicine confidence < MED_GATE
  Merge:   per-medicine, whichever pass read the medicine with higher confidence wins.
           If Pass 2 finds new medicines that Pass 1 missed, they are appended.

Why two passes?
  Handwritten Bangladeshi prescriptions mix Bengali numerals, English drug names,
  informal abbreviations, and doctor shorthand. No single model pass is reliable
  for complex handwriting — a second pass from a fresh context catches misreads.

Prompt engineering highlights:
  • Chain-of-thought: OBSERVE → ENHANCE → EXTRACT → VERIFY → SCORE
  • RAG injection of the 120 top BD brands for disambiguation
  • Patient context (conditions + current meds) to resolve ambiguous names
  • Explicit OCR error guidance (common character confusions, BD numeral patterns)
  • Pass-2 includes Pass-1 preliminary readings so the model can cross-verify
"""
import base64
import json
import re
import logging
from typing import Any

import google.generativeai as genai
from app.config import get_settings
from app.models import ChatResponseEnvelope

log = logging.getLogger(__name__)

_configured = False

# ── Model constants ────────────────────────────────────────────────────────────
#
# Both passes use gemini-2.5-flash (the most capable model available on the
# current quota).  The value of the dual-pass approach lies in the prompt, not
# a different model: Pass 2 receives Pass-1's verbatim readings and is
# instructed to cross-verify each one — a "self-reflection" technique that
# catches systematic OCR misreads even when the same model is used.
#
# When a paid Google AI account is available, swap _OCR_MODEL_ACCURATE to
# "gemini-2.5-pro" for maximum accuracy on the second pass.
_OCR_MODEL_FAST     = "gemini-2.5-flash"   # Primary: fast + very strong vision
_OCR_MODEL_ACCURATE = "gemini-2.5-flash"   # Fallback: same model, guided prompt
_EXPIRY_MODEL       = "gemini-2.5-flash"   # Expiry date extraction
_LEGACY_MODEL       = "gemini-2.5-flash"   # Legacy OCR endpoint

# Confidence gates — below these thresholds a second pass is triggered.
# Keep these low: Pass 1 (gemini-2.5-flash) is already very strong.
# Pass 2 is reserved for genuinely difficult prescriptions, not every scan.
_OCR_GATE  = 0.55   # overall_confidence below this → run second pass
_MED_GATE  = 0.45   # any single medicine below this → run second pass


def _ensure_configured():
    global _configured
    if not _configured:
        genai.configure(api_key=get_settings().gemini_api_key)
        _configured = True


def _parse_json(raw: str) -> dict:
    """
    Extract a JSON object from raw Gemini output.
    Handles markdown fences, leading prose, and trailing garbage.
    """
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fallback: extract the outermost {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in Gemini response: {raw[:200]}")


# ── Legacy endpoint (kept for backwards compatibility) ─────────────────────────

async def ocr_prescription(image_bytes: bytes, mime_type: str = "image/jpeg") -> ChatResponseEnvelope:
    """
    Legacy prescription OCR — kept for backwards compatibility.
    New code should call ocr_prescription_v7().
    """
    _ensure_configured()
    model = genai.GenerativeModel(_LEGACY_MODEL)

    prompt = """You are a medical OCR assistant for Bangladesh.
Extract all medications from this prescription image.
Return ONLY valid JSON in this format:
{
  "message": "I found X medications in the prescription.",
  "format": "prescription_list",
  "data": [
    {"brand": "Napa", "generic": "Paracetamol", "dose": "500mg", "frequency": "3 times daily"},
    ...
  ]
}
If you cannot read the prescription clearly, set data to [] and explain in message.
"""

    image_part = {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
    response = model.generate_content([prompt, image_part])
    raw = response.text or ""

    try:
        obj = _parse_json(raw)
        return ChatResponseEnvelope(**obj)
    except Exception:
        return ChatResponseEnvelope(
            message="Could not parse the prescription clearly. Please try a clearer photo.",
            format="text",
            data=None,
        )


# ── V8 Prescription OCR ────────────────────────────────────────────────────────

_V8_SYSTEM_INSTRUCTION = """\
You are an expert medical document AI specialising in Bangladeshi prescriptions.

EXPERTISE:
• Bangladeshi doctors write in English, Bengali, or mixed script
• Common abbreviations: BD (twice daily), TDS (three times daily), OD (once daily),
  HS (at bedtime), AC (before meals), PC (after meals), SOS (as needed)
• Drug names are often shortened: "Napa" = Napa 500mg, "Seclo" = Seclo 20mg,
  "Losec" = Losec 20mg, "Omi" = Omidon/Omeprazole, "Met" = Metformin
• Strengths: "5" often means "5mg", "500" means "500mg", "20" means "20mg"
• Bengali numerals: ০১২৩৪৫৬৭৮৯ map to 0123456789

OCR ERROR AWARENESS — common confusions to watch for:
  • 0 ↔ O ↔ D  (zero, letter O, letter D)
  • 1 ↔ l ↔ I  (one, lowercase L, capital I)
  • 5 ↔ S      • 6 ↔ b      • 8 ↔ B
  • rn ↔ m     • cl ↔ d     • vv ↔ w
  • "Stiba" and "Stira" could both be "Stiba" — trust DB brand lists
  • "Losec" and "Lasix" are DIFFERENT drugs; do not confuse them

ACCURACY MANDATE: Never fabricate a medicine. Only extract what is physically present.
When uncertain between two brand names, pick the most common one in Bangladesh and
set confidence accordingly. Never skip a partially-legible line — extract it with low
confidence rather than omitting it.\
"""


def _build_v8_prompt(
    rag_medicines: list[dict[str, Any]],
    patient_conditions: list[str],
    current_medicines: list[str],
    pass1_readings: list[dict[str, Any]] | None = None,
    all_brand_names: str | None = None,
) -> str:
    """
    Build the full CoT extraction prompt.

    pass1_readings: if provided, this is Pass 2 — include Pass-1 findings so
    the model can cross-verify each reading rather than starting cold.
    """

    # ── Full brand vocabulary (all ~7,600+ known BD brands) ──────────────────
    # Gemini can see every brand name so it can recognise ANY medicine it reads.
    if all_brand_names:
        vocab_block = (
            "COMPLETE BANGLADESH MEDICINE VOCABULARY — every known BD brand name.\n"
            "When you read a word that looks like a medicine name, check it against this list.\n"
            "If you read something close to a name here (1-2 chars different), use the DB name.\n"
            f"{all_brand_names}\n\n"
        )
    else:
        vocab_block = ""

    # ── Detailed RAG block (top 300 with generic + strength for disambiguation) ─
    if rag_medicines:
        rag_lines = "\n".join(
            f"  • {r['brand_name']} → {r.get('generic_name', '')} {r.get('strength', '')}".rstrip()
            for r in rag_medicines[:300]
        )
        rag_block = f"TOP BRANDS WITH DETAILS (generic + strength for disambiguation):\n{rag_lines}\n\n"
    else:
        rag_block = ""

    # ── Patient context block ─────────────────────────────────────────────────
    ctx_parts = []
    if patient_conditions:
        ctx_parts.append(f"Patient conditions: {', '.join(patient_conditions)}")
    if current_medicines:
        ctx_parts.append(f"Patient's recent medicines: {', '.join(current_medicines)}")
    patient_block = ("PATIENT CONTEXT (use to resolve ambiguous drug names):\n" + "\n".join(ctx_parts) + "\n\n") if ctx_parts else ""

    # ── Pass-2 cross-verification block ───────────────────────────────────────
    if pass1_readings:
        p1_lines = "\n".join(
            f"  {i+1}. \"{m.get('raw_text','?')}\" → {m.get('brand_name','?')} "
            f"{m.get('strength','')} | conf={m.get('confidence',0):.2f}"
            for i, m in enumerate(pass1_readings)
        )
        pass1_block = (
            f"PRELIMINARY PASS-1 READINGS (another model's output — cross-verify each one):\n"
            f"{p1_lines}\n\n"
            "For each Pass-1 reading, confirm it is correct or provide a better interpretation.\n"
            "Also check whether Pass 1 missed any medicine lines.\n\n"
        )
    else:
        pass1_block = ""

    return f"""{vocab_block}{rag_block}{patient_block}{pass1_block}\
Analyse the prescription image in exactly 5 steps:

STEP 1 — OBSERVE
  Describe the physical layout (handwritten/printed/mixed, language, ink quality,
  overall legibility 1-10, any smudges or torn areas).

STEP 2 — ENHANCE DIFFICULT SECTIONS
  Identify any hard-to-read word or number. For each, list 2-3 plausible
  interpretations using the OCR error rules above.

STEP 3 — EXTRACT ALL MEDICINES
  Read every medicine line carefully. For each medicine extract:
    • Exact raw text as written on paper
    • Brand name (proper capitalisation, e.g. "Napa", "Seclo")
    • Strength / dosage form
    • Dose instructions (e.g. "1+0+1", "BD", "OD")

STEP 4 — VERIFY AGAINST BD BRAND LIST
  For each extracted brand name, confirm it appears in the BD brands list above.
  If a name is close but slightly different (OCR error), correct it.
  If a name is truly unknown, keep it as-is and note it.

STEP 5 — SCORE
  Assign each medicine a confidence 0.0–1.0 based on legibility.
  • 0.90–1.00: clearly readable, confirmed in DB
  • 0.75–0.89: readable with minor uncertainty
  • 0.60–0.74: plausible interpretation, some ambiguity
  • Below 0.60: poorly legible, may be wrong

Return ONLY valid JSON (no markdown fences, no prose before/after):
{{
  "legibility": <integer 1-10>,
  "overall_confidence": <float 0.0-1.0, weighted average of medicine confidences>,
  "medicines": [
    {{
      "raw_text": "<verbatim text exactly as written>",
      "brand_name": "<best-match brand name, capitalised>",
      "generic_name": "<INN generic name or null>",
      "strength": "<e.g. 500mg or null>",
      "dosage_form": "<tablet|capsule|syrup|injection|cream|drops|inhaler|null>",
      "dose_instruction": "<e.g. 1+0+1 or null>",
      "frequency": "<once_daily|twice_daily|three_times_daily|four_times_daily|as_needed|weekly|other|null>",
      "timing": "<before_meals|after_meals|with_meals|at_bedtime|on_empty_stomach|other|null>",
      "duration": "<e.g. 5 days|1 month|null>",
      "confidence": <float 0.0-1.0>
    }}
  ]
}}
If no medicines are visible, return "medicines": [].
Do NOT include any text outside the JSON object."""


def _normalise_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure required keys exist and values are clamped to valid ranges."""
    result.setdefault("legibility", 5)
    result.setdefault("overall_confidence", 0.5)
    result.setdefault("medicines", [])
    result["legibility"] = max(1, min(10, int(result["legibility"])))
    result["overall_confidence"] = max(0.0, min(1.0, float(result["overall_confidence"])))
    for med in result["medicines"]:
        med["confidence"] = max(0.0, min(1.0, float(med.get("confidence", 0.5))))
        if not med.get("brand_name"):
            med["brand_name"] = med.get("raw_text", "Unknown") or "Unknown"
    return result


def _merge_passes(
    pass1: dict[str, Any],
    pass2: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge two OCR passes into one result.

    Strategy:
      • Build a set of medicines from Pass 2 (higher-accuracy model).
      • For each Pass-1 medicine, if Pass 2 has a corresponding reading
        (matched by raw_text similarity or brand name), keep whichever has
        higher confidence.
      • Medicines in Pass 2 that Pass 1 missed are appended.
      • overall_confidence is the weighted average of the final medicine list.
      • legibility is the max of both passes (better view wins).
    """
    def _key(m: dict) -> str:
        """Normalised key for medicine dedup."""
        return (m.get("brand_name") or m.get("raw_text") or "").lower().strip()

    p2_by_key = {_key(m): m for m in pass2.get("medicines", [])}
    merged: list[dict[str, Any]] = []
    used_p2_keys: set[str] = set()

    for m1 in pass1.get("medicines", []):
        k = _key(m1)
        m2 = p2_by_key.get(k)
        if m2 is not None:
            # Both found the same medicine — take the higher-confidence reading
            winner = m2 if m2["confidence"] >= m1["confidence"] else m1
            merged.append(winner)
            used_p2_keys.add(k)
        else:
            # Pass 2 didn't find this one; keep Pass-1 reading
            merged.append(m1)

    # Append medicines that Pass 2 found but Pass 1 missed
    for k, m2 in p2_by_key.items():
        if k not in used_p2_keys:
            merged.append(m2)
            log.info("Pass-2 found extra medicine not in Pass-1: %s", m2.get("brand_name"))

    if not merged:
        return pass2  # If pass1 was empty, pass2 wins outright

    total_conf = sum(m["confidence"] for m in merged) / len(merged)
    return {
        "legibility": max(pass1.get("legibility", 1), pass2.get("legibility", 1)),
        "overall_confidence": round(total_conf, 3),
        "medicines": merged,
        "_passes": 2,
    }


async def _run_single_ocr_pass(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Execute one OCR pass with the given model.
    Returns normalised result dict, or error-fallback on failure.
    """
    _ensure_configured()
    model = genai.GenerativeModel(
        model_name,
        system_instruction=_V8_SYSTEM_INSTRUCTION,
    )
    image_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode(),
    }

    response = None
    try:
        response = model.generate_content(
            [prompt, image_part],
            generation_config=genai.types.GenerationConfig(temperature=0.05),
        )
        raw = (response.text or "").strip()
        result = _parse_json(raw)
        return _normalise_result(result)

    except (json.JSONDecodeError, ValueError) as exc:
        preview = (getattr(response, "text", "") or "")[:300]
        log.warning("OCR parse error (%s): %s | raw=%s", model_name, exc, preview)
        return {"legibility": 5, "overall_confidence": 0.0, "medicines": [], "_error": "parse_error"}
    except Exception as exc:
        log.error("OCR unexpected error (%s): %s", model_name, exc, exc_info=True)
        return {"legibility": 1, "overall_confidence": 0.0, "medicines": [], "_error": str(exc)}


async def ocr_prescription_v7(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    rag_medicines: list[dict[str, Any]] | None = None,
    patient_conditions: list[str] | None = None,
    current_medicines: list[str] | None = None,
    all_brand_names: str | None = None,
) -> dict[str, Any]:
    """
    V8 dual-pass prescription OCR with RAG + patient context.

    Pass 1 (always):
      gemini-2.5-flash — strong vision, fast.

    Pass 2 (triggered when confidence is too low):
      gemini-2.5-pro — maximum accuracy, given Pass-1 readings as cross-check context.

    Returns:
        legibility (int), overall_confidence (float),
        medicines (list[ExtractedMedicine-compatible dicts]),
        _passes (int, 1 or 2)
    """
    rag    = rag_medicines or []
    conds  = patient_conditions or []
    meds   = current_medicines or []
    brands = all_brand_names or ""

    # ── Pass 1: fast model ────────────────────────────────────────────────────
    prompt_p1 = _build_v8_prompt(rag, conds, meds, pass1_readings=None, all_brand_names=brands)
    result_p1 = await _run_single_ocr_pass(image_bytes, mime_type, prompt_p1, _OCR_MODEL_FAST)

    log.info(
        "OCR Pass-1 (%s): %d medicines, conf=%.2f, leg=%d",
        _OCR_MODEL_FAST,
        len(result_p1.get("medicines", [])),
        result_p1.get("overall_confidence", 0),
        result_p1.get("legibility", 0),
    )

    # ── Gate: decide whether Pass 2 is needed ────────────────────────────────
    p1_conf     = result_p1.get("overall_confidence", 0.0)
    p1_meds     = result_p1.get("medicines", [])
    low_med_cnt = sum(1 for m in p1_meds if m.get("confidence", 0) < _MED_GATE)
    needs_p2    = (p1_conf < _OCR_GATE) or (low_med_cnt > 0)

    if not needs_p2:
        result_p1["_passes"] = 1
        return result_p1

    # ── Pass 2: accurate model with Pass-1 context ────────────────────────────
    log.info(
        "OCR Pass-2 triggered: overall_conf=%.2f < %.2f OR %d low-conf medicines",
        p1_conf, _OCR_GATE, low_med_cnt,
    )
    prompt_p2 = _build_v8_prompt(rag, conds, meds, pass1_readings=p1_meds, all_brand_names=brands)
    result_p2 = await _run_single_ocr_pass(image_bytes, mime_type, prompt_p2, _OCR_MODEL_ACCURATE)

    log.info(
        "OCR Pass-2 (%s): %d medicines, conf=%.2f, leg=%d",
        _OCR_MODEL_ACCURATE,
        len(result_p2.get("medicines", [])),
        result_p2.get("overall_confidence", 0),
        result_p2.get("legibility", 0),
    )

    # ── Merge both passes ─────────────────────────────────────────────────────
    final = _merge_passes(result_p1, result_p2)
    log.info(
        "OCR merged: %d medicines, conf=%.2f",
        len(final.get("medicines", [])),
        final.get("overall_confidence", 0),
    )
    return final


# ── Expiry date extraction ─────────────────────────────────────────────────────

async def ocr_expiry(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Extract expiry date from a medicine strip/blister pack photo.
    Returns {"expiry_date": "YYYY-MM-DD", "raw_text": "10/26"} or {"expiry_date": None}.
    """
    _ensure_configured()
    model = genai.GenerativeModel(_EXPIRY_MODEL)

    prompt = """You are a pharmaceutical packaging OCR assistant.
Extract the expiry / expiration date from this medicine packaging image.
The date may be printed as: EXP, Exp., Expiry, Use before, Best before.
Common formats: MM/YY, MM/YYYY, DD/MM/YYYY, MON YYYY (e.g. JAN 2026).

Return ONLY valid JSON: {"expiry_date": "YYYY-MM-DD", "raw_text": "<exactly what you read>"}
If no expiry date is visible, return: {"expiry_date": null, "raw_text": null}
"""

    image_part = {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
    response = model.generate_content(
        [prompt, image_part],
        generation_config=genai.types.GenerationConfig(temperature=0.0),
    )
    raw = response.text or ""

    try:
        return _parse_json(raw)
    except Exception:
        return {"expiry_date": None, "raw_text": None}


# ── Audio transcription (Banglish STT) ────────────────────────────────────────

async def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """
    Transcribe and translate Banglish voice note to English.
    Returns the transcribed text string.
    """
    _ensure_configured()
    model = genai.GenerativeModel(_OCR_MODEL_FAST)

    prompt = """Transcribe this audio. The speaker may mix Bengali and English (Banglish).
Transcribe what is said, then translate to English.
Return ONLY valid JSON: {"transcript": "<original>", "english": "<english translation>"}
"""

    audio_part = {"mime_type": mime_type, "data": base64.b64encode(audio_bytes).decode()}
    response = model.generate_content([prompt, audio_part])
    raw = response.text or ""

    try:
        obj = _parse_json(raw)
        return obj.get("english") or obj.get("transcript") or ""
    except Exception:
        return raw
