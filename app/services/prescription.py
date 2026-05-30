"""
Prescription extraction pipeline — V8 architecture.

Flow:
  1. Upload image → Supabase Storage
  2. Fetch RAG context (top BD medicines) + patient context
  3. OCR — tries TrOCR engine first (specialist HTR, medicine-aware)
           falls back to Gemini if TrOCR unavailable or fails
  4. Semantic match each medicine to DB (cascade: exact → prefix → substr → generic)
  5. Fuzzy post-OCR correction (rapidfuzz WRatio) + DB status tagging
  6. Persist to prescriptions + prescription_medicines tables
  7. Determine review tier based on confidence
  8. Return PrescriptionScanResponse

Engine selection:
  Primary  — TrOCR-large-handwritten (microsoft/trocr-large-handwritten)
             97%+ CER on general handwriting; medicine-aware constrained beam search
  Fallback — Gemini 2.5-flash (dual-pass, used if TrOCR model unavailable)

Confidence tiers:
  ≥ 0.85  → auto-confirm (no review widget shown)
  0.60-0.84 → soft review (highlight low-confidence lines)
  < 0.60  → full review (all lines editable, warning banner)
"""
import logging
import uuid
from typing import Any

from app.database import get_supabase
from app.models import ExtractedMedicine, PrescriptionScanResponse
from app.services.gemini import ocr_prescription_v7
from app.services.ocr_engine import run_ocr_pipeline
from app.services.semantic_search import (
    fuzzy_correct_brand,
    get_all_brand_names_compact,
    get_patient_context,
    get_rag_medicines,
    match_medicine_by_name,
)

log = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.60   # below this → full review required
SOFT_THRESHOLD   = 0.85   # below this → soft review (highlight uncertain lines)


# ── Storage ───────────────────────────────────────────────────────────────────

def _upload_image(image_bytes: bytes, content_type: str) -> str:
    """Upload prescription image to Supabase Storage, return public URL."""
    sb = get_supabase()
    path = str(uuid.uuid4())
    sb.storage.from_("prescriptions").upload(
        path=path,
        file=image_bytes,
        file_options={"content-type": content_type},
    )
    return sb.storage.from_("prescriptions").get_public_url(path)


# ── DB persistence ────────────────────────────────────────────────────────────

def _insert_prescription(
    user_id: str,
    image_url: str,
    raw_gemini_output: dict[str, Any],
    overall_confidence: float,
) -> int:
    """Insert a prescriptions row and return its id."""
    sb = get_supabase()
    res = (
        sb.table("prescriptions")
        .insert({
            "user_id": user_id,
            "image_url": image_url,
            "raw_gemini_output": raw_gemini_output,
            "overall_confidence": overall_confidence,
            "status": "pending",
        })
        .execute()
    )
    return res.data[0]["id"]


def _insert_medicine_lines(
    prescription_id: int,
    user_id: str,
    medicines: list[ExtractedMedicine],
) -> list[ExtractedMedicine]:
    """
    Bulk-insert prescription_medicines rows.
    Stamps each ExtractedMedicine with its new pm_id (for frontend corrections).
    Returns the same list with pm_id populated.
    """
    if not medicines:
        return medicines
    sb = get_supabase()
    rows = [
        {
            "prescription_id": prescription_id,
            "user_id": user_id,
            "raw_text": med.raw_text,
            "medicine_id": med.medicine_id,
            "brand_name": med.brand_name,
            "generic_name": med.generic_name,
            "strength": med.strength,
            "dosage_form": med.dosage_form,
            "dose_instruction": med.dose_instruction,
            "frequency": med.frequency,
            "timing": med.timing,
            "duration": med.duration,
            "confidence": med.confidence,
            "was_corrected": False,
            "correction_source": "user",
        }
        for med in medicines
    ]
    res = sb.table("prescription_medicines").insert(rows).execute()
    # Stamp pm_id back onto each ExtractedMedicine object
    if res.data:
        for med, row in zip(medicines, res.data):
            med.pm_id = row.get("id")
    return medicines


# ── Semantic enrichment ───────────────────────────────────────────────────────

def _enrich_with_db_match(medicines: list[dict[str, Any]]) -> list[ExtractedMedicine]:
    """
    For each extracted medicine dict:
      1. Exact DB match  → populate price / id / generic
      2. Fuzzy correction → add db_status / db_suggestion / db_suggestion_generic
         so the frontend can show "Did you mean X?" or "⚠ Similar to dangerous drug"
    """
    result: list[ExtractedMedicine] = []
    for med_raw in medicines:
        brand   = med_raw.get("brand_name") or ""
        generic = med_raw.get("generic_name")

        # ── Exact / tier-cascade match ────────────────────────────────────────
        match = match_medicine_by_name(brand)

        med = ExtractedMedicine(
            raw_text=med_raw.get("raw_text"),
            brand_name=brand or "Unknown",
            generic_name=generic,
            strength=med_raw.get("strength"),
            dosage_form=med_raw.get("dosage_form"),
            dose_instruction=med_raw.get("dose_instruction"),
            frequency=med_raw.get("frequency"),
            timing=med_raw.get("timing"),
            duration=med_raw.get("duration"),
            confidence=float(med_raw.get("confidence", 0.5)),
        )

        if match:
            med.medicine_id    = match.get("id")
            med.matched_generic = match.get("generic_name") or med.generic_name
            med.unit_price     = match.get("price_per_unit")
            if not med.generic_name:
                med.generic_name = match.get("generic_name")
            if not med.strength:
                med.strength = match.get("strength")

        # ── Fuzzy correction pass ─────────────────────────────────────────────
        # Run against full 9,815-brand corpus to catch OCR misreads.
        fuzzy = fuzzy_correct_brand(brand, gemini_generic=generic)
        status = fuzzy["status"]
        db_brand   = fuzzy["brand"]
        db_generic = fuzzy["generic"]

        if status == "confirmed":
            # Exact match — brand name is already correct, nothing to flag
            med.db_status = "confirmed"

        elif status == "suggested" and db_brand and db_brand.lower() != brand.lower():
            # Close match with same generic — safe suggestion
            med.db_status            = "suggested"
            med.db_suggestion        = db_brand
            med.db_suggestion_generic = db_generic

        elif status == "ambiguous" and db_brand:
            # Close match but DIFFERENT generic — dangerous, must warn
            med.db_status            = "ambiguous"
            med.db_suggestion        = db_brand
            med.db_suggestion_generic = db_generic

        else:
            med.db_status = "unknown"

        log.debug(
            "Enrich '%s': exact=%s, fuzzy_status=%s, suggestion=%s",
            brand,
            "yes" if match else "no",
            status,
            db_brand,
        )
        result.append(med)
    return result


# ── Main pipeline entry point ─────────────────────────────────────────────────

async def run_prescription_pipeline(
    user_id: str | None,
    image_bytes: bytes,
    content_type: str,
) -> PrescriptionScanResponse:
    """
    Full V7 prescription pipeline.

    Logged-in (user_id is set):
      1. Upload image to Storage
      2. Load RAG context + patient context
      3. Gemini V7 OCR
      4. DB semantic match per medicine
      5. Persist to DB (prescriptions + prescription_medicines)
      6. Return structured response with prescription_id

    Guest (user_id is None):
      Steps 1 & 5 are skipped — image is never stored, results are never persisted.
      The user still gets full OCR + enrichment output.
    """
    # Step 1 — upload image (logged-in only)
    image_url = ""
    if user_id:
        try:
            image_url = _upload_image(image_bytes, content_type)
        except Exception as exc:
            log.error("Image upload failed: %s", exc)

    # Step 2 — context
    rag_medicines   = get_rag_medicines()
    all_brand_names = get_all_brand_names_compact()   # full ~7,600+ brand vocabulary
    conditions, current_meds = (
        get_patient_context(user_id) if user_id else ([], [])
    )

    log.info(
        "Pipeline start: user=%s, rag=%d medicines, vocab=%d brands, conditions=%s",
        user_id or "guest", len(rag_medicines), len(all_brand_names.split(",")), conditions,
    )

    # Step 3 — OCR extraction
    # Primary engine: Gemini 2.5-flash — multimodal VLM that understands the full
    #   prescription as a document: printed forms, messy handwriting, stamps, variable
    #   lighting, mixed layouts. This is what actually works on real phone photos.
    # Fallback: TrOCR — kept as backup; shines once fine-tuned on pre-segmented lines,
    #   but unreliable on raw prescription images without that fine-tuning.
    ocr_engine_used = "gemini"
    try:
        gemini_result = await ocr_prescription_v7(
            image_bytes=image_bytes,
            mime_type=content_type,
            rag_medicines=rag_medicines,
            patient_conditions=conditions,
            current_medicines=current_meds,
            all_brand_names=all_brand_names,
        )
        if not gemini_result.get("medicines") and gemini_result.get("overall_confidence", 0) < 0.1:
            raise ValueError("Gemini returned empty result")
    except Exception as gemini_exc:
        log.warning("Gemini OCR failed (%s) — falling back to TrOCR", gemini_exc)
        ocr_engine_used = "trocr"
        gemini_result = await run_ocr_pipeline(
            image_bytes=image_bytes,
            mime_type=content_type,
            rag_medicines=rag_medicines,
            patient_conditions=conditions,
            current_medicines=current_meds,
        )
    log.info("OCR complete: engine=%s, medicines=%d", ocr_engine_used, len(gemini_result.get("medicines", [])))

    raw_medicines: list[dict] = gemini_result.get("medicines", [])
    overall_confidence: float = gemini_result.get("overall_confidence", 0.0)
    legibility: int = gemini_result.get("legibility", 5)

    # Step 4 — DB semantic enrichment (works for everyone)
    enriched_medicines = _enrich_with_db_match(raw_medicines)

    # Step 5 — persist (logged-in only)
    prescription_id: int = -1
    if user_id:
        try:
            prescription_id = _insert_prescription(
                user_id=user_id,
                image_url=image_url,
                raw_gemini_output=gemini_result,
                overall_confidence=overall_confidence,
            )
            enriched_medicines = _insert_medicine_lines(
                prescription_id=prescription_id,
                user_id=user_id,
                medicines=enriched_medicines,
            )
        except Exception as exc:
            log.error("DB persist failed: %s", exc, exc_info=True)

    # Step 6 — build response
    review_required = overall_confidence < REVIEW_THRESHOLD

    guest_suffix = "" if user_id else " Sign in to save & correct prescriptions."

    n = len(enriched_medicines)
    if n == 0:
        message = (
            "No medicines could be extracted. Please try a clearer, well-lit photo."
        )
    elif review_required:
        message = (
            f"Found {n} medicine{'s' if n != 1 else ''} but confidence is low "
            f"({overall_confidence:.0%}). Please review and correct any errors.{guest_suffix}"
        )
    elif overall_confidence < SOFT_THRESHOLD:
        message = (
            f"Extracted {n} medicine{'s' if n != 1 else ''}. "
            f"A few items may need verification — highlighted in yellow.{guest_suffix}"
        )
    else:
        message = (
            f"Successfully extracted {n} medicine{'s' if n != 1 else ''} "
            f"with {overall_confidence:.0%} confidence.{guest_suffix}"
        )

    return PrescriptionScanResponse(
        prescription_id=prescription_id,
        image_url=image_url,
        medicines=enriched_medicines,
        overall_confidence=overall_confidence,
        legibility_score=legibility,
        review_required=review_required,
        message=message,
    )


# ── Correction helper (called from router) ────────────────────────────────────

def apply_correction(
    pm_id: int,
    user_id: str,
    correction: dict[str, Any],
) -> bool:
    """
    Apply a user correction to a prescription_medicines row (identified by pm_id).
    Marks was_corrected=True for the data flywheel.
    Returns True on success.
    """
    sb = get_supabase()

    update_payload: dict[str, Any] = {"was_corrected": True}
    update_payload["correction_source"] = correction.pop("correction_source", "user")

    # Only update fields that were actually provided
    allowed = {
        "brand_name", "generic_name", "strength", "dosage_form",
        "dose_instruction", "frequency", "timing", "duration",
    }
    for key in allowed:
        if correction.get(key) is not None:
            update_payload[key] = correction[key]

    try:
        sb.table("prescription_medicines").update(update_payload).eq(
            "id", pm_id
        ).eq("user_id", user_id).execute()
        return True
    except Exception as exc:
        log.error("Correction update failed: %s", exc)
        return False


def confirm_prescription(prescription_id: int, user_id: str) -> bool:
    """Mark a prescription as user-verified → eligible for training data flywheel."""
    sb = get_supabase()
    try:
        sb.table("prescriptions").update({
            "status": "confirmed",
            "verified_by_user": True,
        }).eq("id", prescription_id).eq("user_id", user_id).execute()
        return True
    except Exception as exc:
        log.error("Prescription confirm failed: %s", exc)
        return False
