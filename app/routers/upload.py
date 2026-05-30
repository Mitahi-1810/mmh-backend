"""
/api/upload — prescription OCR (V7 pipeline) and expiry date scanner.

Endpoints:
  POST   /api/upload/prescription                          → run V7 pipeline, return PrescriptionScanResponse
  POST   /api/upload/prescription/{prescription_id}/confirm → mark as verified (training flywheel)
  PATCH  /api/upload/prescription/{prescription_id}/medicines/{med_id} → correct one medicine line
  POST   /api/upload/expiry                                → extract expiry date + create reminder
"""
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import get_optional_user_id
from app.database import get_supabase
from app.models import (
    ChatResponseEnvelope,
    MedicineCorrectionRequest,
    PrescriptionScanResponse,
)
from app.services.gemini import ocr_expiry
from app.services.prescription import (
    apply_correction,
    confirm_prescription,
    run_prescription_pipeline,
)

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Browsers sometimes send image/jpg (non-standard) — accept it
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# Normalise non-standard MIME types before passing to Gemini / Supabase
_MIME_NORM = {"image/jpg": "image/jpeg"}


def _normalise_mime(ct: str | None) -> str:
    ct = (ct or "image/jpeg").lower().split(";")[0].strip()
    return _MIME_NORM.get(ct, ct)


def _validate_image(file: UploadFile) -> None:
    ct = _normalise_mime(file.content_type)
    if ct not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WebP images are accepted.")


# ── Prescription scan (V7) ────────────────────────────────────────────────────

@router.post("/prescription", response_model=ChatResponseEnvelope)
async def upload_prescription(
    file: UploadFile = File(...),
    user_id: str | None = Depends(get_optional_user_id),
):
    """
    Upload a prescription image.
    Logged-in users: full V7 pipeline (OCR → DB match → persist → training flywheel).
    Guests: OCR + DB match only — results are NOT persisted (no prescription_id in response).
    """
    _validate_image(file)

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be under 10MB.")

    result: PrescriptionScanResponse = await run_prescription_pipeline(
        user_id=user_id,          # None → pipeline skips DB writes
        image_bytes=data,
        content_type=_normalise_mime(file.content_type),
    )

    return ChatResponseEnvelope(
        message=result.message,
        format="prescription_scan",
        data=result.model_dump(),
    )


@router.post("/prescription/{prescription_id}/confirm")
async def confirm_prescription_endpoint(
    prescription_id: int,
    user_id: str | None = Depends(get_optional_user_id),
):
    """Mark a prescription as verified (requires the scan to have been persisted — logged-in only)."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in to confirm prescriptions.")
    ok = confirm_prescription(prescription_id, user_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not confirm prescription.")
    return {"message": "Prescription confirmed. Thank you for helping improve accuracy!"}


@router.patch("/prescription/{prescription_id}/medicines/{pm_id}")
async def correct_medicine_line(
    prescription_id: int,
    pm_id: int,
    correction: MedicineCorrectionRequest,
    user_id: str | None = Depends(get_optional_user_id),
):
    """Correct a single medicine line (logged-in only, as the row must exist in DB)."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in to save corrections.")
    ok = apply_correction(
        pm_id=pm_id,
        user_id=user_id,
        correction=correction.model_dump(exclude_none=True),
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Could not save correction.")
    return {"message": "Correction saved. Thank you!"}


# ── Expiry scan ───────────────────────────────────────────────────────────────

def _store_file(data: bytes, bucket: str, content_type: str) -> str:
    """Upload bytes to Supabase Storage, return the public URL."""
    sb = get_supabase()
    path = str(uuid.uuid4())
    sb.storage.from_(bucket).upload(
        path=path,
        file=data,
        file_options={"content-type": content_type},
    )
    return sb.storage.from_(bucket).get_public_url(path)


@router.post("/expiry", response_model=ChatResponseEnvelope)
async def upload_expiry(
    file: UploadFile = File(...),
    medicine_name: str = "",
    user_id: str | None = Depends(get_optional_user_id),
):
    """
    Upload a medicine packaging photo.
    Extracts the expiry date and creates a reminder 30 days before.
    """
    _validate_image(file)

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be under 10MB.")

    ct = _normalise_mime(file.content_type)
    _store_file(data, "expiry-scans", ct)
    result = await ocr_expiry(data, mime_type=ct)

    expiry_date = result.get("expiry_date")
    if expiry_date:
        remind_at = (
            datetime.fromisoformat(expiry_date) - timedelta(days=30)
        ).isoformat()

        # Only persist the reminder if the user is logged in
        if user_id:
            sb = get_supabase()
            sb.table("user_reminders").insert({
                "user_id": user_id,
                "medicine_name": medicine_name or "Unknown medicine",
                "remind_at": remind_at,
                "note": f"Expires on {expiry_date}",
                "type": "expiry",
            }).execute()

        suffix = "" if user_id else " (Sign in to save reminders.)"
        return ChatResponseEnvelope(
            message=(
                f"Expiry date detected: {expiry_date}. "
                f"Reminder: {remind_at[:10]} (30 days before expiry).{suffix}"
            ),
            format="reminder_confirm",
            data={
                "medicine": medicine_name,
                "remind_at": remind_at,
                "note": f"Expires {expiry_date}",
            },
        )

    return ChatResponseEnvelope(
        message="Could not detect an expiry date. Please try a clearer photo of the packaging.",
        format="text",
        data=None,
    )
