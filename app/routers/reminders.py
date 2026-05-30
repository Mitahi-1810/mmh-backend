"""
Reminders router — /api/reminders/*

Endpoints:
  Schedules
    POST   /schedules          create one schedule
    POST   /schedules/bulk     create all schedules from an Rx scan
    GET    /schedules          list active schedules
    PATCH  /schedules/{id}     update a schedule
    DELETE /schedules/{id}     deactivate (soft-delete)

  Today's pill tracker
    GET    /today              all dose slots for today + status
    GET    /upcoming           next 5 scheduled doses

  Dose actions
    POST   /doses/{id}/take    mark dose taken
    POST   /doses/{id}/skip    mark dose skipped
    POST   /doses/{id}/snooze  snooze N minutes

  Adherence
    GET    /adherence          30-day stats + daily calendar data
    GET    /refill-alerts      schedules running low on stock

  Notifications
    POST   /push-subscription         register a Web Push subscription
    DELETE /push-subscription         remove it
    GET    /push-public-key           VAPID public key for the frontend
    PATCH  /whatsapp                  update WhatsApp opt-in / phone number
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import get_user_id
from app.config import get_settings
from app.models import (
    BulkScheduleCreateRequest,
    DoseActionRequest,
    MedicationScheduleOut,
    PushSubscriptionRequest,
    ScheduleCreateRequest,
    SnoozeRequest,
    TodayDose,
    WhatsAppSettingsRequest,
)
from app.services.push_service import (
    delete_push_subscription,
    save_push_subscription,
    send_push_to_user,
    build_dose_notification,
)
from app.services.reminder_service import (
    create_schedule,
    create_schedules_bulk,
    deactivate_schedule,
    get_adherence_stats,
    get_refill_alerts,
    get_todays_doses,
    get_upcoming_doses,
    list_schedules,
    mark_dose_skipped,
    mark_dose_taken,
    snooze_dose,
    update_schedule,
)
from app.database import get_supabase

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reminders", tags=["reminders"])


# ── Auth dependency ───────────────────────────────────────────────────────────

def require_user(user_id: str = Depends(get_user_id)) -> str:
    return user_id


# ── Schedules ─────────────────────────────────────────────────────────────────

@router.post("/schedules", response_model=MedicationScheduleOut, status_code=201)
async def create_schedule_endpoint(
    req: ScheduleCreateRequest,
    user_id: str = Depends(require_user),
):
    """Create a single medication schedule (manual or from a corrected Rx line)."""
    result = create_schedule(user_id, req)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create schedule")
    # Generate today's dose log immediately so it shows up in /today
    from app.services.reminder_service import generate_dose_logs_for_date
    generate_dose_logs_for_date()
    return result


@router.post("/schedules/bulk", status_code=201)
async def create_schedules_bulk_endpoint(
    req: BulkScheduleCreateRequest,
    user_id: str = Depends(require_user),
) -> dict[str, Any]:
    """
    Create all medication schedules from a confirmed prescription scan.
    Called right after the user taps "Set up reminders" in PrescriptionScanner.
    """
    results = create_schedules_bulk(user_id, req.prescription_id, req.schedules)
    from app.services.reminder_service import generate_dose_logs_for_date
    generate_dose_logs_for_date()
    return {
        "created": len(results),
        "schedules": [s.model_dump() for s in results],
    }


@router.get("/schedules", response_model=list[MedicationScheduleOut])
async def list_schedules_endpoint(user_id: str = Depends(require_user)):
    """List all active medication schedules for the logged-in user."""
    return list_schedules(user_id)


@router.patch("/schedules/{schedule_id}", response_model=MedicationScheduleOut)
async def update_schedule_endpoint(
    schedule_id: int,
    updates: dict[str, Any],
    user_id: str = Depends(require_user),
):
    result = update_schedule(schedule_id, user_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return result


@router.delete("/schedules/{schedule_id}", status_code=204)
async def delete_schedule_endpoint(
    schedule_id: int,
    user_id: str = Depends(require_user),
):
    ok = deactivate_schedule(schedule_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Schedule not found")


# ── Today's pill tracker ──────────────────────────────────────────────────────

@router.get("/today", response_model=list[TodayDose])
async def get_today_endpoint(user_id: str = Depends(require_user)):
    """
    Return today's complete dose timeline for the pill tracker UI.
    Generates missing dose_logs on-the-fly if the nightly job was missed.
    """
    return get_todays_doses(user_id)


@router.get("/upcoming", response_model=list[TodayDose])
async def get_upcoming_endpoint(
    limit: int = 5,
    user_id: str = Depends(require_user),
):
    """Return the next N scheduled / snoozed doses (cross-day)."""
    return get_upcoming_doses(user_id, limit=min(limit, 20))


# ── Dose actions ──────────────────────────────────────────────────────────────

@router.post("/doses/{dose_log_id}/take", status_code=200)
async def take_dose_endpoint(
    dose_log_id: int,
    body: DoseActionRequest,
    user_id: str = Depends(require_user),
) -> dict[str, bool]:
    ok = mark_dose_taken(dose_log_id, user_id, body.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Dose log not found")
    return {"ok": True}


@router.post("/doses/{dose_log_id}/skip", status_code=200)
async def skip_dose_endpoint(
    dose_log_id: int,
    body: DoseActionRequest,
    user_id: str = Depends(require_user),
) -> dict[str, bool]:
    ok = mark_dose_skipped(dose_log_id, user_id, body.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Dose log not found")
    return {"ok": True}


@router.post("/doses/{dose_log_id}/snooze", status_code=200)
async def snooze_dose_endpoint(
    dose_log_id: int,
    body: SnoozeRequest,
    user_id: str = Depends(require_user),
) -> dict[str, Any]:
    ok = snooze_dose(dose_log_id, user_id, body.minutes, body.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Dose log not found")
    from app.services.reminder_service import _utc_now
    from datetime import timedelta
    snoozed_until = (_utc_now() + timedelta(minutes=body.minutes)).isoformat()
    return {"ok": True, "snoozed_until": snoozed_until}


# ── Adherence ─────────────────────────────────────────────────────────────────

@router.get("/adherence")
async def get_adherence_endpoint(
    days: int = 30,
    user_id: str = Depends(require_user),
):
    """Return adherence stats + per-day calendar data for the last N days (max 90)."""
    period = min(max(days, 7), 90)
    return get_adherence_stats(user_id, period)


@router.get("/refill-alerts")
async def get_refill_alerts_endpoint(user_id: str = Depends(require_user)):
    """Return schedules that are running low on remaining quantity."""
    return get_refill_alerts(user_id)


# ── Push notifications ────────────────────────────────────────────────────────

@router.get("/push-public-key")
async def get_push_public_key():
    """Return the VAPID public key so the frontend can subscribe."""
    key = get_settings().vapid_public_key
    if not key:
        raise HTTPException(status_code=503, detail="Push notifications not configured")
    return {"public_key": key}


@router.post("/push-subscription", status_code=201)
async def register_push_subscription(
    body: PushSubscriptionRequest,
    request: Request,
    user_id: str = Depends(get_user_id),
) -> dict[str, bool]:
    ua = request.headers.get("User-Agent", "")
    ok = save_push_subscription(
        user_id=user_id,
        endpoint=body.endpoint,
        p256dh=body.p256dh,
        auth_key=body.auth_key,
        platform=body.platform,
        user_agent=ua[:255] if ua else None,
    )
    return {"ok": ok}


@router.delete("/push-subscription", status_code=204)
async def unregister_push_subscription(
    body: PushSubscriptionRequest,
    user_id: str = Depends(require_user),
):
    delete_push_subscription(user_id, body.endpoint)


# ── WhatsApp settings ─────────────────────────────────────────────────────────

@router.patch("/whatsapp", status_code=200)
async def update_whatsapp_settings(
    body: WhatsAppSettingsRequest,
    user_id: str = Depends(require_user),
) -> dict[str, bool]:
    sb = get_supabase()
    try:
        sb.table("whatsapp_settings").upsert({
            "user_id": user_id,
            "phone_number": body.phone_number,
            "opted_in": body.opted_in,
            "verified": False,  # phone number changed → re-verify
        }, on_conflict="user_id").execute()
        return {"ok": True}
    except Exception as exc:
        log.error("update_whatsapp_settings failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update settings")


@router.get("/whatsapp", status_code=200)
async def get_whatsapp_settings_endpoint(user_id: str = Depends(require_user)):
    sb = get_supabase()
    try:
        res = (
            sb.table("whatsapp_settings")
            .select("phone_number, verified, opted_in")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        return res.data or {"phone_number": None, "verified": False, "opted_in": False}
    except Exception:
        return {"phone_number": None, "verified": False, "opted_in": False}


# ── Demo / test endpoint ──────────────────────────────────────────────────────

@router.post("/test-whatsapp", status_code=200)
async def test_whatsapp(
    body: dict,
    user_id: str = Depends(require_user),
) -> dict[str, Any]:
    """
    Send a test WhatsApp message to a given number directly.
    Used for demo purposes — bypasses whatsapp_settings table.

    Body: { "to": "+8801XXXXXXXXX", "message": "Hello from mmh.io!" }
    """
    from app.config import get_settings
    from app.services.push_service import send_whatsapp_direct

    settings = get_settings()
    if not settings.whatsapp_enabled:
        raise HTTPException(
            status_code=503,
            detail="Set WHATSAPP_ENABLED=true and CALLMEBOT_API_KEY in .env"
        )

    to      = body.get("to", "").strip()
    message = body.get("message", "💊 mmh.io reminder — time to take your medicine!")

    if not to:
        raise HTTPException(status_code=400, detail="'to' required (e.g. +8801XXXXXXXXX)")

    ok = await send_whatsapp_direct(to, message)
    if not ok:
        raise HTTPException(status_code=502, detail="WhatsApp delivery failed — check logs")
    return {"ok": True, "to": to, "provider": settings.whatsapp_provider}
