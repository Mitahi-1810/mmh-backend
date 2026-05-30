"""
Reminder service — core business logic for the medication schedule system.

Responsibilities:
  • Generate dose_logs for a given date from active medication_schedules
  • CRUD for medication_schedules
  • Dose actions: take / skip / snooze
  • Missed dose detection (run by scheduler every 10 min)
  • Adherence stats + daily materialization
  • Refill alert detection

Timezone strategy:
  BD app is primarily BST (UTC+6). dose_times are stored as "HH:MM" local time.
  All TIMESTAMPTZ stored in Supabase are UTC.
  This module converts BST → UTC when writing and UTC → BST when reading.
"""
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.config import get_settings
from app.database import get_supabase
from app.models import (
    AdherenceDayStats,
    AdherenceStats,
    MedicationScheduleOut,
    ScheduleCreateRequest,
    TodayDose,
)

log = logging.getLogger(__name__)


# ── Timezone helpers ──────────────────────────────────────────────────────────

def _local_offset() -> timedelta:
    return timedelta(hours=get_settings().tz_offset_hours)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _local_now() -> datetime:
    return _utc_now() + _local_offset()


def _local_date_today() -> date:
    return _local_now().date()


def _dose_time_to_utc(local_date: date, time_str: str) -> datetime:
    """Convert "HH:MM" on a local date to UTC datetime."""
    h, m = map(int, time_str.split(":"))
    local_dt = datetime(local_date.year, local_date.month, local_date.day, h, m,
                        tzinfo=timezone.utc) - _local_offset()
    return local_dt


def _fmt_time_label(utc_dt_str: str) -> str:
    """Convert ISO UTC timestamp → "08:00 AM" local display label."""
    try:
        dt = datetime.fromisoformat(utc_dt_str.replace("Z", "+00:00"))
        local = dt + _local_offset()
        return local.strftime("%I:%M %p").lstrip("0") or "12:00 AM"
    except Exception:
        return utc_dt_str


# ── Schedule CRUD ─────────────────────────────────────────────────────────────

def create_schedule(user_id: str, req: ScheduleCreateRequest) -> MedicationScheduleOut | None:
    sb = get_supabase()
    start = req.start_date or _local_date_today().isoformat()
    try:
        res = sb.table("medication_schedules").insert({
            "user_id": user_id,
            "prescription_id": req.prescription_id,
            "pm_id": req.pm_id,
            "brand_name": req.brand_name,
            "generic_name": req.generic_name,
            "strength": req.strength,
            "dosage_form": req.dosage_form,
            "frequency": req.frequency,
            "dose_times": req.dose_times,
            "timing": req.timing,
            "dose_instruction": req.dose_instruction,
            "start_date": start,
            "end_date": req.end_date,
            "total_quantity": req.total_quantity,
            "remaining_quantity": req.total_quantity,
            "refill_alert_days": req.refill_alert_days,
            "notify_push": req.notify_push,
            "notify_whatsapp": req.notify_whatsapp,
            "snooze_minutes": req.snooze_minutes,
            "is_active": True,
        }).execute()
        if res.data:
            return _row_to_schedule(res.data[0])
    except Exception as exc:
        log.error("create_schedule failed: %s", exc)
    return None


def create_schedules_bulk(
    user_id: str,
    prescription_id: int,
    requests: list[ScheduleCreateRequest],
) -> list[MedicationScheduleOut]:
    results = []
    for req in requests:
        req.prescription_id = prescription_id
        sched = create_schedule(user_id, req)
        if sched:
            results.append(sched)
    return results


def list_schedules(user_id: str, active_only: bool = True) -> list[MedicationScheduleOut]:
    sb = get_supabase()
    try:
        q = sb.table("medication_schedules").select("*").eq("user_id", user_id)
        if active_only:
            q = q.eq("is_active", True)
        res = q.order("created_at", desc=False).execute()
        return [_row_to_schedule(r) for r in (res.data or [])]
    except Exception as exc:
        log.error("list_schedules failed: %s", exc)
        return []


def update_schedule(
    schedule_id: int, user_id: str, updates: dict[str, Any]
) -> MedicationScheduleOut | None:
    sb = get_supabase()
    # Protect fields that should not be user-editable
    updates.pop("id", None)
    updates.pop("user_id", None)
    updates["updated_at"] = _utc_now().isoformat()
    try:
        res = (
            sb.table("medication_schedules")
            .update(updates)
            .eq("id", schedule_id)
            .eq("user_id", user_id)
            .execute()
        )
        if res.data:
            return _row_to_schedule(res.data[0])
    except Exception as exc:
        log.error("update_schedule failed: %s", exc)
    return None


def deactivate_schedule(schedule_id: int, user_id: str) -> bool:
    """Soft-delete: set is_active=False."""
    sb = get_supabase()
    try:
        sb.table("medication_schedules").update({"is_active": False}).eq(
            "id", schedule_id
        ).eq("user_id", user_id).execute()
        return True
    except Exception as exc:
        log.error("deactivate_schedule failed: %s", exc)
        return False


def _row_to_schedule(row: dict) -> MedicationScheduleOut:
    dose_times = row.get("dose_times") or ["08:00"]
    if isinstance(dose_times, str):
        import json
        dose_times = json.loads(dose_times)
    return MedicationScheduleOut(
        id=row["id"],
        user_id=row["user_id"],
        prescription_id=row.get("prescription_id"),
        pm_id=row.get("pm_id"),
        brand_name=row["brand_name"],
        generic_name=row.get("generic_name"),
        strength=row.get("strength"),
        dosage_form=row.get("dosage_form"),
        frequency=row["frequency"],
        dose_times=dose_times,
        timing=row.get("timing"),
        dose_instruction=row.get("dose_instruction"),
        start_date=str(row["start_date"]),
        end_date=str(row["end_date"]) if row.get("end_date") else None,
        total_quantity=row.get("total_quantity"),
        remaining_quantity=row.get("remaining_quantity"),
        refill_alert_days=row.get("refill_alert_days", 5),
        notify_push=row.get("notify_push", True),
        notify_whatsapp=row.get("notify_whatsapp", False),
        snooze_minutes=row.get("snooze_minutes", 15),
        is_active=row.get("is_active", True),
        created_at=str(row.get("created_at", "")),
    )


# ── Dose log generation ───────────────────────────────────────────────────────

def generate_dose_logs_for_date(target_date: date | None = None) -> int:
    """
    Create dose_log rows for all active schedules on target_date (default: today).
    Safe to call multiple times — ON CONFLICT DO NOTHING (unique constraint).
    Returns count of new rows inserted.
    """
    target = target_date or _local_date_today()
    sb = get_supabase()

    # Fetch all active schedules that cover target_date
    try:
        res = (
            sb.table("medication_schedules")
            .select("id, user_id, dose_times, start_date, end_date, notify_push, notify_whatsapp")
            .eq("is_active", True)
            .lte("start_date", target.isoformat())
            .execute()
        )
    except Exception as exc:
        log.error("generate_dose_logs: fetch schedules failed: %s", exc)
        return 0

    schedules = [
        r for r in (res.data or [])
        if r.get("end_date") is None or r["end_date"] >= target.isoformat()
    ]

    if not schedules:
        log.debug("generate_dose_logs: no active schedules for %s", target)
        return 0

    rows = []
    for s in schedules:
        dose_times = s.get("dose_times") or ["08:00"]
        if isinstance(dose_times, str):
            import json
            dose_times = json.loads(dose_times)
        for t in dose_times:
            try:
                scheduled_utc = _dose_time_to_utc(target, t)
                rows.append({
                    "schedule_id": s["id"],
                    "user_id": s["user_id"],
                    "scheduled_at": scheduled_utc.isoformat(),
                    "status": "scheduled",
                })
            except Exception as exc:
                log.warning("Bad dose_time '%s' in schedule %s: %s", t, s["id"], exc)

    if not rows:
        return 0

    try:
        # upsert with ON CONFLICT DO NOTHING (unique on schedule_id + scheduled_at)
        res2 = (
            sb.table("dose_logs")
            .upsert(rows, on_conflict="schedule_id,scheduled_at", ignore_duplicates=True)
            .execute()
        )
        count = len(res2.data or [])
        log.info("generate_dose_logs[%s]: %d new rows", target, count)
        return count
    except Exception as exc:
        log.error("generate_dose_logs: insert failed: %s", exc)
        return 0


# ── Today's pill tracker ──────────────────────────────────────────────────────

def get_todays_doses(user_id: str) -> list[TodayDose]:
    """
    Return all dose slots for today, ordered by scheduled_at.
    Generates any missing dose_logs on-the-fly (catches missed generation).
    """
    # Ensure today's logs exist
    generate_dose_logs_for_date()

    sb = get_supabase()
    today_local = _local_date_today()

    # UTC window for today in BST: today 00:00 BST → today 23:59 BST
    day_start_utc = _dose_time_to_utc(today_local, "00:00")
    day_end_utc   = _dose_time_to_utc(today_local, "23:59")

    try:
        res = (
            sb.table("dose_logs")
            .select(
                "id, schedule_id, user_id, scheduled_at, taken_at, snoozed_until, status, note, "
                "medication_schedules(brand_name, generic_name, strength, dosage_form, "
                "dose_instruction, timing)"
            )
            .eq("user_id", user_id)
            .gte("scheduled_at", day_start_utc.isoformat())
            .lte("scheduled_at", day_end_utc.isoformat())
            .order("scheduled_at")
            .execute()
        )
    except Exception as exc:
        log.error("get_todays_doses failed: %s", exc)
        return []

    now_utc = _utc_now()
    doses: list[TodayDose] = []

    for row in (res.data or []):
        sched = row.get("medication_schedules") or {}
        scheduled_at = row["scheduled_at"]
        try:
            sched_dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        except Exception:
            sched_dt = now_utc

        is_overdue = (
            row["status"] == "scheduled"
            and sched_dt < now_utc
        )

        doses.append(TodayDose(
            dose_log_id=row["id"],
            schedule_id=row["schedule_id"],
            brand_name=sched.get("brand_name", "Medicine"),
            generic_name=sched.get("generic_name"),
            strength=sched.get("strength"),
            dosage_form=sched.get("dosage_form"),
            dose_instruction=sched.get("dose_instruction"),
            timing=sched.get("timing"),
            scheduled_at=scheduled_at,
            taken_at=row.get("taken_at"),
            snoozed_until=row.get("snoozed_until"),
            status=row["status"],
            time_label=_fmt_time_label(scheduled_at),
            is_overdue=is_overdue,
            note=row.get("note"),
        ))

    return doses


# ── Dose actions ──────────────────────────────────────────────────────────────

def _update_dose_log(
    dose_log_id: int,
    user_id: str,
    updates: dict[str, Any],
) -> bool:
    sb = get_supabase()
    try:
        sb.table("dose_logs").update(updates).eq("id", dose_log_id).eq(
            "user_id", user_id
        ).execute()
        return True
    except Exception as exc:
        log.error("_update_dose_log failed: %s", exc)
        return False


def _decrement_remaining(schedule_id: int) -> None:
    """Decrement remaining_quantity by 1 (stop at 0)."""
    sb = get_supabase()
    try:
        # Fetch current value
        res = (
            sb.table("medication_schedules")
            .select("remaining_quantity")
            .eq("id", schedule_id)
            .single()
            .execute()
        )
        if res.data and res.data.get("remaining_quantity") is not None:
            new_qty = max(0, res.data["remaining_quantity"] - 1)
            sb.table("medication_schedules").update({
                "remaining_quantity": new_qty
            }).eq("id", schedule_id).execute()
    except Exception:
        pass


def mark_dose_taken(dose_log_id: int, user_id: str, note: str | None = None) -> bool:
    now = _utc_now().isoformat()
    ok = _update_dose_log(dose_log_id, user_id, {
        "status": "taken",
        "taken_at": now,
        "snoozed_until": None,
        "note": note,
    })
    if ok:
        # Decrement stock
        sb = get_supabase()
        try:
            res = sb.table("dose_logs").select("schedule_id").eq("id", dose_log_id).single().execute()
            if res.data:
                _decrement_remaining(res.data["schedule_id"])
        except Exception:
            pass
    return ok


def mark_dose_skipped(dose_log_id: int, user_id: str, note: str | None = None) -> bool:
    return _update_dose_log(dose_log_id, user_id, {
        "status": "skipped",
        "note": note,
    })


def snooze_dose(dose_log_id: int, user_id: str, minutes: int = 15, note: str | None = None) -> bool:
    snoozed_until = (_utc_now() + timedelta(minutes=minutes)).isoformat()
    return _update_dose_log(dose_log_id, user_id, {
        "status": "snoozed",
        "snoozed_until": snoozed_until,
        "note": note,
    })


# ── Missed dose detection ─────────────────────────────────────────────────────

def mark_overdue_as_missed(grace_minutes: int = 60) -> int:
    """
    Mark dose_logs as 'missed' if they are:
      - still 'scheduled' (never taken)
      - scheduled_at < now - grace_period
    Called by scheduler every 10 minutes.
    Returns count of newly-missed doses.
    """
    cutoff = (_utc_now() - timedelta(minutes=grace_minutes)).isoformat()
    sb = get_supabase()
    try:
        res = (
            sb.table("dose_logs")
            .update({"status": "missed"})
            .eq("status", "scheduled")
            .lt("scheduled_at", cutoff)
            .execute()
        )
        count = len(res.data or [])
        if count:
            log.info("mark_overdue_as_missed: %d doses marked missed (cutoff=%s)", count, cutoff)
        return count
    except Exception as exc:
        log.error("mark_overdue_as_missed failed: %s", exc)
        return 0


# ── Upcoming doses ────────────────────────────────────────────────────────────

def get_upcoming_doses(user_id: str, limit: int = 5) -> list[TodayDose]:
    """Return the next N scheduled/snoozed doses across all medicines."""
    sb = get_supabase()
    now_utc = _utc_now()
    try:
        res = (
            sb.table("dose_logs")
            .select(
                "id, schedule_id, user_id, scheduled_at, taken_at, snoozed_until, status, note, "
                "medication_schedules(brand_name, generic_name, strength, dosage_form, "
                "dose_instruction, timing)"
            )
            .eq("user_id", user_id)
            .in_("status", ["scheduled", "snoozed"])
            .gte("scheduled_at", now_utc.isoformat())
            .order("scheduled_at")
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        log.error("get_upcoming_doses failed: %s", exc)
        return []

    doses = []
    for row in (res.data or []):
        sched = row.get("medication_schedules") or {}
        doses.append(TodayDose(
            dose_log_id=row["id"],
            schedule_id=row["schedule_id"],
            brand_name=sched.get("brand_name", "Medicine"),
            generic_name=sched.get("generic_name"),
            strength=sched.get("strength"),
            dosage_form=sched.get("dosage_form"),
            dose_instruction=sched.get("dose_instruction"),
            timing=sched.get("timing"),
            scheduled_at=row["scheduled_at"],
            taken_at=row.get("taken_at"),
            snoozed_until=row.get("snoozed_until"),
            status=row["status"],
            time_label=_fmt_time_label(row["scheduled_at"]),
            is_overdue=False,
            note=row.get("note"),
        ))
    return doses


# ── Adherence stats ───────────────────────────────────────────────────────────

def compute_adherence_daily(target_date: date | None = None) -> bool:
    """
    Materialise adherence_daily row for target_date (default: today).
    Called by nightly scheduler.
    """
    target = target_date or _local_date_today()
    sb = get_supabase()

    day_start_utc = _dose_time_to_utc(target, "00:00")
    day_end_utc   = _dose_time_to_utc(target, "23:59")

    try:
        res = (
            sb.table("dose_logs")
            .select("user_id, status")
            .gte("scheduled_at", day_start_utc.isoformat())
            .lte("scheduled_at", day_end_utc.isoformat())
            .execute()
        )
    except Exception as exc:
        log.error("compute_adherence_daily: fetch failed: %s", exc)
        return False

    rows = res.data or []
    if not rows:
        return True  # nothing to materialise

    # Group by user
    from collections import defaultdict
    user_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "taken": 0, "missed": 0, "skipped": 0})

    for row in rows:
        uid = row["user_id"]
        status = row["status"]
        user_stats[uid]["total"] += 1
        if status == "taken":
            user_stats[uid]["taken"] += 1
        elif status == "missed":
            user_stats[uid]["missed"] += 1
        elif status == "skipped":
            user_stats[uid]["skipped"] += 1

    upsert_rows = []
    for uid, s in user_stats.items():
        score = (s["taken"] / s["total"]) if s["total"] > 0 else 0.0
        upsert_rows.append({
            "user_id": uid,
            "date": target.isoformat(),
            "total": s["total"],
            "taken": s["taken"],
            "missed": s["missed"],
            "skipped": s["skipped"],
            "score": round(score, 3),
        })

    try:
        sb.table("adherence_daily").upsert(
            upsert_rows, on_conflict="user_id,date"
        ).execute()
        log.info("compute_adherence_daily[%s]: %d users updated", target, len(upsert_rows))
        return True
    except Exception as exc:
        log.error("compute_adherence_daily: upsert failed: %s", exc)
        return False


def get_adherence_stats(user_id: str, period_days: int = 30) -> AdherenceStats:
    """Return aggregated adherence stats for the last N days."""
    sb = get_supabase()
    today = _local_date_today()
    since = (today - timedelta(days=period_days - 1)).isoformat()

    try:
        res = (
            sb.table("adherence_daily")
            .select("date, total, taken, missed, skipped, score")
            .eq("user_id", user_id)
            .gte("date", since)
            .order("date", desc=False)
            .execute()
        )
    except Exception as exc:
        log.error("get_adherence_stats failed: %s", exc)
        res_data: list = []
    else:
        res_data = res.data or []

    # Build a full date range (fill missing days with zeros)
    daily_map: dict[str, AdherenceDayStats] = {}
    total_total = total_taken = total_missed = total_skipped = 0

    for r in res_data:
        daily_map[r["date"]] = AdherenceDayStats(
            date=r["date"],
            total=r["total"],
            taken=r["taken"],
            missed=r["missed"],
            skipped=r["skipped"],
            score=r.get("score"),
        )
        total_total   += r["total"]
        total_taken   += r["taken"]
        total_missed  += r["missed"]
        total_skipped += r["skipped"]

    # Fill missing dates
    full_daily: list[AdherenceDayStats] = []
    for i in range(period_days):
        d = (today - timedelta(days=period_days - 1 - i)).isoformat()
        full_daily.append(daily_map.get(d, AdherenceDayStats(
            date=d, total=0, taken=0, missed=0, skipped=0, score=None
        )))

    # Calculate current streak
    streak = 0
    for day in reversed(full_daily):
        if day.taken > 0:
            streak += 1
        elif day.total > 0:
            break  # streak broken

    overall_score = (total_taken / total_total) if total_total > 0 else 0.0

    return AdherenceStats(
        period_days=period_days,
        total=total_total,
        taken=total_taken,
        missed=total_missed,
        skipped=total_skipped,
        score=round(overall_score, 3),
        streak_days=streak,
        daily=full_daily,
    )


# ── Refill alerts ─────────────────────────────────────────────────────────────

def get_refill_alerts(user_id: str) -> list[dict[str, Any]]:
    """
    Return schedules where remaining_quantity <= refill_alert_days * doses_per_day.
    """
    sb = get_supabase()
    try:
        res = (
            sb.table("medication_schedules")
            .select("id, brand_name, strength, remaining_quantity, refill_alert_days, dose_times")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .not_.is_("remaining_quantity", "null")
            .execute()
        )
    except Exception as exc:
        log.error("get_refill_alerts failed: %s", exc)
        return []

    alerts = []
    for row in (res.data or []):
        dose_times = row.get("dose_times") or ["08:00"]
        if isinstance(dose_times, str):
            import json
            dose_times = json.loads(dose_times)
        doses_per_day = len(dose_times)
        threshold = row["refill_alert_days"] * doses_per_day
        remaining = row.get("remaining_quantity", 0) or 0
        if remaining <= threshold:
            days_left = remaining // doses_per_day if doses_per_day > 0 else remaining
            alerts.append({
                "schedule_id": row["id"],
                "brand_name": row["brand_name"],
                "strength": row.get("strength"),
                "remaining_quantity": remaining,
                "days_left": days_left,
            })
    return alerts


# ── Dose log query for push service ──────────────────────────────────────────

def get_due_doses_for_push(window_seconds: int = 120) -> list[dict[str, Any]]:
    """
    Return dose_logs where scheduled_at is within ±window_seconds of now
    and status is 'scheduled' (not yet notified / taken).
    Used by the scheduler to fire push notifications.
    """
    sb = get_supabase()
    now = _utc_now()
    lo = (now - timedelta(seconds=window_seconds)).isoformat()
    hi = (now + timedelta(seconds=window_seconds)).isoformat()

    try:
        res = (
            sb.table("dose_logs")
            .select(
                "id, schedule_id, user_id, scheduled_at, "
                "medication_schedules(brand_name, strength, dose_instruction, "
                "timing, notify_push, notify_whatsapp, snooze_minutes)"
            )
            .eq("status", "scheduled")
            .gte("scheduled_at", lo)
            .lte("scheduled_at", hi)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        log.error("get_due_doses_for_push failed: %s", exc)
        return []


def get_due_snoozed_doses() -> list[dict[str, Any]]:
    """Return snoozed doses where snoozed_until <= now (time to re-notify)."""
    sb = get_supabase()
    now = _utc_now().isoformat()
    try:
        res = (
            sb.table("dose_logs")
            .select(
                "id, schedule_id, user_id, snoozed_until, "
                "medication_schedules(brand_name, strength, dose_instruction, "
                "timing, notify_push, snooze_minutes)"
            )
            .eq("status", "snoozed")
            .lte("snoozed_until", now)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        log.error("get_due_snoozed_doses failed: %s", exc)
        return []
