"""
APScheduler cron jobs:
  1. Nightly Medex diff-scraper (re-scrape changed pages)
  2. Morning dose-log generator  (5:30 AM BST — pre-generate today's dose_logs)
  3. Next-day pre-generation     (11:30 PM BST — pre-generate tomorrow's dose_logs)
  4. Dose reminder sender        (every 2 min — fire push for upcoming doses)
  5. Missed-dose detector        (every 10 min — mark stale 'scheduled' as 'missed')
  6. Adherence materialization   (11:00 PM BST — compute adherence_daily for today)
  7. Snoozed dose re-notifier    (every 2 min — re-fire push for expired snooze)
  8. Refill alerts                (9:00 AM BST daily — warn on low stock)
  9. Streak milestone notifier   (8:00 AM BST daily — celebrate streaks)
"""
import logging
from datetime import timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler

log = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()


# ── 1. Nightly Medex diff-scraper ─────────────────────────────────────────────

@scheduler.scheduled_job("cron", hour=2, minute=0, id="nightly_medex_diff",
                          timezone="Asia/Dhaka")
async def nightly_medex_diff():
    """Trigger the diff scraper — only re-scrapes pages changed since last run."""
    import subprocess, sys
    subprocess.Popen([sys.executable, "scripts/seed_medex.py", "--diff-only"])


# ── 2 & 3. Dose-log generator ─────────────────────────────────────────────────

@scheduler.scheduled_job("cron", hour=5, minute=30, id="generate_today_dose_logs",
                          timezone="Asia/Dhaka")
async def generate_today_dose_logs():
    """Pre-generate today's dose_log rows at 5:30 AM BST so they're ready for push."""
    try:
        from app.services.reminder_service import generate_dose_logs_for_date
        count = generate_dose_logs_for_date()
        log.info("Morning dose-log generation: %d new rows", count)
    except Exception as exc:
        log.error("generate_today_dose_logs failed: %s", exc)


@scheduler.scheduled_job("cron", hour=23, minute=30, id="generate_tomorrow_dose_logs",
                          timezone="Asia/Dhaka")
async def generate_tomorrow_dose_logs():
    """Pre-generate tomorrow's dose_logs at 11:30 PM BST (helps midnight-dose users)."""
    try:
        from datetime import date, timedelta
        from app.services.reminder_service import generate_dose_logs_for_date, _local_date_today
        tomorrow = _local_date_today() + timedelta(days=1)
        count = generate_dose_logs_for_date(tomorrow)
        log.info("Tomorrow dose-log pre-generation: %d new rows", count)
    except Exception as exc:
        log.error("generate_tomorrow_dose_logs failed: %s", exc)


# ── 4. Dose reminder push sender ──────────────────────────────────────────────

@scheduler.scheduled_job("interval", minutes=2, id="send_dose_reminders")
async def send_dose_reminders():
    """
    Every 2 minutes: fire Web Push + WhatsApp for doses due within ±2 min.
    This tight window prevents double-firing while keeping latency under 4 min.
    """
    try:
        from app.services.reminder_service import get_due_doses_for_push
        from app.services.push_service import send_push_to_user, send_whatsapp_to_user, build_dose_notification

        due = get_due_doses_for_push(window_seconds=120)
        for row in due:
            sched = row.get("medication_schedules") or {}
            if not sched.get("notify_push", True):
                continue

            title, body = build_dose_notification(
                brand_name=sched.get("brand_name", "Medicine"),
                strength=sched.get("strength"),
                dose_instruction=sched.get("dose_instruction"),
                timing=sched.get("timing"),
                dose_log_id=row["id"],
            )
            sent = send_push_to_user(
                user_id=row["user_id"],
                title=title,
                body=body,
                data={"dose_log_id": row["id"], "action": "dose_due"},
                tag=f"dose-{row['id']}",
            )
            if sent:
                log.debug("Push sent for dose_log %s to user %s", row["id"], row["user_id"])

            # WhatsApp (parallel, if opted in)
            if sched.get("notify_whatsapp"):
                await send_whatsapp_to_user(row["user_id"], f"{title}\n{body}")

    except Exception as exc:
        log.error("send_dose_reminders failed: %s", exc)


# ── 5. Snoozed dose re-notifier ───────────────────────────────────────────────

@scheduler.scheduled_job("interval", minutes=2, id="send_snoozed_reminders")
async def send_snoozed_reminders():
    """Re-fire push for doses whose snooze timer has expired."""
    try:
        from app.services.reminder_service import get_due_snoozed_doses
        from app.services.push_service import send_push_to_user, build_dose_notification
        from app.database import get_supabase
        from app.services.reminder_service import _utc_now
        from datetime import timedelta

        snoozed = get_due_snoozed_doses()
        for row in snoozed:
            sched = row.get("medication_schedules") or {}
            title, body = build_dose_notification(
                brand_name=sched.get("brand_name", "Medicine"),
                strength=sched.get("strength"),
                dose_instruction=sched.get("dose_instruction"),
                timing=sched.get("timing"),
                dose_log_id=row["id"],
            )
            send_push_to_user(
                user_id=row["user_id"],
                title=f"⏰ Reminder: {title}",
                body=body,
                data={"dose_log_id": row["id"], "action": "snooze_expired"},
                tag=f"snooze-{row['id']}",
            )
            # Reset status back to scheduled so user can take/skip/snooze again
            sb = get_supabase()
            sb.table("dose_logs").update({
                "status": "scheduled",
                "snoozed_until": None,
            }).eq("id", row["id"]).execute()

    except Exception as exc:
        log.error("send_snoozed_reminders failed: %s", exc)


# ── 6. Missed-dose detector ───────────────────────────────────────────────────

@scheduler.scheduled_job("interval", minutes=10, id="mark_missed_doses")
async def mark_missed_doses():
    """
    Every 10 minutes: mark doses as 'missed' if they were due >60 min ago
    and were never taken. Sends a gentle missed-dose notification.
    """
    try:
        from app.services.reminder_service import mark_overdue_as_missed
        count = mark_overdue_as_missed(grace_minutes=60)
        if count:
            log.info("Marked %d doses as missed", count)
    except Exception as exc:
        log.error("mark_missed_doses failed: %s", exc)


# ── 7. Adherence materialization ──────────────────────────────────────────────

@scheduler.scheduled_job("cron", hour=23, minute=0, id="compute_adherence",
                          timezone="Asia/Dhaka")
async def compute_adherence():
    """At 11 PM BST: compute adherence_daily for today."""
    try:
        from app.services.reminder_service import compute_adherence_daily
        ok = compute_adherence_daily()
        log.info("Adherence materialization: %s", "ok" if ok else "failed")
    except Exception as exc:
        log.error("compute_adherence failed: %s", exc)


# ── 8. Refill alerts ──────────────────────────────────────────────────────────

@scheduler.scheduled_job("cron", hour=9, minute=0, id="send_refill_alerts",
                          timezone="Asia/Dhaka")
async def send_refill_alerts():
    """9 AM BST: notify users whose medicine stock is running low."""
    try:
        from app.database import get_supabase
        from app.services.reminder_service import get_refill_alerts
        from app.services.push_service import send_push_to_user

        sb = get_supabase()
        # Get all users with active schedules
        res = (
            sb.table("medication_schedules")
            .select("user_id")
            .eq("is_active", True)
            .execute()
        )
        user_ids = list({r["user_id"] for r in (res.data or [])})

        for uid in user_ids:
            alerts = get_refill_alerts(uid)
            for alert in alerts:
                name = alert["brand_name"]
                days = alert["days_left"]
                send_push_to_user(
                    user_id=uid,
                    title=f"💊 Refill needed: {name}",
                    body=f"Only {days} day{'s' if days != 1 else ''} of {name} left. Time to refill!",
                    data={"schedule_id": alert["schedule_id"], "action": "refill_alert"},
                    tag=f"refill-{alert['schedule_id']}",
                )
    except Exception as exc:
        log.error("send_refill_alerts failed: %s", exc)


# ── 9. Streak milestone notifier ──────────────────────────────────────────────

@scheduler.scheduled_job("cron", hour=8, minute=0, id="streak_notifications",
                          timezone="Asia/Dhaka")
async def streak_notifications():
    """8 AM BST: celebrate streak milestones (7, 14, 30, 60, 100 days)."""
    MILESTONES = {7, 14, 30, 60, 100}
    try:
        from app.database import get_supabase
        from app.services.reminder_service import get_adherence_stats
        from app.services.push_service import send_push_to_user, build_streak_notification

        sb = get_supabase()
        res = (
            sb.table("medication_schedules")
            .select("user_id")
            .eq("is_active", True)
            .execute()
        )
        user_ids = list({r["user_id"] for r in (res.data or [])})

        for uid in user_ids:
            stats = get_adherence_stats(uid, period_days=100)
            if stats.streak_days in MILESTONES:
                title, body = build_streak_notification(stats.streak_days)
                send_push_to_user(
                    user_id=uid,
                    title=title,
                    body=body,
                    data={"streak": stats.streak_days, "action": "streak_milestone"},
                    tag="streak-milestone",
                )
    except Exception as exc:
        log.error("streak_notifications failed: %s", exc)


# ── Startup ───────────────────────────────────────────────────────────────────

def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        log.info("APScheduler started with %d jobs", len(scheduler.get_jobs()))
