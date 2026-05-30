"""
Push notification delivery — Web Push (VAPID) + WhatsApp Business API stub.

Web Push flow:
  1. Frontend subscribes via browser Push API → sends {endpoint, p256dh, auth_key}
  2. Stored in push_subscriptions table
  3. Scheduler or dose-action event calls send_push_to_user(user_id, ...)
  4. pywebpush signs the payload and POSTs to the push service endpoint
  5. Browser/OS delivers notification via service worker

WhatsApp flow (feature-flagged):
  1. User opts in + verifies phone number in whatsapp_settings
  2. Call send_whatsapp_to_user(user_id, template_name, params)
  3. Uses Meta Graph API /messages endpoint
  4. Requires WHATSAPP_API_TOKEN + WHATSAPP_PHONE_NUMBER_ID in .env
"""
import json
import logging
import httpx
from typing import Any

from app.config import get_settings
from app.database import get_supabase

log = logging.getLogger(__name__)

# ── pywebpush import (optional; graceful fallback if not installed) ────────────
try:
    from pywebpush import webpush, WebPushException
    _WEBPUSH_AVAILABLE = True
except ImportError:
    log.warning("pywebpush not installed — Web Push notifications disabled. "
                "Run: pip install pywebpush")
    _WEBPUSH_AVAILABLE = False
    webpush = None
    WebPushException = Exception


# ── Subscription management ───────────────────────────────────────────────────

def save_push_subscription(
    user_id: str,
    endpoint: str,
    p256dh: str,
    auth_key: str,
    platform: str = "web",
    user_agent: str | None = None,
) -> bool:
    """Upsert a push subscription (on conflict: update last_used)."""
    sb = get_supabase()
    try:
        sb.table("push_subscriptions").upsert({
            "user_id": user_id,
            "endpoint": endpoint,
            "p256dh": p256dh,
            "auth_key": auth_key,
            "platform": platform,
            "user_agent": user_agent,
        }, on_conflict="endpoint").execute()
        return True
    except Exception as exc:
        log.error("save_push_subscription failed: %s", exc)
        return False


def delete_push_subscription(user_id: str, endpoint: str) -> bool:
    sb = get_supabase()
    try:
        sb.table("push_subscriptions").delete().eq("user_id", user_id).eq(
            "endpoint", endpoint
        ).execute()
        return True
    except Exception as exc:
        log.error("delete_push_subscription failed: %s", exc)
        return False


def get_user_push_subscriptions(user_id: str) -> list[dict[str, Any]]:
    sb = get_supabase()
    try:
        res = (
            sb.table("push_subscriptions")
            .select("endpoint, p256dh, auth_key")
            .eq("user_id", user_id)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        log.warning("get_user_push_subscriptions failed: %s", exc)
        return []


# ── Web Push delivery ─────────────────────────────────────────────────────────

def send_web_push(
    endpoint: str,
    p256dh: str,
    auth_key: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
    icon: str = "/icon-192.png",
    badge: str = "/badge-72.png",
    tag: str | None = None,
) -> bool:
    """
    Send a single Web Push notification.
    Returns True on success, False on failure (subscription stale → caller should delete).
    """
    if not _WEBPUSH_AVAILABLE:
        log.debug("Web Push skipped (pywebpush not installed)")
        return False

    settings = get_settings()
    if not settings.vapid_private_key or not settings.vapid_public_key:
        log.warning("VAPID keys not configured — Web Push skipped")
        return False

    payload = json.dumps({
        "notification": {
            "title": title,
            "body": body,
            "icon": icon,
            "badge": badge,
            "tag": tag or "medicine-reminder",
            "renotify": True,
            "vibrate": [200, 100, 200],
            "data": data or {},
            "actions": [
                {"action": "take",  "title": "✅ Taken"},
                {"action": "snooze","title": "⏰ Snooze 15 min"},
            ],
        }
    })

    try:
        webpush(
            subscription_info={
                "endpoint": endpoint,
                "keys": {"p256dh": p256dh, "auth": auth_key},
            },
            data=payload,
            vapid_private_key=settings.vapid_private_key,
            vapid_claims={
                "sub": f"mailto:{settings.vapid_claims_email}",
            },
        )
        return True
    except WebPushException as exc:
        # 410 Gone / 404 Not Found → subscription expired, caller should delete it
        if hasattr(exc, "response") and exc.response and exc.response.status_code in (404, 410):
            log.info("Stale push subscription deleted: %s", endpoint[:40])
            return False
        log.error("Web Push failed for %s: %s", endpoint[:40], exc)
        return False
    except Exception as exc:
        log.error("Web Push unexpected error: %s", exc)
        return False


def send_push_to_user(
    user_id: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
    tag: str | None = None,
) -> int:
    """
    Send Web Push to ALL registered subscriptions for a user.
    Returns count of successful deliveries.
    Stale subscriptions (410/404) are deleted automatically.
    """
    subs = get_user_push_subscriptions(user_id)
    if not subs:
        return 0

    sent = 0
    for sub in subs:
        ok = send_web_push(
            endpoint=sub["endpoint"],
            p256dh=sub["p256dh"],
            auth_key=sub["auth_key"],
            title=title,
            body=body,
            data=data,
            tag=tag,
        )
        if ok:
            sent += 1
        else:
            # Possibly stale — delete so we don't keep retrying
            delete_push_subscription(user_id, sub["endpoint"])
    return sent


# ── WhatsApp delivery ─────────────────────────────────────────────────────────
#
# Provider selection (WHATSAPP_PROVIDER in .env):
#   "callmebot" — instant setup, free, personal use, no business verification
#                 Setup: save +34 644 52 74 82 on WhatsApp, send "I allow callmebot to send me messages"
#                 Needs: CALLMEBOT_API_KEY  (they reply with it in ~1 min)
#
#   "meta"      — Meta Cloud API, requires business verification (takes days)
#                 Needs: WHATSAPP_API_TOKEN + WHATSAPP_PHONE_NUMBER_ID
#
#   "twilio"    — Twilio sandbox, free trial, 5-min setup
#                 Needs: TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_WHATSAPP_FROM
# ─────────────────────────────────────────────────────────────────────────────

def get_whatsapp_settings(user_id: str) -> dict[str, Any] | None:
    sb = get_supabase()
    try:
        res = (
            sb.table("whatsapp_settings")
            .select("phone_number, verified, opted_in")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return res.data
    except Exception:
        return None


async def _send_via_callmebot(phone: str, message: str) -> bool:
    """
    CallMeBot WhatsApp gateway — zero signup, works in 2 minutes.
    https://www.callmebot.com/blog/free-api-whatsapp-messages/
    """
    settings = get_settings()
    api_key = settings.callmebot_api_key
    if not api_key:
        log.warning("CALLMEBOT_API_KEY not set")
        return False

    import urllib.parse
    encoded = urllib.parse.quote(message)
    # Remove + prefix and leading zeros that callmebot doesn't want
    clean_phone = phone.lstrip("+")
    url = f"https://api.callmebot.com/whatsapp.php?phone={clean_phone}&text={encoded}&apikey={api_key}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
        if resp.status_code == 200 and "Message queued" in resp.text:
            log.info("CallMeBot sent to %s", phone)
            return True
        log.warning("CallMeBot response %s: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.error("CallMeBot send failed: %s", exc)
        return False


async def _send_via_twilio(phone: str, message: str) -> bool:
    """Twilio WhatsApp Sandbox — free trial, 5-min setup."""
    settings = get_settings()
    sid   = settings.twilio_account_sid
    token = settings.twilio_auth_token
    from_ = settings.twilio_whatsapp_from   # e.g. "whatsapp:+14155238886"
    if not all([sid, token, from_]):
        log.warning("Twilio credentials not fully set")
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                auth=(sid, token),
                data={
                    "From": from_,
                    "To":   f"whatsapp:{phone}",
                    "Body": message,
                },
            )
        if resp.status_code in (200, 201):
            log.info("Twilio WhatsApp sent to %s", phone)
            return True
        log.warning("Twilio response %s: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.error("Twilio send failed: %s", exc)
        return False


async def _send_via_meta(phone: str, message: str) -> bool:
    """Meta Cloud API — requires business verification."""
    settings = get_settings()
    url = f"https://graph.facebook.com/v19.0/{settings.whatsapp_phone_number_id}/messages"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.whatsapp_api_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": phone,
                    "type": "text",
                    "text": {"body": message},
                },
            )
        if resp.status_code == 200:
            return True
        log.warning("Meta WhatsApp error %s: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.error("Meta WhatsApp send failed: %s", exc)
        return False


async def send_whatsapp_to_user(user_id: str, message: str) -> bool:
    """
    Send a WhatsApp message to a user.
    Provider is selected by WHATSAPP_PROVIDER in .env:
      callmebot  (default — instant setup)
      twilio     (sandbox — 5-min setup)
      meta       (production — days of verification)
    """
    settings = get_settings()
    if not settings.whatsapp_enabled:
        log.debug("WhatsApp disabled — skipping for user %s", user_id)
        return False

    ws = get_whatsapp_settings(user_id)
    if not ws or not ws.get("opted_in"):
        return False

    phone    = ws["phone_number"]
    provider = settings.whatsapp_provider.lower()

    if provider == "callmebot":
        return await _send_via_callmebot(phone, message)
    elif provider == "twilio":
        return await _send_via_twilio(phone, message)
    else:
        return await _send_via_meta(phone, message)


async def send_whatsapp_direct(phone: str, message: str) -> bool:
    """
    Send WhatsApp directly to a phone number (no DB lookup).
    Used by the test endpoint and the demo flow.
    """
    settings = get_settings()
    provider = settings.whatsapp_provider.lower()

    if provider == "callmebot":
        return await _send_via_callmebot(phone, message)
    elif provider == "twilio":
        return await _send_via_twilio(phone, message)
    else:
        return await _send_via_meta(phone, message)


# ── Notification content helpers ──────────────────────────────────────────────

def build_dose_notification(
    brand_name: str,
    strength: str | None,
    dose_instruction: str | None,
    timing: str | None,
    dose_log_id: int,
) -> tuple[str, str]:
    """Returns (title, body) for a dose-due push notification."""
    strength_str = f" {strength}" if strength else ""
    dose_str = f" — {dose_instruction}" if dose_instruction else ""
    timing_str = ""
    if timing:
        t = timing.replace("_", " ")
        timing_str = f" ({t})"

    title = f"💊 Time for {brand_name}{strength_str}"
    body = f"Take your dose now{dose_str}{timing_str}. Tap to confirm."
    return title, body


def build_missed_notification(brand_name: str, missed_count: int) -> tuple[str, str]:
    if missed_count == 1:
        return "⚠️ Missed dose", f"You missed {brand_name}. Your streak may break."
    return "⚠️ Missed doses", f"You missed {missed_count} doses including {brand_name}."


def build_streak_notification(streak_days: int) -> tuple[str, str]:
    if streak_days == 7:
        return "🔥 1-week streak!", "Amazing! You've taken all medicines for 7 days straight."
    if streak_days == 30:
        return "🏆 30-day streak!", "Incredible discipline — 30 days of perfect adherence!"
    return f"🔥 {streak_days}-day streak!", f"Keep going — {streak_days} days of on-time doses."
