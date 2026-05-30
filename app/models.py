from pydantic import BaseModel, Field
from typing import Any, Literal


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: str | None = None


# ── Response envelope ─────────────────────────────────────────────────────────
# The LLM always returns this structure so the frontend knows which component to render.

ResponseFormat = Literal[
    "text",
    "price_table",
    "interaction_cards",
    "prescription_list",
    "reminder_confirm",
    "dosage_card",
    "prescription_scan",
]


class ChatResponseEnvelope(BaseModel):
    message: str
    format: ResponseFormat = "text"
    data: Any = None


# ── Medicine ──────────────────────────────────────────────────────────────────

class MedicineRow(BaseModel):
    id: int | None = None
    brand_name: str
    generic_name: str
    manufacturer: str
    price_per_unit: float
    unit: str
    dosage_form: str | None = None
    strength: str | None = None
    indications: str | None = None
    medex_slug: str | None = None


# ── Drug Interaction ─────────────────────────────────────────────────────────

class InteractionCard(BaseModel):
    drug_a: str
    drug_b: str
    severity: Literal["major", "moderate", "minor"]
    description: str


# ── Reminder ─────────────────────────────────────────────────────────────────

class ReminderRow(BaseModel):
    user_id: str
    medicine_name: str
    remind_at: str  # ISO datetime string
    note: str | None = None


# ── Prescription Extraction (V7) ──────────────────────────────────────────────

FrequencyType = Literal[
    "once_daily", "twice_daily", "three_times_daily",
    "four_times_daily", "as_needed", "weekly", "other"
]

TimingType = Literal[
    "before_meals", "after_meals", "with_meals",
    "at_bedtime", "on_empty_stomach", "other"
]


class ExtractedMedicine(BaseModel):
    """One medicine line extracted from a prescription image."""
    raw_text: str | None = None            # verbatim text Gemini read
    brand_name: str
    generic_name: str | None = None
    strength: str | None = None            # e.g. "500mg"
    dosage_form: str | None = None         # tablet | capsule | syrup | injection
    dose_instruction: str | None = None    # e.g. "1+0+1"
    frequency: str | None = None           # FrequencyType value
    timing: str | None = None              # TimingType value
    duration: str | None = None            # e.g. "5 days"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Populated after semantic DB match (medicines table)
    medicine_id: int | None = None
    matched_generic: str | None = None
    unit_price: float | None = None
    # Populated after DB insert (prescription_medicines table row id)
    pm_id: int | None = None
    # ── Post-OCR fuzzy correction ─────────────────────────────────────────────
    # "confirmed"  → exact DB match (OCR name is a known brand)
    # "suggested"  → fuzzy match found with same generic  (safe to accept)
    # "ambiguous"  → fuzzy match found but different generic (WARN, never auto-correct)
    # "unknown"    → no close DB match at all
    db_status: str | None = None
    db_suggestion: str | None = None          # suggested brand name from DB
    db_suggestion_generic: str | None = None  # generic of suggestion (for safety display)


class PrescriptionScanResponse(BaseModel):
    """Full response from the V7 prescription pipeline."""
    prescription_id: int
    image_url: str
    medicines: list[ExtractedMedicine]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    legibility_score: int = Field(ge=1, le=10)
    review_required: bool                  # True when confidence < 0.70
    message: str


class MedicineCorrectionRequest(BaseModel):
    """User correction payload for a single medicine line."""
    brand_name: str | None = None
    generic_name: str | None = None
    strength: str | None = None
    dosage_form: str | None = None
    dose_instruction: str | None = None
    frequency: str | None = None
    timing: str | None = None
    duration: str | None = None
    correction_source: Literal["user", "pharmacist", "doctor"] = "user"


# ── Reminder / Medication Schedule System ────────────────────────────────────

DoseStatus = Literal["scheduled", "taken", "missed", "skipped", "snoozed"]


class ScheduleCreateRequest(BaseModel):
    """Create one medication schedule (from Rx scan or manually)."""
    prescription_id: int | None = None
    pm_id: int | None = None
    brand_name: str
    generic_name: str | None = None
    strength: str | None = None
    dosage_form: str | None = None
    frequency: str = "once_daily"
    dose_times: list[str] = Field(default_factory=lambda: ["08:00"])  # 24h local "HH:MM"
    timing: str | None = None
    dose_instruction: str | None = None
    start_date: str = ""           # "YYYY-MM-DD"; defaults to today if blank
    end_date: str | None = None
    total_quantity: int | None = None
    refill_alert_days: int = 5
    notify_push: bool = True
    notify_whatsapp: bool = False
    snooze_minutes: int = 15


class BulkScheduleCreateRequest(BaseModel):
    """Create multiple schedules from one prescription scan in a single call."""
    prescription_id: int
    schedules: list[ScheduleCreateRequest]


class MedicationScheduleOut(BaseModel):
    """Full schedule record returned to the client."""
    id: int
    user_id: str
    prescription_id: int | None
    pm_id: int | None
    brand_name: str
    generic_name: str | None
    strength: str | None
    dosage_form: str | None
    frequency: str
    dose_times: list[str]
    timing: str | None
    dose_instruction: str | None
    start_date: str
    end_date: str | None
    total_quantity: int | None
    remaining_quantity: int | None
    refill_alert_days: int
    notify_push: bool
    notify_whatsapp: bool
    snooze_minutes: int
    is_active: bool
    created_at: str


class TodayDose(BaseModel):
    """One dose slot rendered in today's pill tracker."""
    dose_log_id: int
    schedule_id: int
    brand_name: str
    generic_name: str | None
    strength: str | None
    dosage_form: str | None
    dose_instruction: str | None
    timing: str | None
    scheduled_at: str          # ISO-8601 UTC
    taken_at: str | None
    snoozed_until: str | None
    status: DoseStatus
    time_label: str            # "08:00 AM"  (local display)
    is_overdue: bool
    note: str | None = None


class DoseActionRequest(BaseModel):
    note: str | None = None


class SnoozeRequest(BaseModel):
    minutes: int = 15
    note: str | None = None


class AdherenceDayStats(BaseModel):
    date: str
    total: int
    taken: int
    missed: int
    skipped: int
    score: float | None


class AdherenceStats(BaseModel):
    period_days: int
    total: int
    taken: int
    missed: int
    skipped: int
    score: float           # 0.0–1.0
    streak_days: int       # current consecutive days with ≥1 taken dose
    daily: list[AdherenceDayStats]


class PushSubscriptionRequest(BaseModel):
    endpoint: str
    p256dh: str
    auth_key: str
    platform: str = "web"
    user_agent: str | None = None


class WhatsAppSettingsRequest(BaseModel):
    phone_number: str
    opted_in: bool = True
