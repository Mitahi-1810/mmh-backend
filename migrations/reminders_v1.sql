-- ============================================================
-- Medicine Reminder System — v1 schema
-- Run via: Supabase SQL editor or psql
-- ============================================================

-- ── 1. Medication schedules ───────────────────────────────────
-- One row per medicine per active course.
-- Created automatically from prescription scan or manually.
CREATE TABLE IF NOT EXISTS medication_schedules (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    -- Link to prescription (optional — can be created manually too)
    prescription_id BIGINT REFERENCES prescriptions(id) ON DELETE SET NULL,
    pm_id           BIGINT REFERENCES prescription_medicines(id) ON DELETE SET NULL,

    -- Medicine info (denormalised for fast reads)
    brand_name      TEXT NOT NULL,
    generic_name    TEXT,
    strength        TEXT,
    dosage_form     TEXT,

    -- Schedule
    frequency       TEXT NOT NULL,  -- once_daily | twice_daily | three_times_daily |
                                    -- four_times_daily | as_needed | weekly | other
    dose_times      JSONB NOT NULL DEFAULT '["08:00"]',  -- ["08:00","20:00"] (24h, local)
    timing          TEXT,           -- before_meals | after_meals | with_meals |
                                    -- at_bedtime | on_empty_stomach
    dose_instruction TEXT,          -- "1+0+1" display string
    start_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    end_date        DATE,           -- NULL = indefinite (chronic medicine)

    -- Refill tracking
    total_quantity      INTEGER,    -- tablets / capsules dispensed
    remaining_quantity  INTEGER,    -- decremented on each taken dose
    refill_alert_days   INTEGER DEFAULT 5,  -- warn when ≤ N days supply left

    -- Notification preferences
    notify_push     BOOLEAN DEFAULT TRUE,
    notify_whatsapp BOOLEAN DEFAULT FALSE,
    snooze_minutes  INTEGER DEFAULT 15,

    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_med_schedules_user
    ON medication_schedules(user_id) WHERE is_active = TRUE;


-- ── 2. Dose logs ──────────────────────────────────────────────
-- One row per scheduled dose instance (generated on the fly or pre-generated).
CREATE TABLE IF NOT EXISTS dose_logs (
    id              BIGSERIAL PRIMARY KEY,
    schedule_id     BIGINT NOT NULL REFERENCES medication_schedules(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL,

    scheduled_at    TIMESTAMPTZ NOT NULL,  -- exact time this dose was due
    taken_at        TIMESTAMPTZ,           -- when user tapped "Taken"
    snoozed_until   TIMESTAMPTZ,           -- set on snooze
    status          TEXT NOT NULL DEFAULT 'scheduled',
                    -- scheduled | taken | missed | skipped | snoozed

    note            TEXT,                  -- optional user note
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Prevent duplicate dose-log rows when generator runs twice on the same day
ALTER TABLE dose_logs
    ADD CONSTRAINT uq_dose_logs_schedule_time
    UNIQUE (schedule_id, scheduled_at);

CREATE INDEX IF NOT EXISTS idx_dose_logs_schedule
    ON dose_logs(schedule_id, scheduled_at DESC);
CREATE INDEX IF NOT EXISTS idx_dose_logs_user_date
    ON dose_logs(user_id, scheduled_at DESC);
CREATE INDEX IF NOT EXISTS idx_dose_logs_status
    ON dose_logs(status, scheduled_at DESC) WHERE status IN ('scheduled', 'snoozed');


-- ── 3. Push subscriptions (Web Push / FCM) ────────────────────
CREATE TABLE IF NOT EXISTS push_subscriptions (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    endpoint    TEXT NOT NULL UNIQUE,
    p256dh      TEXT NOT NULL,   -- client public key
    auth_key    TEXT NOT NULL,   -- client auth secret
    platform    TEXT,            -- 'web' | 'android' | 'ios'
    user_agent  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_used   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_push_subs_user
    ON push_subscriptions(user_id);


-- ── 4. WhatsApp settings ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS whatsapp_settings (
    user_id         UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    phone_number    TEXT NOT NULL,  -- with country code, e.g. +8801711234567
    verified        BOOLEAN DEFAULT FALSE,
    opted_in        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);


-- ── 5. Adherence summary (materialised daily by scheduler) ────
CREATE TABLE IF NOT EXISTS adherence_daily (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL,
    date        DATE NOT NULL,
    total       INTEGER NOT NULL DEFAULT 0,   -- doses scheduled that day
    taken       INTEGER NOT NULL DEFAULT 0,   -- doses marked taken
    missed      INTEGER NOT NULL DEFAULT 0,
    skipped     INTEGER NOT NULL DEFAULT 0,
    score       NUMERIC(4,3),                 -- taken / total, 0-1
    UNIQUE(user_id, date)
);

CREATE INDEX IF NOT EXISTS idx_adherence_user
    ON adherence_daily(user_id, date DESC);


-- ── RLS policies ──────────────────────────────────────────────
ALTER TABLE medication_schedules  ENABLE ROW LEVEL SECURITY;
ALTER TABLE dose_logs              ENABLE ROW LEVEL SECURITY;
ALTER TABLE push_subscriptions     ENABLE ROW LEVEL SECURITY;
ALTER TABLE whatsapp_settings      ENABLE ROW LEVEL SECURITY;
ALTER TABLE adherence_daily        ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users own their schedules"
    ON medication_schedules FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "users own their dose logs"
    ON dose_logs FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "users own their push subs"
    ON push_subscriptions FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "users own their whatsapp settings"
    ON whatsapp_settings FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "users own their adherence"
    ON adherence_daily FOR ALL USING (auth.uid() = user_id);
