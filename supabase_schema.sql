-- ============================================================
-- Sanjibani — Supabase Schema
-- Run this in the Supabase SQL editor.
-- ============================================================

-- Enable pgvector for semantic drug name search
create extension if not exists vector;

-- ── Medicines ────────────────────────────────────────────────────────────────
create table if not exists medicines (
    id               bigserial primary key,
    brand_name       text not null,
    generic_name     text not null,
    manufacturer     text,
    price_per_unit   numeric(10, 2),
    unit             text default 'tablet',
    dosage_form      text,
    strength         text,
    indications      text,
    medex_slug       text unique,
    last_scraped_at  timestamptz,
    name_embedding   vector(768),  -- Gemini text-embedding-004 produces 768-dim
    created_at       timestamptz default now()
);

-- Semantic search index
create index if not exists medicines_embedding_idx
    on medicines using ivfflat (name_embedding vector_cosine_ops)
    with (lists = 100);

-- Fast lookup by generic name
create index if not exists medicines_generic_idx on medicines (generic_name);
create index if not exists medicines_brand_idx   on medicines (brand_name);

-- ── pgvector match function (called from FastAPI) ────────────────────────────
create or replace function match_medicines(query_text text, match_count int default 10)
returns table (
    id             bigint,
    brand_name     text,
    generic_name   text,
    manufacturer   text,
    price_per_unit numeric,
    unit           text,
    dosage_form    text,
    strength       text,
    indications    text,
    medex_slug     text,
    similarity     float
)
language plpgsql as $$
declare
    query_embedding vector(768);
begin
    -- Embed via pg_net + Supabase edge function (or pass pre-computed embedding)
    -- For simplicity, fall back to ilike if embedding is null
    return query
        select
            m.id, m.brand_name, m.generic_name, m.manufacturer,
            m.price_per_unit, m.unit, m.dosage_form, m.strength,
            m.indications, m.medex_slug,
            1 - (m.name_embedding <=> query_embedding) as similarity
        from medicines m
        where m.name_embedding is not null
        order by m.name_embedding <=> query_embedding
        limit match_count;
end;
$$;

-- ── Drug Interactions ────────────────────────────────────────────────────────
create table if not exists drug_interactions (
    id          bigserial primary key,
    drug_a      text not null,
    drug_b      text not null,
    severity    text check (severity in ('major', 'moderate', 'minor')) not null,
    description text not null,
    source      text default 'DrugBank',
    created_at  timestamptz default now(),
    unique (drug_a, drug_b)
);

create index if not exists interactions_drug_a_idx on drug_interactions (lower(drug_a));
create index if not exists interactions_drug_b_idx on drug_interactions (lower(drug_b));

-- ── Users (extends Supabase auth.users) ──────────────────────────────────────
create table if not exists user_profiles (
    id              uuid primary key references auth.users (id) on delete cascade,
    display_name    text,
    preferred_lang  text default 'en',
    created_at      timestamptz default now()
);

-- ── User Reminders ────────────────────────────────────────────────────────────
create table if not exists user_reminders (
    id              bigserial primary key,
    user_id         uuid not null references auth.users (id) on delete cascade,
    medicine_name   text not null,
    remind_at       timestamptz not null,
    note            text,
    type            text default 'schedule' check (type in ('schedule', 'expiry')),
    last_sent_at    timestamptz,
    created_at      timestamptz default now()
);

create index if not exists reminders_user_idx    on user_reminders (user_id);
create index if not exists reminders_remind_idx  on user_reminders (remind_at);

-- ── Notifications (pushed by scheduler, read by frontend via Realtime) ────────
create table if not exists notifications (
    id              bigserial primary key,
    user_id         uuid not null references auth.users (id) on delete cascade,
    message         text not null,
    tone_used       text,
    reminder_id     bigint references user_reminders (id),
    sent_at         timestamptz default now(),
    acknowledged    boolean default false
);

-- Enable Realtime on notifications so frontend gets live push
alter publication supabase_realtime add table notifications;

-- ── RL Tone Scores ────────────────────────────────────────────────────────────
create table if not exists user_tone_scores (
    user_id     uuid not null references auth.users (id) on delete cascade,
    tone        text not null,
    score       int default 0,
    primary key (user_id, tone)
);

-- Seed default tone scores for new users
create or replace function init_tone_scores()
returns trigger language plpgsql as $$
begin
    insert into user_tone_scores (user_id, tone, score) values
        (new.id, 'friendly', 0),
        (new.id, 'strict', 0),
        (new.id, 'scientific', 0),
        (new.id, 'motivational', 0)
    on conflict do nothing;
    return new;
end;
$$;

create or replace trigger on_user_created
    after insert on auth.users
    for each row execute procedure init_tone_scores();

-- ── Row Level Security ────────────────────────────────────────────────────────
alter table user_profiles    enable row level security;
alter table user_reminders   enable row level security;
alter table notifications    enable row level security;
alter table user_tone_scores enable row level security;

-- Users can only access their own data
create policy "own profile"    on user_profiles    for all using (auth.uid() = id);
create policy "own reminders"  on user_reminders   for all using (auth.uid() = user_id);
create policy "own notifs"     on notifications    for all using (auth.uid() = user_id);
create policy "own tones"      on user_tone_scores for all using (auth.uid() = user_id);

-- Medicines and interactions are public (read-only)
alter table medicines          enable row level security;
alter table drug_interactions  enable row level security;
create policy "public medicines"     on medicines         for select using (true);
create policy "public interactions"  on drug_interactions for select using (true);


-- ============================================================
-- Prescription Extraction Tables (V7 Architecture)
-- Run this block separately in Supabase SQL editor
-- ============================================================

-- ── Prescriptions ────────────────────────────────────────────────────────────
-- Stores each uploaded prescription image + extraction result
create table if not exists prescriptions (
    id                  bigserial primary key,
    user_id             uuid not null references auth.users(id) on delete cascade,
    image_url           text not null,
    raw_gemini_output   jsonb,                          -- full Gemini response
    overall_confidence  float check (overall_confidence between 0 and 1),
    status              text default 'pending'
                        check (status in ('pending', 'confirmed', 'rejected')),
    verified_by_user    boolean default false,
    used_for_training   boolean default false,          -- flagged after fine-tune export
    created_at          timestamptz default now(),
    updated_at          timestamptz default now()
);

create index if not exists prescriptions_user_idx    on prescriptions(user_id);
create index if not exists prescriptions_status_idx  on prescriptions(status);
create index if not exists prescriptions_training_idx on prescriptions(used_for_training)
    where verified_by_user = true;

-- ── Prescription Medicine Lines ───────────────────────────────────────────────
-- Each medicine extracted from a prescription (user-correctable)
create table if not exists prescription_medicines (
    id                  bigserial primary key,
    prescription_id     bigint not null references prescriptions(id) on delete cascade,
    user_id             uuid not null references auth.users(id) on delete cascade,

    -- Raw extraction from Gemini
    raw_text            text,                           -- exactly what Gemini read

    -- Matched to our medicines DB
    medicine_id         bigint references medicines(id) on delete set null,
    brand_name          text not null,
    generic_name        text,
    strength            text,
    dosage_form         text,

    -- Dosage instruction (parsed)
    dose_instruction    text,                           -- "1+0+1", "1 tablet"
    frequency           text,                           -- "twice_daily", "once_daily"
    timing              text,                           -- "before_meals", "at_bedtime"
    duration            text,                           -- "5 days", "1 month"

    -- Confidence + correction tracking
    confidence          float check (confidence between 0 and 1),
    was_corrected       boolean default false,          -- user edited this line
    correction_source   text default 'user'
                        check (correction_source in ('user', 'pharmacist', 'doctor')),

    created_at          timestamptz default now()
);

create index if not exists rx_medicines_prescription_idx on prescription_medicines(prescription_id);
create index if not exists rx_medicines_user_idx         on prescription_medicines(user_id);
create index if not exists rx_medicines_medicine_idx     on prescription_medicines(medicine_id);

-- ── User Trust Scores (for weighted training) ─────────────────────────────────
alter table user_profiles
    add column if not exists trust_score int default 1
        check (trust_score between 1 and 10),
    add column if not exists is_pharmacist boolean default false,
    add column if not exists conditions text[];         -- ["diabetes", "hypertension"]

-- ── RLS for new tables ───────────────────────────────────────────────────────
alter table prescriptions           enable row level security;
alter table prescription_medicines  enable row level security;

create policy "own prescriptions"
    on prescriptions for all using (auth.uid() = user_id);

create policy "own rx medicines"
    on prescription_medicines for all using (auth.uid() = user_id);

-- ── Helper: get training-ready samples ───────────────────────────────────────
create or replace view training_samples as
    select
        p.id            as prescription_id,
        p.image_url,
        p.user_id,
        up.trust_score,
        up.is_pharmacist,
        json_agg(json_build_object(
            'brand_name',       pm.brand_name,
            'generic_name',     pm.generic_name,
            'strength',         pm.strength,
            'dose_instruction', pm.dose_instruction,
            'frequency',        pm.frequency,
            'timing',           pm.timing,
            'duration',         pm.duration,
            'was_corrected',    pm.was_corrected
        ) order by pm.id) as medicines,
        p.created_at
    from prescriptions p
    join prescription_medicines pm on pm.prescription_id = p.id
    join user_profiles up on up.id = p.user_id
    where p.verified_by_user = true
    group by p.id, p.image_url, p.user_id, up.trust_score, up.is_pharmacist, p.created_at;
