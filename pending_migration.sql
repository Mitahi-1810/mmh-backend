-- ============================================================
-- PASTE THIS IN: Supabase Dashboard → SQL Editor → New query
-- ============================================================

-- ── FIX broken signup trigger (was failing with "Database error creating new user") ──
-- Drop the legacy trigger that prevents new auth.users inserts
drop trigger if exists on_user_created on auth.users;

-- Recreate with SECURITY DEFINER so it runs with table-owner privileges
create or replace function init_tone_scores()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.user_tone_scores (user_id, tone, score) values
        (new.id, 'friendly', 0),
        (new.id, 'strict', 0),
        (new.id, 'scientific', 0),
        (new.id, 'motivational', 0)
    on conflict do nothing;

    insert into public.user_profiles (id, display_name)
    values (new.id, coalesce(new.raw_user_meta_data->>'display_name', split_part(new.email, '@', 1)))
    on conflict do nothing;

    return new;
end;
$$;

create trigger on_user_created
    after insert on auth.users
    for each row execute procedure init_tone_scores();


-- ── Prescriptions ────────────────────────────────────────────────────────────
create table if not exists prescriptions (
    id                  bigserial primary key,
    user_id             uuid not null references auth.users(id) on delete cascade,
    image_url           text not null,
    raw_gemini_output   jsonb,
    overall_confidence  float check (overall_confidence between 0 and 1),
    status              text default 'pending'
                        check (status in ('pending', 'confirmed', 'rejected')),
    verified_by_user    boolean default false,
    used_for_training   boolean default false,
    created_at          timestamptz default now(),
    updated_at          timestamptz default now()
);

create index if not exists prescriptions_user_idx     on prescriptions(user_id);
create index if not exists prescriptions_status_idx   on prescriptions(status);
create index if not exists prescriptions_training_idx on prescriptions(used_for_training)
    where verified_by_user = true;

-- ── Prescription Medicine Lines ───────────────────────────────────────────────
create table if not exists prescription_medicines (
    id                  bigserial primary key,
    prescription_id     bigint not null references prescriptions(id) on delete cascade,
    user_id             uuid not null references auth.users(id) on delete cascade,
    raw_text            text,
    medicine_id         bigint references medicines(id) on delete set null,
    brand_name          text not null,
    generic_name        text,
    strength            text,
    dosage_form         text,
    dose_instruction    text,
    frequency           text,
    timing              text,
    duration            text,
    confidence          float check (confidence between 0 and 1),
    was_corrected       boolean default false,
    correction_source   text default 'user'
                        check (correction_source in ('user', 'pharmacist', 'doctor')),
    created_at          timestamptz default now()
);

create index if not exists rx_medicines_prescription_idx on prescription_medicines(prescription_id);
create index if not exists rx_medicines_user_idx         on prescription_medicines(user_id);
create index if not exists rx_medicines_medicine_idx     on prescription_medicines(medicine_id);

-- ── Extend user_profiles ──────────────────────────────────────────────────────
alter table user_profiles
    add column if not exists trust_score   int     default 1 check (trust_score between 1 and 10),
    add column if not exists is_pharmacist boolean default false,
    add column if not exists conditions    text[];

-- ── RLS ──────────────────────────────────────────────────────────────────────
alter table prescriptions           enable row level security;
alter table prescription_medicines  enable row level security;

drop policy if exists "own prescriptions"  on prescriptions;
drop policy if exists "own rx medicines"   on prescription_medicines;

create policy "own prescriptions"
    on prescriptions for all using (auth.uid() = user_id);

create policy "own rx medicines"
    on prescription_medicines for all using (auth.uid() = user_id);

-- ── Training view ─────────────────────────────────────────────────────────────
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
