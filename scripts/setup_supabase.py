"""
One-time Supabase setup:
  1. Create storage buckets (prescriptions, expiry-scans)
  2. Verify schema tables exist
  3. Print checklist

Usage:
    python scripts/setup_supabase.py
"""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from supabase import create_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REQUIRED_BUCKETS = [
    {"name": "prescriptions",  "public": True},
    {"name": "expiry-scans",   "public": False},
]

REQUIRED_TABLES = [
    "medicines",
    "prescriptions",
    "prescription_medicines",
    "user_profiles",
    "user_reminders",
    "notifications",
    "user_tone_scores",
    "drug_interactions",
]


def ensure_buckets(sb):
    log.info("── Storage buckets ──────────────────────────────")
    try:
        existing = [b.name for b in sb.storage.list_buckets()]
    except Exception as exc:
        log.error("Could not list buckets: %s", exc)
        existing = []

    for bucket in REQUIRED_BUCKETS:
        name = bucket["name"]
        if name in existing:
            log.info("  ✓ bucket '%s' already exists", name)
        else:
            try:
                sb.storage.create_bucket(name, options={"public": bucket["public"]})
                log.info("  ✓ created bucket '%s' (public=%s)", name, bucket["public"])
            except Exception as exc:
                log.error("  ✗ failed to create bucket '%s': %s", name, exc)


def check_tables(sb):
    log.info("── Database tables ──────────────────────────────")
    for table in REQUIRED_TABLES:
        try:
            res = sb.table(table).select("id").limit(1).execute()
            log.info("  ✓ table '%s' exists", table)
        except Exception as exc:
            err_str = str(exc)
            if "does not exist" in err_str or "relation" in err_str:
                log.error("  ✗ table '%s' MISSING — run supabase_schema.sql", table)
            else:
                log.warning("  ? table '%s' check error: %s", table, exc)


def count_medicines(sb):
    log.info("── Medicine data ─────────────────────────────────")
    try:
        res = sb.table("medicines").select("id", count="exact").execute()
        count = res.count or 0
        if count == 0:
            log.warning("  ⚠ medicines table is EMPTY — run: python scripts/import_medicines.py")
        else:
            log.info("  ✓ %d medicines in DB", count)
    except Exception as exc:
        log.error("  ✗ could not count medicines: %s", exc)


def main():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        log.error("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        sys.exit(1)

    sb = create_client(url, key)
    log.info("Connected to %s", url)

    ensure_buckets(sb)
    check_tables(sb)
    count_medicines(sb)

    log.info("")
    log.info("Setup complete. Next steps if any ✗ above:")
    log.info("  1. Apply schema: paste supabase_schema.sql into Supabase SQL editor")
    log.info("  2. Import CSV:   python scripts/import_medicines.py")
    log.info("  3. Start server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
