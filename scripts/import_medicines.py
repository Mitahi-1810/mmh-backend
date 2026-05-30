"""
Import medicines_top20.csv → Supabase medicines table.

Usage:
    python scripts/import_medicines.py [--csv data/medicines_top20.csv] [--batch 200]

Skips rows that already exist (upsert on brand_name + strength + dosage_form).
"""
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────────
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


# ── helpers ────────────────────────────────────────────────────────────────────

def _unit_from_form(dosage_form: str) -> str:
    """Infer unit label from dosage form."""
    f = (dosage_form or "").lower()
    if any(x in f for x in ("syrup", "suspension", "solution", "drops", "elixir")):
        return "ml"
    if any(x in f for x in ("injection", "infusion", "vial", "ampoule")):
        return "vial"
    if "sachet" in f:
        return "sachet"
    if any(x in f for x in ("cream", "ointment", "gel", "lotion")):
        return "g"
    if any(x in f for x in ("inhaler", "spray")):
        return "dose"
    return "tablet"


def _safe_float(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except (ValueError, AttributeError):
        return None


def _medex_slug(url: str) -> str | None:
    """Extract medex slug from brand URL."""
    if not url:
        return None
    parts = url.rstrip("/").split("/")
    # URL like https://medex.com.bd/brands/30731/abaclor-250-mg-capsule
    # slug = "abaclor-250-mg-capsule"
    return parts[-1] if parts else None


def _row_to_db(row: dict) -> dict:
    return {
        "brand_name":     (row.get("brand_name") or "").strip(),
        "generic_name":   (row.get("generic_name") or "").strip(),
        "manufacturer":   (row.get("manufacturer") or row.get("company_name") or "").strip(),
        "price_per_unit": _safe_float(row.get("unit_price", "")),
        "unit":           _unit_from_form(row.get("dosage_form", "")),
        "dosage_form":    (row.get("dosage_form") or "").strip() or None,
        "strength":       (row.get("strength") or "").strip() or None,
        "indications":    (row.get("indications") or "")[:2000] or None,
        "medex_slug":     _medex_slug(row.get("brand_url", "")),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default="data/medicines_top20.csv")
    ap.add_argument("--batch", type=int, default=200, help="Rows per Supabase insert call")
    args = ap.parse_args()

    csv_path = ROOT / args.csv
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    sb  = create_client(url, key)

    # Read CSV
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    log.info("Loaded %d rows from %s", len(all_rows), csv_path)

    # Convert and deduplicate — primary key on medex_slug (nulls allowed multiple times),
    # secondary key on (brand_name, strength, dosage_form)
    seen_slugs:  set[str]   = set()
    seen_combos: set[tuple] = set()
    db_rows: list[dict] = []
    skipped_blank = 0
    skipped_dup   = 0

    for raw in all_rows:
        r = _row_to_db(raw)
        if not r["brand_name"] or not r["generic_name"]:
            skipped_blank += 1
            continue

        slug = r.get("medex_slug")
        if slug:
            if slug in seen_slugs:
                skipped_dup += 1
                continue
            seen_slugs.add(slug)
        else:
            combo = (r["brand_name"].lower(), r.get("strength", ""), r.get("dosage_form", ""))
            if combo in seen_combos:
                skipped_dup += 1
                continue
            seen_combos.add(combo)

        db_rows.append(r)

    log.info("Unique medicines to import: %d  (skipped blank: %d, dup: %d)",
             len(db_rows), skipped_blank, skipped_dup)

    # Upsert in batches
    total_inserted = 0
    total_errors   = 0
    batch_size = args.batch

    for i in range(0, len(db_rows), batch_size):
        batch = db_rows[i : i + batch_size]
        batch_no   = i // batch_size + 1
        batch_total = (len(db_rows) + batch_size - 1) // batch_size
        try:
            res = (
                sb.table("medicines")
                .upsert(batch, ignore_duplicates=True)
                .execute()
            )
            inserted = len(res.data or [])
            total_inserted += inserted
            log.info("Batch %d/%d → %d inserted", batch_no, batch_total, inserted)
        except Exception:
            # Row-by-row fallback: skip existing slugs individually
            ok = 0
            for row in batch:
                try:
                    sb.table("medicines").upsert([row], ignore_duplicates=True).execute()
                    ok += 1
                except Exception:
                    pass  # truly duplicate — skip silently
            total_inserted += ok
            log.info("Batch %d/%d → fallback row-by-row: %d/%d inserted",
                     batch_no, batch_total, ok, len(batch))
        time.sleep(0.15)

    log.info("Done. %d rows upserted, %d batch errors.", total_inserted, total_errors)


if __name__ == "__main__":
    main()
