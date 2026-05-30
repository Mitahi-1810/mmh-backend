"""
Medex.com.bd full scraper + nightly diff updater.

Usage:
  python scripts/seed_medex.py              # Full scrape (first run)
  python scripts/seed_medex.py --diff-only  # Only re-scrape pages updated since last run

Strategy:
  1. Fetch the A-Z brand listing pages to collect all /brands/<slug> URLs.
  2. For each URL, extract: brand_name, generic_name, manufacturer, price, strength, dosage_form.
  3. Upsert into Supabase medicines table (conflict on medex_slug).
  4. Delta validation: if new price deviates >50% from stored price, log to delta_warnings.csv
     and skip the DB update until manual review.
"""

import argparse
import asyncio
import csv
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
DELAY = float(os.getenv("MEDEX_SCRAPE_DELAY_SECONDS", "1.0"))
MAX_RETRIES = int(os.getenv("MEDEX_MAX_RETRIES", "3"))

BASE_URL = "https://medex.com.bd"
LISTING_URL = f"{BASE_URL}/brands"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("seed_medex")

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
DELTA_LOG = Path(__file__).parent.parent / "delta_warnings.csv"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, retries: int = MAX_RETRIES) -> httpx.Response | None:
    for attempt in range(retries):
        try:
            r = httpx.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429):
                log.warning("Blocked on %s (status %d). Backing off.", url, r.status_code)
                time.sleep(DELAY * (2 ** attempt) * 5)
        except httpx.RequestError as e:
            log.warning("Request error on %s: %s", url, e)
            time.sleep(DELAY * (attempt + 1))
    return None


# ── Listing page parser ───────────────────────────────────────────────────────

def _get_all_brand_slugs() -> list[str]:
    """Collect all /brands/<slug> paths from the A-Z listing pages."""
    slugs: list[str] = []
    page = 1

    while True:
        url = f"{LISTING_URL}?page={page}"
        log.info("Fetching listing page %d …", page)
        r = _get(url)
        if not r:
            log.error("Failed to fetch listing page %d. Stopping.", page)
            break

        soup = BeautifulSoup(r.text, "lxml")

        # Medex brand links follow the pattern /brands/<slug>
        links = soup.select("a[href^='/brands/']")
        new_slugs = [a["href"].split("/brands/")[1].strip("/") for a in links if a.get("href")]
        new_slugs = [s for s in new_slugs if s and "/" not in s]

        if not new_slugs:
            log.info("No more slugs on page %d. Done listing.", page)
            break

        slugs.extend(new_slugs)
        log.info("  Found %d slugs (total so far: %d)", len(new_slugs), len(slugs))

        # Check for next page
        next_btn = soup.select_one("a[rel='next']")
        if not next_btn:
            break
        page += 1
        time.sleep(DELAY)

    return list(dict.fromkeys(slugs))  # deduplicate, preserve order


# ── Brand page parser ─────────────────────────────────────────────────────────

def _parse_price(text: str) -> float | None:
    """Extract numeric price from strings like '৳8.00' or 'Unit Price: ৳8.00'."""
    match = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _parse_brand_page(slug: str) -> dict | None:
    url = f"{BASE_URL}/brands/{slug}"
    r = _get(url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "lxml")

    # --- Brand name ---
    brand_el = soup.select_one("h1.brand-title") or soup.select_one("h1")
    brand_name = brand_el.get_text(strip=True) if brand_el else slug.replace("-", " ").title()

    # --- Generic name ---
    generic_el = (
        soup.select_one(".generic-name a")
        or soup.select_one("a[href*='/generics/']")
    )
    generic_name = generic_el.get_text(strip=True) if generic_el else ""

    # --- Manufacturer ---
    mfr_el = (
        soup.select_one(".company-name a")
        or soup.select_one("a[href*='/companies/']")
    )
    manufacturer = mfr_el.get_text(strip=True) if mfr_el else ""

    # --- Price ---
    price_el = soup.find(string=re.compile(r"Unit Price", re.I))
    price_per_unit: float | None = None
    unit = "tablet"
    if price_el:
        price_per_unit = _parse_price(price_el)
        # Detect unit type from surrounding text
        ctx = price_el.lower() if isinstance(price_el, str) else ""
        for u in ("ml", "capsule", "syrup", "injection", "suppository", "cream", "ointment"):
            if u in ctx:
                unit = u
                break

    # --- Strength & dosage form ---
    strength_el = soup.select_one(".strength") or soup.find(string=re.compile(r"\d+\s*mg", re.I))
    strength = strength_el.get_text(strip=True) if hasattr(strength_el, "get_text") else (
        str(strength_el).strip() if strength_el else ""
    )

    form_el = soup.select_one(".dosage-form") or soup.find(string=re.compile(
        r"\b(tablet|capsule|syrup|injection|cream|ointment|drop|suppository)\b", re.I
    ))
    dosage_form = form_el.get_text(strip=True) if hasattr(form_el, "get_text") else (
        str(form_el).strip() if form_el else ""
    )

    # --- Indications (short) ---
    ind_el = soup.select_one(".indications p") or soup.select_one(".indications")
    indications = ind_el.get_text(strip=True)[:500] if ind_el else ""

    if not brand_name or not generic_name:
        return None

    return {
        "brand_name": brand_name,
        "generic_name": generic_name,
        "manufacturer": manufacturer,
        "price_per_unit": price_per_unit,
        "unit": unit,
        "strength": strength or None,
        "dosage_form": dosage_form or None,
        "indications": indications or None,
        "medex_slug": slug,
        "last_scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Delta validation ──────────────────────────────────────────────────────────

def _check_delta(slug: str, new_price: float | None) -> bool:
    """Return True if price is safe to write, False if it's a suspicious spike."""
    if new_price is None:
        return True
    result = sb.table("medicines").select("price_per_unit").eq("medex_slug", slug).execute()
    rows = result.data or []
    if not rows or rows[0]["price_per_unit"] is None:
        return True
    old_price = float(rows[0]["price_per_unit"])
    if old_price == 0:
        return True
    ratio = new_price / old_price
    if ratio > 1.5 or ratio < 0.5:
        log.warning("DELTA ALERT: %s old=%.2f new=%.2f (ratio=%.2f)", slug, old_price, new_price, ratio)
        with open(DELTA_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), slug, old_price, new_price, ratio])
        return False
    return True


# ── Upsert ────────────────────────────────────────────────────────────────────

def _upsert(row: dict):
    if not _check_delta(row["medex_slug"], row.get("price_per_unit")):
        return  # Blocked by delta validation
    sb.table("medicines").upsert(row, on_conflict="medex_slug").execute()


# ── Diff mode helpers ─────────────────────────────────────────────────────────

def _get_stale_slugs(hours: int = 25) -> list[str]:
    """Return slugs not updated in the last `hours` hours."""
    cutoff = datetime.now(timezone.utc).replace(microsecond=0)
    from datetime import timedelta
    cutoff -= timedelta(hours=hours)
    result = (
        sb.table("medicines")
        .select("medex_slug")
        .or_(f"last_scraped_at.lte.{cutoff.isoformat()},last_scraped_at.is.null")
        .execute()
    )
    return [r["medex_slug"] for r in (result.data or []) if r.get("medex_slug")]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(diff_only: bool = False):
    if diff_only:
        log.info("Diff mode: fetching stale slugs …")
        slugs = _get_stale_slugs()
        log.info("Found %d stale records to refresh.", len(slugs))
    else:
        log.info("Full scrape: collecting all brand slugs from medex.com.bd …")
        slugs = _get_all_brand_slugs()
        log.info("Total slugs found: %d", len(slugs))

    success = 0
    for i, slug in enumerate(slugs, 1):
        log.info("[%d/%d] Scraping: %s", i, len(slugs), slug)
        row = _parse_brand_page(slug)
        if row:
            _upsert(row)
            success += 1
        time.sleep(DELAY)

    log.info("Done. %d/%d records successfully upserted.", success, len(slugs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medex.com.bd scraper for Sanjibani")
    parser.add_argument("--diff-only", action="store_true", help="Only re-scrape stale pages")
    args = parser.parse_args()
    run(diff_only=args.diff_only)
