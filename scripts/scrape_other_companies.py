"""
Medex.com.bd — All Other Companies Scraper  v1
===============================================
Scrapes every pharmaceutical company on medex.com.bd EXCEPT the top 20
that were already scraped by scrape_top_companies.py.

Stack:
  • ScrapingAnt API  — residential proxy rotation, anti-bot bypass
  • BeautifulSoup    — HTML parsing
  • httpx            — HTTP client
  • Checkpoint/resume — Ctrl-C safe

Usage:
  python scripts/scrape_other_companies.py --api-key YOUR_KEY
  python scripts/scrape_other_companies.py --api-key YOUR_KEY --resume
  python scripts/scrape_other_companies.py --api-key YOUR_KEY --resume --delay 1.0

New key: b36559c249bd4fb1882ce7fc5bca290b
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent.parent / "scrape_others.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("medex_others")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_URL   = "https://medex.com.bd"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "medicines_other_companies.csv"
CHECKPOINT = OUTPUT_DIR / "scrape_checkpoint_others.json"

# ── ScrapingAnt ───────────────────────────────────────────────────────────────
ANT_URL = "https://api.scrapingant.com/v2/general"

# ── Tuning ────────────────────────────────────────────────────────────────────
BASE_DELAY = 1.0

# ── Companies to SKIP (top 20 — already scraped) ─────────────────────────────
SKIP_KEYWORDS = [
    "square pharmaceuticals",
    "incepta",
    "beximco",
    "opsonin",
    "renata",
    "healthcare pharmaceuticals",
    "aristopharma",
    "aci limited",
    "aci pharmaceuticals",
    "eskayef",
    "sk+f",
    "skf",
    "acme laboratories",
    "drug international",
    "radiant pharmaceuticals",
    "popular pharmaceuticals",
    "general pharmaceuticals",
    "unimed",
    "unihealth",
    "beacon pharmaceuticals",
    "ibn sina",
    "ibnsina",
    "orion pharma",
    "novo nordisk",
    "ziska",
]

CSV_FIELDS = [
    "company_name", "brand_name", "generic_name", "dosage_form", "strength",
    "unit_price", "strip_price", "pack_info", "therapeutic_class", "manufacturer",
    "indications", "contraindications", "side_effects", "drug_interactions",
    "pharmacology", "dosage_administration", "pregnancy_lactation",
    "precautions_warnings", "overdose_effects", "storage_conditions",
    "composition", "brand_url", "scraped_at",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPINGANT FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch(url: str, api_key: str, retries: int = 3) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(
                ANT_URL,
                params={"url": url, "x-api-key": api_key, "browser": "false"},
                timeout=30,
            )
            if r.status_code == 200 and r.text:
                return r.text
            elif r.status_code == 422:
                log.error("  ScrapingAnt 422 — invalid URL: %s", url)
                return None
            elif r.status_code == 403:
                log.error("  ScrapingAnt 403 — check API key")
                return None
            elif r.status_code == 429:
                log.warning("  ScrapingAnt 429 — rate limit, backing off 30s")
                time.sleep(30)
            else:
                log.warning("  ScrapingAnt HTTP %d (attempt %d)", r.status_code, attempt)
        except Exception as e:
            log.warning("  Request error (attempt %d): %s", attempt, e)

        if attempt < retries:
            time.sleep(random.uniform(3, 7))

    log.warning("  All %d attempts failed for %s", retries, url)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# HTML PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def _section(soup: BeautifulSoup, div_id: str) -> str:
    anchor = soup.find("div", id=div_id)
    if not anchor:
        return ""
    body = anchor.find_next_sibling("div", class_="ac-body")
    if not body:
        return ""
    return " ".join(body.get_text(separator=" ", strip=True).split())


def is_blocked(html: str, soup: BeautifulSoup) -> bool:
    if not soup.find(id="ms-block"):
        return True
    title_tag = soup.find("title")
    title_text = (title_tag.string or "").lower() if title_tag else ""
    if "captcha" in title_text or "captcha" in html[:2000].lower():
        return True
    return False


def parse_brand(html: str, brand_url: str, company_name: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    if is_blocked(html, soup):
        log.warning("  Blocked page detected for %s", brand_url)
        return None

    h1 = soup.select_one("h1.page-heading-1-l")
    dosage_form = brand_name = ""
    if h1:
        sm = h1.find("small")
        if sm:
            dosage_form = sm.get_text(strip=True)
            sm.extract()
        brand_name = h1.get_text(strip=True)

    if not brand_name:
        return None

    strength = ""
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        parts = [p.strip() for p in title_tag.string.split("|")]
        if len(parts) >= 2:
            strength = parts[1]
            if not dosage_form and len(parts) >= 3:
                dosage_form = parts[2]

    gen_el = soup.select_one("a[href*='/generics/']")
    mfr_el = soup.select_one("a[href*='/companies/']")

    unit_price = strip_price = pack_info = ""
    pkg = soup.select_one(".packages-wrapper") or soup.find("div", class_="package-container")
    if pkg:
        raw = pkg.get_text(" ", strip=True)
        m = re.search(r"Unit\s*Price\s*[:\s]*৳?\s*([\d,.]+)", raw, re.I)
        if m: unit_price = m.group(1)
        m2 = re.search(r"Strip\s*Price\s*[:\s]*৳?\s*([\d,.]+)", raw, re.I)
        if m2: strip_price = m2.group(1)
        m3 = re.search(r"\(([^)]+৳[^)]+)\)", raw)
        if m3: pack_info = m3.group(1).strip()

    return {
        "company_name":          company_name,
        "brand_name":            brand_name,
        "generic_name":          gen_el.get_text(strip=True) if gen_el else "",
        "dosage_form":           dosage_form,
        "strength":              strength,
        "unit_price":            unit_price,
        "strip_price":           strip_price,
        "pack_info":             pack_info,
        "therapeutic_class":     _section(soup, "drug_classes"),
        "manufacturer":          mfr_el.get_text(strip=True) if mfr_el else "",
        "indications":           _section(soup, "indications"),
        "contraindications":     _section(soup, "contraindications"),
        "side_effects":          _section(soup, "side_effects"),
        "drug_interactions":     _section(soup, "interaction"),
        "pharmacology":          _section(soup, "mode_of_action"),
        "dosage_administration": _section(soup, "dosage"),
        "pregnancy_lactation":   _section(soup, "pregnancy_cat"),
        "precautions_warnings":  _section(soup, "precautions"),
        "overdose_effects":      _section(soup, "overdose_effects"),
        "storage_conditions":    _section(soup, "storage_conditions"),
        "composition":           _section(soup, "composition"),
        "brand_url":             brand_url,
        "scraped_at":            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY DISCOVERY — ALL except top 20
# ═══════════════════════════════════════════════════════════════════════════════

def _is_top20(name: str) -> bool:
    """Return True if this company is one of the top 20 already scraped."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in SKIP_KEYWORDS)


def discover_other_companies(api_key: str) -> list[dict]:
    """
    Paginate through medex.com.bd/companies and collect every company
    that is NOT in the top 20.  Returns list of {name, brands_url}.
    """
    found: dict[str, dict] = {}
    page = 1
    skipped = 0

    while True:
        log.info("Fetching companies page %d …", page)
        html = fetch(f"{BASE_URL}/companies?page={page}", api_key)
        if not html:
            log.error("Failed to fetch companies page %d", page)
            break

        soup  = BeautifulSoup(html, "lxml")
        links = soup.select("a[href*='/companies/']")

        if not links:
            log.info("No more company links on page %d — done discovering", page)
            break

        new_this_page = 0
        for a in links:
            href = a.get("href", "")
            if not re.search(r"/companies/\d+/", href):
                continue

            raw_name = a.get_text(strip=True)
            if not raw_name or raw_name in found:
                continue

            if _is_top20(raw_name):
                skipped += 1
                log.debug("  SKIP (top20): %s", raw_name)
                continue

            # Build brands URL
            base = href.split("/brands")[0].rstrip("/")
            full = base if base.startswith("http") else BASE_URL + base
            found[raw_name] = {
                "name":       raw_name,
                "brands_url": full + "/brands",
            }
            new_this_page += 1
            log.info("  + %s", raw_name)

        log.info("  Page %d: +%d new, %d total, %d top20 skipped",
                 page, new_this_page, len(found), skipped)

        # Check for next page
        if not soup.select_one("a[rel='next']"):
            log.info("No 'next' link — finished paginating companies")
            break

        page += 1
        time.sleep(random.uniform(0.8, 2.0))

    log.info("Discovered %d non-top20 companies (%d top20 skipped)", len(found), skipped)
    return list(found.values())


def get_brand_urls(company: dict, api_key: str) -> list[str]:
    urls: list[str] = []
    seen: set[str]  = set()
    page = 1
    while True:
        html = fetch(f"{company['brands_url']}?page={page}", api_key)
        if not html:
            break
        soup  = BeautifulSoup(html, "lxml")
        added = 0
        for a in soup.select("a[href*='/brands/']"):
            href = a.get("href", "")
            if not re.search(r"/brands/\d+/[^/?#]+$", href):
                continue
            full = href if href.startswith("http") else BASE_URL + href
            if full not in seen:
                seen.add(full)
                urls.append(full)
                added += 1
        log.info("    Page %d → +%d (total %d)", page, added, len(urls))
        if added == 0 or not soup.select_one("a[rel='next']"):
            break
        page += 1
        time.sleep(random.uniform(0.5, 1.5))
    return urls


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def load_cp() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"scraped_urls": [], "company_urls": {}}


def save_cp(cp: dict):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(api_key: str, output_path: Path, resume: bool, base_delay: float):
    cp   = load_cp() if resume else {"scraped_urls": [], "company_urls": {}}
    done = set(cp.get("scraped_urls", []))

    if resume and output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u = row.get("brand_url", "").strip()
                if u and u not in done:
                    done.add(u)
                    cp["scraped_urls"].append(u)
        save_cp(cp)
        log.info("Checkpoint synced — %d URLs already done", len(done))

    log.info("=" * 70)
    log.info("Medex Others Scraper — ScrapingAnt key: %s…", api_key[:8])
    log.info("=" * 70)

    # Use cached company list from checkpoint, or discover fresh
    company_urls_cache = cp.setdefault("company_urls", {})
    companies_cache_key = "__companies__"

    if resume and companies_cache_key in company_urls_cache:
        companies = company_urls_cache[companies_cache_key]
        log.info("Using cached company list: %d companies", len(companies))
    else:
        companies = discover_other_companies(api_key)
        company_urls_cache[companies_cache_key] = companies
        save_cp(cp)
        log.info("Discovered and cached %d companies", len(companies))

    log.info("Target: %d companies to scrape", len(companies))

    append   = output_path.exists() and resume
    out_file = open(output_path, "a" if append else "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(out_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if not append:
        writer.writeheader()

    total = 0
    try:
        for ci, company in enumerate(companies, 1):
            co_key = company["name"]
            log.info("")
            log.info("━" * 70)
            log.info("[%d/%d] %s", ci, len(companies), co_key)

            # Use cached brand URL list if available
            brand_urls_key = f"brands_{co_key}"
            if brand_urls_key in company_urls_cache:
                brand_urls = company_urls_cache[brand_urls_key]
                log.info("  Brands: %d (from cache)", len(brand_urls))
            else:
                brand_urls = get_brand_urls(company, api_key)
                company_urls_cache[brand_urls_key] = brand_urls
                save_cp(cp)
                log.info("  Brands found: %d", len(brand_urls))

            todo = [u for u in brand_urls if u not in done]
            if not todo:
                log.info("  → Already complete, skipping")
                continue
            log.info("  → %d to scrape, %d already done", len(todo), len(brand_urls) - len(todo))

            co_count = 0
            for bi, burl in enumerate(todo, 1):
                log.info("  [%d/%d] %s", bi, len(todo), burl)
                html = fetch(burl, api_key)

                if html is None:
                    log.warning("  Fetch failed — skipping")
                else:
                    row = parse_brand(html, burl, co_key)
                    if row:
                        writer.writerow(row)
                        out_file.flush()
                        co_count += 1
                        total    += 1
                        done.add(burl)
                        cp["scraped_urls"].append(burl)

                        if total % 100 == 0:
                            save_cp(cp)
                            log.info("  ✓ checkpoint: %d total rows", total)
                    else:
                        log.warning("  No data parsed")

                time.sleep(base_delay + random.uniform(0, 0.5))

            log.info("  → %d written for %s", co_count, co_key)
            save_cp(cp)

            if ci < len(companies):
                time.sleep(random.uniform(1.5, 3.5))

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        out_file.close()
        save_cp(cp)
        log.info("=" * 70)
        log.info("Done — %d total rows written | %s", total, output_path)


def main():
    ap = argparse.ArgumentParser(description="Medex other companies scraper")
    ap.add_argument("--api-key", required=True, help="ScrapingAnt API key")
    ap.add_argument("--output", default=str(OUTPUT_CSV))
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument("--delay", type=float, default=BASE_DELAY, help="Base delay between requests (s)")
    args = ap.parse_args()
    run(args.api_key, Path(args.output), args.resume, args.delay)


if __name__ == "__main__":
    main()
