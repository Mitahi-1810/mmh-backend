"""
Medex.com.bd — Top 20 BD Pharmaceutical Companies Scraper  v5
==============================================================
Stack:
  • ScrapingAnt API  — residential proxy rotation, anti-bot bypass
  • BeautifulSoup    — HTML parsing
  • httpx            — HTTP client
  • Checkpoint/resume — Ctrl-C safe, --resume to continue

Usage:
  python scripts/scrape_top_companies.py --api-key YOUR_KEY
  python scripts/scrape_top_companies.py --api-key YOUR_KEY --resume
  python scripts/scrape_top_companies.py --api-key YOUR_KEY --resume --delay 1.0

ScrapingAnt free tier: 10,000 credits/month (1 credit per request, no-browser mode)
Sign up: https://scrapingant.com
"""

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
LOG_FILE = Path(__file__).parent.parent / "scrape_medex_v5.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("medex_v5")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_URL   = "https://medex.com.bd"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "medicines_top20.csv"
CHECKPOINT = OUTPUT_DIR / "scrape_checkpoint.json"

# ── ScrapingAnt ───────────────────────────────────────────────────────────────
ANT_URL = "https://api.scrapingant.com/v2/general"

# ── Tuning ────────────────────────────────────────────────────────────────────
BASE_DELAY = 1.0   # seconds between requests (ScrapingAnt handles proxy rotation)

# ── Target companies ──────────────────────────────────────────────────────────
TOP_20 = [
    {"name": "Square Pharmaceuticals Ltd.",              "kw": ["square pharmaceuticals"]},
    {"name": "Incepta Pharmaceuticals Ltd.",             "kw": ["incepta"]},
    {"name": "Beximco Pharmaceuticals Ltd.",             "kw": ["beximco"]},
    {"name": "Opsonin Pharma Ltd.",                      "kw": ["opsonin pharma ltd", "opsonin pharma"]},
    {"name": "Renata Limited",                           "kw": ["renata"]},
    {"name": "Healthcare Pharmaceuticals Ltd.",          "kw": ["healthcare pharmaceuticals"]},
    {"name": "Aristopharma Limited",                     "kw": ["aristopharma"]},
    {"name": "ACI Pharmaceuticals",                      "kw": ["aci limited", "aci pharmaceuticals"]},
    {"name": "Eskayef Pharmaceuticals Ltd. (SK+F)",      "kw": ["eskayef", "sk+f", "skf"]},
    {"name": "The ACME Laboratories Ltd.",               "kw": ["acme"]},
    {"name": "Drug International Limited",               "kw": ["drug international"]},
    {"name": "Radiant Pharmaceuticals Limited",          "kw": ["radiant pharmaceuticals ltd", "radiant pharmaceuticals"]},
    {"name": "Popular Pharmaceuticals Ltd.",             "kw": ["popular pharmaceuticals"]},
    {"name": "General Pharmaceuticals Limited",         "kw": ["general pharmaceuticals"]},
    {"name": "UniMed UniHealth Pharmaceuticals Limited", "kw": ["unimed", "unihealth"]},
    {"name": "Beacon Pharmaceuticals Limited",           "kw": ["beacon"]},
    {"name": "The IBN SINA Pharmaceutical Industry Ltd.","kw": ["ibn sina", "ibnsina"]},
    {"name": "Orion Pharma Limited",                     "kw": ["orion pharma"]},
    {"name": "Novo Nordisk Pharma",                      "kw": ["novo nordisk"]},
    {"name": "Ziska Pharmaceuticals Limited",            "kw": ["ziska"]},
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
    """
    Fetch a URL via ScrapingAnt residential proxy API.
    Returns raw HTML string or None on failure.
    browser=false → 1 credit per request (sufficient for medex.com.bd).
    """
    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(
                ANT_URL,
                params={
                    "url": url,
                    "x-api-key": api_key,
                    "browser": "false",
                },
                timeout=30,
            )
            if r.status_code == 200:
                html = r.text
                if html:
                    return html
                log.warning("  Empty content from ScrapingAnt (attempt %d)", attempt)
            elif r.status_code == 422:
                log.error("  ScrapingAnt 422 — invalid params or URL: %s", url)
                return None
            elif r.status_code == 403:
                log.error("  ScrapingAnt 403 — check your API key")
                return None
            else:
                log.warning("  ScrapingAnt HTTP %d (attempt %d)", r.status_code, attempt)
        except Exception as e:
            log.warning("  ScrapingAnt request error (attempt %d): %s", attempt, e)

        if attempt < retries:
            time.sleep(random.uniform(2, 5))

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
    """Detect CAPTCHA / soft-block pages."""
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

    # Strength & dosage form from <title>
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
# SITE NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def discover_companies(api_key: str) -> list[dict]:
    found: dict[str, dict] = {}
    page = 1
    while True:
        log.info("Fetching companies page %d …", page)
        html = fetch(f"{BASE_URL}/companies?page={page}", api_key)
        if not html:
            log.error("Failed to fetch companies page %d", page)
            break
        soup  = BeautifulSoup(html, "lxml")
        links = soup.select("a[href*='/companies/']")
        if not links:
            break
        for a in links:
            href = a.get("href", "")
            if not re.search(r"/companies/\d+/", href):
                continue
            name_lower = a.get_text(strip=True).lower()
            for t in TOP_20:
                if t["name"] in found:
                    continue
                if any(kw in name_lower for kw in t["kw"]):
                    base = href.split("/brands")[0].rstrip("/")
                    full = base if base.startswith("http") else BASE_URL + base
                    found[t["name"]] = {
                        "name": t["name"],
                        "site_name": a.get_text(strip=True),
                        "brands_url": full + "/brands",
                    }
                    log.info("  ✓ %-45s → %s", t["name"], a.get_text(strip=True))
        if len(found) == len(TOP_20) or not soup.select_one("a[rel='next']"):
            break
        page += 1
        time.sleep(random.uniform(0.5, 1.5))

    missing = [t["name"] for t in TOP_20 if t["name"] not in found]
    if missing:
        log.warning("Companies not found: %s", missing)
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
        log.info("  Page %d → +%d (total %d)", page, added, len(urls))
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
    # Sync checkpoint ↔ CSV
    cp   = load_cp() if resume else {"scraped_urls": []}
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
    log.info("Medex v5 — ScrapingAnt residential proxies")
    log.info("=" * 70)

    companies = discover_companies(api_key)
    log.info("Companies found: %d / 20", len(companies))

    # Open output CSV
    append   = output_path.exists() and resume
    out_file = open(output_path, "a" if append else "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(out_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if not append:
        writer.writeheader()

    total = 0
    try:
        company_urls_cache = cp.setdefault("company_urls", {})

        for ci, company in enumerate(companies, 1):
            log.info("")
            log.info("━" * 70)
            log.info("[%d/%d] %s", ci, len(companies), company["name"])

            # Use cached brand URLs if available (saves API credits on resume)
            co_key = company["name"]
            if co_key in company_urls_cache:
                brand_urls = company_urls_cache[co_key]
                log.info("  Brands: %d (from cache)", len(brand_urls))
            else:
                brand_urls = get_brand_urls(company, api_key)
                company_urls_cache[co_key] = brand_urls
                save_cp(cp)
                log.info("  Brands: %d", len(brand_urls))

            # Skip entire company if all brands already done
            todo = [u for u in brand_urls if u not in done]
            if not todo:
                log.info("  → Already complete, skipping")
                continue
            log.info("  → %d to scrape, %d already done", len(todo), len(brand_urls) - len(todo))

            co_count = 0
            for bi, burl in enumerate(brand_urls, 1):
                if burl in done:
                    continue

                log.info("  [%d/%d] %s", bi, len(brand_urls), burl)
                html = fetch(burl, api_key)

                if html is None:
                    log.warning("  Fetch failed — skipping")
                else:
                    row = parse_brand(html, burl, company["name"])
                    if row:
                        writer.writerow(row)
                        out_file.flush()
                        co_count += 1
                        total    += 1
                        done.add(burl)
                        cp["scraped_urls"].append(burl)

                        if total % 50 == 0:
                            save_cp(cp)
                            log.info("  ✓ checkpoint: %d total rows", total)
                    else:
                        log.warning("  No data parsed")

                time.sleep(base_delay + random.uniform(0, 0.5))

            log.info("  → %d written for %s", co_count, company["name"])
            if ci < len(companies):
                time.sleep(random.uniform(2, 4))

    finally:
        out_file.close()
        save_cp(cp)
        log.info("=" * 70)
        log.info("Done — %d total rows | %s", total, output_path)


def main():
    ap = argparse.ArgumentParser(description="Medex scraper — ScrapingAnt edition")
    ap.add_argument("--api-key", required=True, help="ScrapingAnt API key")
    ap.add_argument("--output", default=str(OUTPUT_CSV))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--delay", type=float, default=BASE_DELAY)
    args = ap.parse_args()
    run(args.api_key, Path(args.output), args.resume, args.delay)


if __name__ == "__main__":
    main()
