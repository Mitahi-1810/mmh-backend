"""
Semantic + text search for matching extracted medicine names to the medicines DB.

Strategy (fastest-first cascade):
  1. Exact brand_name ILIKE match  (covers ~70% of cases)
  2. Generic name ILIKE match
  3. Trigram / partial match on both columns
  4. (Future) pgvector cosine similarity once embeddings are populated

Fuzzy matching uses rapidfuzz (C++ SequenceMatcher) which is 10–100x faster and
more accurate than difflib, especially for short strings like drug brand names.
WRatio (Weighted Ratio) handles character transpositions, partial matches, and
OCR-typical errors (0/O, l/1, rn/m) better than plain ratio.

All DB calls use the Supabase Python client (REST/PostgREST).
"""
import logging
import time
from typing import Any

from rapidfuzz import fuzz, process as rf_process

from app.database import get_supabase

log = logging.getLogger(__name__)

# How many medicines to load for the detailed RAG block (brand + generic + strength)
RAG_LIMIT = 300

# Simple process-lifetime cache (TTL = 1 hour)
_rag_cache: list[dict[str, Any]] = []
_rag_cache_at: float = 0.0
_RAG_TTL = 3600.0


def get_rag_medicines() -> list[dict[str, Any]]:
    """
    Return top RAG_LIMIT medicines (by id) from the medicines table.
    Cached for 1 hour so we don't hit Supabase on every prescription upload.
    """
    global _rag_cache, _rag_cache_at
    now = time.monotonic()
    if _rag_cache and (now - _rag_cache_at) < _RAG_TTL:
        return _rag_cache
    try:
        sb = get_supabase()
        res = (
            sb.table("medicines")
            .select("brand_name, generic_name, strength, dosage_form")
            .order("id")
            .limit(RAG_LIMIT)
            .execute()
        )
        _rag_cache = res.data or []
        _rag_cache_at = now
        log.info("RAG cache refreshed: %d medicines", len(_rag_cache))
        return _rag_cache
    except Exception as exc:
        log.warning("RAG medicines fetch failed: %s", exc)
        return _rag_cache  # return stale cache on error


def match_medicine_by_name(brand_name: str) -> dict[str, Any] | None:
    """
    Try to find a matching row in the medicines table for a given brand name.

    Returns a dict with at least {id, brand_name, generic_name, strength, unit_price}
    or None if no confident match is found.

    Cascade:
      1. brand_name exact (case-insensitive)
      2. brand_name ILIKE '<brand>%'  (prefix)
      3. brand_name ILIKE '%<brand>%' (substring)
      4. generic_name substring match
    """
    if not brand_name or not brand_name.strip():
        return None

    sb = get_supabase()
    clean = brand_name.strip()

    # --- Tier 1: exact brand match (case-insensitive) ---
    try:
        res = (
            sb.table("medicines")
            .select("id, brand_name, generic_name, strength, dosage_form, price_per_unit, unit")
            .ilike("brand_name", clean)
            .limit(1)
            .execute()
        )
        if res.data:
            log.debug("Match T1 exact: %s → %s", clean, res.data[0]["brand_name"])
            return res.data[0]
    except Exception as exc:
        log.warning("Tier-1 match error: %s", exc)

    # --- Tier 2: prefix brand match ---
    try:
        res = (
            sb.table("medicines")
            .select("id, brand_name, generic_name, strength, dosage_form, price_per_unit, unit")
            .ilike("brand_name", f"{clean}%")
            .limit(1)
            .execute()
        )
        if res.data:
            log.debug("Match T2 prefix: %s → %s", clean, res.data[0]["brand_name"])
            return res.data[0]
    except Exception as exc:
        log.warning("Tier-2 match error: %s", exc)

    # --- Tier 3: substring brand match ---
    try:
        res = (
            sb.table("medicines")
            .select("id, brand_name, generic_name, strength, dosage_form, price_per_unit, unit")
            .ilike("brand_name", f"%{clean}%")
            .limit(1)
            .execute()
        )
        if res.data:
            log.debug("Match T3 substr: %s → %s", clean, res.data[0]["brand_name"])
            return res.data[0]
    except Exception as exc:
        log.warning("Tier-3 match error: %s", exc)

    # --- Tier 4: generic name substring match ---
    try:
        res = (
            sb.table("medicines")
            .select("id, brand_name, generic_name, strength, dosage_form, price_per_unit, unit")
            .ilike("generic_name", f"%{clean}%")
            .limit(1)
            .execute()
        )
        if res.data:
            log.debug("Match T4 generic: %s → %s", clean, res.data[0]["generic_name"])
            return res.data[0]
    except Exception as exc:
        log.warning("Tier-4 match error: %s", exc)

    log.debug("No DB match for: %s", clean)
    return None


# ── Fuzzy post-OCR correction ─────────────────────────────────────────────────
# All 9,815 brand names are held in a process-lifetime cache so fuzzy matching
# runs in memory (< 5 ms per name) without hitting the DB on every prescription.

_fuzzy_cache: dict[str, str]  = {}   # lowercase_brand → canonical brand_name
_fuzzy_generic: dict[str, str] = {}  # lowercase_brand → generic_name
_fuzzy_cache_at: float = 0.0
_FUZZY_TTL = 3600.0


def _ensure_fuzzy_cache() -> None:
    global _fuzzy_cache, _fuzzy_generic, _fuzzy_cache_at
    now = time.monotonic()
    if _fuzzy_cache and (now - _fuzzy_cache_at) < _FUZZY_TTL:
        return
    try:
        sb = get_supabase()
        # Fetch ALL brand + generic names — this is the full correction corpus
        res = sb.table("medicines").select("brand_name, generic_name").execute()
        fc: dict[str, str] = {}
        fg: dict[str, str] = {}
        for row in (res.data or []):
            key = (row["brand_name"] or "").lower().strip()
            if key:
                fc[key] = row["brand_name"]
                fg[key] = (row.get("generic_name") or "").lower().strip()
        _fuzzy_cache = fc
        _fuzzy_generic = fg
        _fuzzy_cache_at = now
        log.info("Fuzzy name cache loaded: %d brands", len(fc))
    except Exception as exc:
        log.warning("Fuzzy cache load failed: %s", exc)


def get_medicine_names_for_vocab() -> list[str]:
    """
    Return the list of all canonical brand names loaded in the fuzzy cache.
    Used by medicine_vocab.py to build the TrOCR LogitsProcessor trie.
    Triggers a cache load from DB if the cache is empty.
    """
    _ensure_fuzzy_cache()
    return list(_fuzzy_cache.values())  # canonical casing (e.g. "Napa", not "napa")


def get_all_brand_names_compact() -> str:
    """
    Return all BD brand names as a compact comma-separated string for Gemini injection.
    e.g. "Napa, Seclo, Losec, Metformin, ..."
    Gemini 2.5 Flash has a 1M-token context so injecting all ~7,600+ names is safe.
    This makes OCR truly medicine-aware — Gemini can recognise ANY known BD brand.
    """
    _ensure_fuzzy_cache()
    names = sorted(_fuzzy_cache.values())  # sorted for deterministic ordering
    return ", ".join(names)


def fuzzy_correct_brand(
    ocr_name: str,
    gemini_generic: str | None = None,
) -> dict[str, Any]:
    """
    Compare an OCR'd brand name against all known Bangladeshi brands using
    rapidfuzz WRatio (C++ weighted ratio — handles char swaps, OCR errors,
    partial matches, and word reordering all in one score).

    Returns:
        {
          "status":   "confirmed" | "suggested" | "ambiguous" | "unknown",
          "brand":    str | None,    # canonical DB spelling (if different from OCR)
          "generic":  str | None,    # generic of the DB match
          "safe":     bool | None,   # True = same generic family, False = different
          "score":    float,         # similarity 0.0–100.0 (rapidfuzz convention)
        }

    Safety tiers:
      confirmed  — exact DB match (normalised); OCR name is a known brand
      suggested  — WRatio ≥ 78 AND same generic family → safe one-tap accept
      ambiguous  — WRatio ≥ 78 BUT different generics → red warning; NEVER auto-correct
      unknown    — no close match (WRatio < 70); OCR reading kept as-is

    Why WRatio?
      WRatio is a weighted combination of ratio, partial_ratio, token_sort_ratio,
      and token_set_ratio. For drug names this handles:
        • Simple OCR swaps: "Napa" vs "Mapa"  (character substitution)
        • Extra/missing chars: "Seclo" vs "Secloo" (insertion)
        • Word order: "Metformin 500" vs "500 Metformin"
        • Prefix matches: "Napa" vs "Napa Extra"
    """
    _ensure_fuzzy_cache()
    if not _fuzzy_cache:
        return {"status": "unknown", "brand": None, "generic": None, "safe": None, "score": 0.0}

    normalized = (ocr_name or "").lower().strip()
    if not normalized:
        return {"status": "unknown", "brand": None, "generic": None, "safe": None, "score": 0.0}

    # ── Tier 1: exact match (fastest path, ~70% of cases) ───────────────────
    if normalized in _fuzzy_cache:
        return {
            "status":  "confirmed",
            "brand":   _fuzzy_cache[normalized],
            "generic": _fuzzy_generic.get(normalized),
            "safe":    True,
            "score":   100.0,
        }

    # ── Tier 2: rapidfuzz WRatio fuzzy match ────────────────────────────────
    # process.extractOne returns (match_key, score, index) — O(n) but C++ fast
    # Score threshold: 70 = minimum plausible match; below this → unknown
    all_keys = list(_fuzzy_cache.keys())
    match = rf_process.extractOne(
        normalized,
        all_keys,
        scorer=fuzz.WRatio,
        score_cutoff=70,
    )

    if match is None:
        return {"status": "unknown", "brand": None, "generic": None, "safe": None, "score": 0.0}

    best_key, score, _ = match
    best_brand   = _fuzzy_cache[best_key]
    best_generic = _fuzzy_generic.get(best_key, "")

    # ── Safety gate: compare generics ───────────────────────────────────────
    # "same generic family" = one name is a substring of the other
    # e.g. "paracetamol" in "paracetamol 500mg" → same
    # e.g. "furosemide" vs "omeprazole" → different → ambiguous (dangerous)
    safe: bool | None = None
    if gemini_generic and best_generic:
        g1 = gemini_generic.lower().strip()
        g2 = best_generic.lower().strip()
        safe = (g1 in g2) or (g2 in g1)
    elif not gemini_generic:
        safe = None  # can't verify — show suggestion but not flagged dangerous

    # Determine status
    # A high WRatio score (≥78) is reliable enough for a "Did you mean?" suggestion.
    # Safe = False means demonstrably different generics → ambiguous (show red warning)
    if safe is False:
        status = "ambiguous"
    elif score >= 78:
        status = "suggested"
    else:
        # Score 70–77: possible match but low confidence — still show it
        status = "suggested"

    log.debug(
        "Fuzzy (WRatio): '%s' → '%s' (score=%.1f, safe=%s, status=%s)",
        ocr_name, best_brand, score, safe, status,
    )
    return {
        "status":  status,
        "brand":   best_brand,
        "generic": best_generic or None,
        "safe":    safe,
        "score":   score,
    }


def get_patient_context(user_id: str) -> tuple[list[str], list[str]]:
    """
    Fetch patient conditions and current medicines for RAG context.
    Returns (conditions, current_medicine_names).
    """
    sb = get_supabase()
    conditions: list[str] = []
    current_meds: list[str] = []

    try:
        profile = (
            sb.table("user_profiles")
            .select("conditions")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if profile.data and profile.data.get("conditions"):
            conditions = profile.data["conditions"]
    except Exception as exc:
        log.debug("Patient conditions fetch failed: %s", exc)

    try:
        # Get distinct medicines from the last 10 confirmed prescriptions
        recent = (
            sb.table("prescription_medicines")
            .select("brand_name")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(30)
            .execute()
        )
        if recent.data:
            seen: set[str] = set()
            for row in recent.data:
                name = row.get("brand_name", "")
                if name and name not in seen:
                    seen.add(name)
                    current_meds.append(name)
                    if len(current_meds) >= 10:
                        break
    except Exception as exc:
        log.debug("Current meds fetch failed: %s", exc)

    return conditions, current_meds
