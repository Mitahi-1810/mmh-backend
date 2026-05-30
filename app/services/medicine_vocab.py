"""
Medicine vocabulary tools for constrained OCR decoding and strength-based disambiguation.

Two main exports:

1. MedicineVocabLogitsProcessor
   A HuggingFace LogitsProcessor that biases TrOCR beam search toward known medicine
   brand names.  During decoding of the first word of a prescription line (the brand
   name position) it walks a prefix trie of tokenised medicine names and adds a log-
   probability boost to token IDs that continue valid medicine-name paths.

   Effect: if OCR is ambiguous between "Stira" and "Stiba", and only "Stiba" is in the
   BD medicine vocabulary, the decoder will prefer "Stiba" at the character level.

2. disambiguate_by_strength(ocr_brand, extracted_strength, supabase_client)
   Queries the medicines table to find the row that best matches both the OCR'd brand
   name AND the strength found on the same line.  This resolves the dangerous case where
   two similarly-spelled brands (e.g. "Lasix" vs "Locid") share a fuzzy match but have
   completely different standard strengths — the one whose strength matches the DB row
   wins.
"""
import logging
import re
from typing import Any

import torch
from rapidfuzz import fuzz, process as rf_process
from transformers import LogitsProcessor

log = logging.getLogger(__name__)

# How many tokens from the start of a line are considered "brand name position"
_BRAND_POSITION_MAX_TOKENS = 10
# Log-probability boost applied to tokens that continue a known medicine name
_DEFAULT_BOOST = 3.0
# rapidfuzz threshold for strength disambiguation fallback
_BRAND_FUZZY_CUTOFF = 72.0


# ── Prefix trie builder ───────────────────────────────────────────────────────

def build_medicine_trie(medicine_names: list[str], tokenizer) -> dict:
    """
    Build a prefix trie of BPE token-ID sequences for all medicine names.

    The trie maps  tok_id → {tok_id → {..., "__end__": True}}.
    It is used at decoding time: given the current beam's partial token sequence,
    we walk the trie and boost the next-token logits for any token that continues
    a valid medicine name path.

    Args:
        medicine_names: list of canonical brand-name strings (e.g. ["Napa", "Seclo", ...])
        tokenizer: the TrOCRProcessor's tokenizer (RoBERTa-based BPE)

    Returns:
        Nested dict trie.  Example for "Napa" and "Napro":
          {<N>: {<a>: {<p>: {<a>: {"__end__": True}, <r>: {<o>: {"__end__": True}}}}}}
    """
    trie: dict = {}
    built = 0
    for name in medicine_names:
        name = name.strip()
        if not name:
            continue
        try:
            token_ids = tokenizer(name, add_special_tokens=False).input_ids
        except Exception:
            continue
        if not token_ids:
            continue
        node = trie
        for tid in token_ids:
            node = node.setdefault(tid, {})
        node["__end__"] = True
        built += 1

    log.info("Medicine trie built: %d names → %d root entries", built, len(trie))
    return trie


# ── LogitsProcessor ───────────────────────────────────────────────────────────

class MedicineVocabLogitsProcessor(LogitsProcessor):
    """
    Injects medicine-vocabulary knowledge into TrOCR beam search.

    At each decode step, if we are still within the "brand name position"
    (first _BRAND_POSITION_MAX_TOKENS generated tokens), the processor:
      1. Walks the medicine prefix trie using the tokens decoded so far.
      2. If the current partial sequence matches a trie path, it adds
         `boost` to the log-probabilities of tokens that continue valid paths.

    This makes the model prefer known medicine brand-name spellings over
    similar-sounding but unrecognised strings — without changing the model
    weights or requiring retraining.

    Usage:
        processor_list = LogitsProcessorList([
            MedicineVocabLogitsProcessor(names, tokenizer)
        ])
        output = model.generate(..., logits_processor=processor_list)
    """

    def __init__(
        self,
        medicine_names: list[str],
        tokenizer,
        boost: float = _DEFAULT_BOOST,
    ):
        self.trie  = build_medicine_trie(medicine_names, tokenizer)
        self.boost = boost
        self.bos_id = getattr(tokenizer, "bos_token_id", 0) or 0
        self.pad_id = getattr(tokenizer, "pad_token_id", 1) or 1

    def __call__(
        self,
        input_ids: torch.LongTensor,   # (batch * num_beams, seq_len)
        scores: torch.FloatTensor,      # (batch * num_beams, vocab_size)
    ) -> torch.FloatTensor:

        for beam_idx in range(input_ids.shape[0]):
            # Tokens generated so far (strip special tokens)
            seq = input_ids[beam_idx].tolist()
            effective = [
                t for t in seq
                if t != self.bos_id and t != self.pad_id and t > 3
            ]

            # Only apply brand-name boost in the first N generated tokens
            if len(effective) >= _BRAND_POSITION_MAX_TOKENS:
                continue

            # Walk the trie
            node = self.trie
            on_path = True
            for tok in effective:
                if tok in node:
                    node = node[tok]
                else:
                    on_path = False
                    break

            if not on_path or not isinstance(node, dict):
                continue

            # Boost next tokens that continue valid medicine-name paths
            for next_tok, child in node.items():
                if next_tok == "__end__":
                    continue
                if next_tok < scores.shape[-1]:
                    scores[beam_idx, next_tok] = scores[beam_idx, next_tok] + self.boost

        return scores


# ── Strength extraction helper ─────────────────────────────────────────────────

_STRENGTH_RE = re.compile(
    r'(\d+\.?\d*)\s*(mg|ml|mcg|µg|g\b|iu|IU|mEq|mmol|unit)',
    re.IGNORECASE,
)


def extract_strength(text: str) -> str | None:
    """
    Pull the first dosage strength from a text string.
    Returns normalised string like "500mg", "20mg", or None.
    """
    m = _STRENGTH_RE.search(text)
    if m:
        # Normalise: "500 mg" → "500mg"
        return f"{m.group(1)}{m.group(2).lower().strip()}"
    return None


# ── Strength-based brand disambiguation ───────────────────────────────────────

def disambiguate_by_strength(
    ocr_brand: str,
    extracted_strength: str | None,
    supabase_client,
) -> dict[str, Any]:
    """
    Resolve ambiguous OCR brand names using the strength found on the same line.

    Algorithm:
      1. Try exact brand + strength match in the DB.
         If found → confirmed, high confidence.
      2. If no exact brand but a fuzzy brand match (WRatio ≥ 72) exists AND
         that fuzzy-matched brand HAS the extracted strength → suggest correction.
      3. If exact brand exists but with a DIFFERENT standard strength → flag review.
      4. If no strength provided → fall back to brand-only fuzzy match.

    Returns:
        {
          "resolved_brand": str,        # The best brand name we could determine
          "generic_name": str | None,
          "strength": str | None,       # Confirmed strength from DB (may differ from OCR)
          "confidence_bonus": float,    # Additional confidence: 0.0–0.20
          "disambiguation": str,        # One of: strength_confirmed | strength_corrected |
                                        #   strength_mismatch | brand_only | not_found
          "medicine_id": int | None,
          "unit_price": float | None,
        }
    """
    result: dict[str, Any] = {
        "resolved_brand": ocr_brand,
        "generic_name": None,
        "strength": extracted_strength,
        "confidence_bonus": 0.0,
        "disambiguation": "not_found",
        "medicine_id": None,
        "unit_price": None,
    }

    if not ocr_brand:
        return result

    sb = supabase_client

    # ── Path A: exact brand match in DB ───────────────────────────────────────
    try:
        q = (
            sb.table("medicines")
            .select("id, brand_name, generic_name, strength, price_per_unit")
            .ilike("brand_name", ocr_brand.strip())
        )
        if extracted_strength:
            # Strip unit for partial match (e.g. "500" matches "500mg", "500 mg")
            num = re.match(r'(\d+\.?\d*)', extracted_strength)
            if num:
                q = q.ilike("strength", f"%{num.group(1)}%")
        rows = q.limit(3).execute().data or []
    except Exception as exc:
        log.debug("disambiguate DB error: %s", exc)
        rows = []

    if rows:
        best = rows[0]
        result.update({
            "resolved_brand":   best["brand_name"],
            "generic_name":     best.get("generic_name"),
            "strength":         best.get("strength") or extracted_strength,
            "confidence_bonus": 0.18 if extracted_strength else 0.08,
            "disambiguation":   "strength_confirmed" if extracted_strength else "brand_only",
            "medicine_id":      best.get("id"),
            "unit_price":       best.get("price_per_unit"),
        })
        log.debug("Disambiguate '%s' + '%s' → confirmed %s", ocr_brand, extracted_strength, best["brand_name"])
        return result

    # ── Path B: fuzzy brand match + check if that brand has the strength ──────
    if extracted_strength:
        try:
            # Fetch all brands (use the already-loaded fuzzy cache if available)
            from app.services.semantic_search import get_medicine_names_for_vocab
            all_brands = get_medicine_names_for_vocab()  # list[str]
        except Exception:
            all_brands = []

        if all_brands:
            match = rf_process.extractOne(
                ocr_brand.lower(),
                [b.lower() for b in all_brands],
                scorer=fuzz.WRatio,
                score_cutoff=_BRAND_FUZZY_CUTOFF,
            )
            if match:
                matched_brand = all_brands[match[2]]  # original casing
                log.debug("Fuzzy brand match: '%s' → '%s' (score=%.1f)", ocr_brand, matched_brand, match[1])

                # Now check if this fuzzy match + strength exists in DB
                num = re.match(r'(\d+\.?\d*)', extracted_strength)
                try:
                    rows2 = (
                        sb.table("medicines")
                        .select("id, brand_name, generic_name, strength, price_per_unit")
                        .ilike("brand_name", matched_brand)
                        .ilike("strength", f"%{num.group(1)}%" if num else "%")
                        .limit(1)
                        .execute()
                        .data or []
                    )
                except Exception:
                    rows2 = []

                if rows2:
                    best2 = rows2[0]
                    bonus = 0.12 if match[1] >= 85 else 0.06
                    result.update({
                        "resolved_brand":   best2["brand_name"],
                        "generic_name":     best2.get("generic_name"),
                        "strength":         best2.get("strength") or extracted_strength,
                        "confidence_bonus": bonus,
                        "disambiguation":   "strength_corrected",
                        "medicine_id":      best2.get("id"),
                        "unit_price":       best2.get("price_per_unit"),
                    })
                    log.info(
                        "Strength disambiguation: '%s' corrected to '%s' via strength '%s'",
                        ocr_brand, matched_brand, extracted_strength,
                    )
                    return result

    # ── Path C: brand-only, no strength help ──────────────────────────────────
    # Just do a plain ILIKE brand lookup, no strength filter
    try:
        rows3 = (
            sb.table("medicines")
            .select("id, brand_name, generic_name, strength, price_per_unit")
            .ilike("brand_name", f"%{ocr_brand.strip()}%")
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        rows3 = []

    if rows3:
        best3 = rows3[0]
        result.update({
            "resolved_brand":   best3["brand_name"],
            "generic_name":     best3.get("generic_name"),
            "strength":         best3.get("strength") or extracted_strength,
            "confidence_bonus": 0.05,
            "disambiguation":   "brand_only",
            "medicine_id":      best3.get("id"),
            "unit_price":       best3.get("price_per_unit"),
        })

    return result
