"""
/api/chat — main conversational endpoint.

Intent router: detects price / interaction / dosage / general, searches the DB,
injects real data as context, then lets the LLM write a warm human message.
Structured data (prices, cards) is always overridden with actual DB values —
the LLM never gets to invent numbers.
"""
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import get_optional_user_id
from app.models import ChatRequest, ChatResponseEnvelope, MedicineRow
from app.services.search import search_medicines, get_cheapest_generics
from app.services.interactions import check_interactions
from app.services.llm import chat as llm_chat, _detect_language

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ── Intent detection ──────────────────────────────────────────────────────────

PRICE_KEYWORDS = {
    "price", "daam", "cost", "দাম", "কত", "koto", "cheap", "sasta",
    "alternative", "generic", "কম দাম", "সস্তা", "দামী", "দাম কত",
}
INTERACTION_KEYWORDS = {
    "interact", "together", "combine", "safe", "reaction", "mixed",
    "ek shathe", "একসাথে", "একই সাথে", "সাথে",
}
DOSAGE_KEYWORDS = {
    "dose", "dosage", "kore khabo", "how to take", "when to take",
    "খাবো", "কখন", "কিভাবে খাব", "নেওয়া", "খাওয়া",
}


def _detect_intent(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in INTERACTION_KEYWORDS):
        return "interaction"
    if any(k in lower for k in PRICE_KEYWORDS):
        return "price"
    if any(k in lower for k in DOSAGE_KEYWORDS):
        return "dosage"
    return "general"


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _lookup_brands(query: str) -> list[MedicineRow]:
    """Search DB for medicine and expand to all brands of found generic."""
    drug_query = " ".join(query.split()[:4])
    medicines = await search_medicines(drug_query, limit=5)
    if not medicines:
        return []
    generic = medicines[0].generic_name
    return await get_cheapest_generics(generic, limit=10)


def _brands_to_context(brands: list[MedicineRow]) -> str:
    lines = [f"  • {b.brand_name} ({b.manufacturer}) — ৳{b.price_per_unit}/{b.unit}" for b in brands]
    generic = brands[0].generic_name if brands else "?"
    return f"Real-time DB prices for {generic}:\n" + "\n".join(lines)


def _brands_to_data(brands: list[MedicineRow]) -> list[dict]:
    cheapest_price = min(b.price_per_unit for b in brands)
    return [
        {
            "brand":        b.brand_name,
            "generic":      b.generic_name,
            "manufacturer": b.manufacturer,
            "price":        float(b.price_per_unit),   # always a number
            "unit":         b.unit,
            "is_cheapest":  float(b.price_per_unit) == float(cheapest_price),
        }
        for b in brands
    ]


def _medicine_context(medicines: list[MedicineRow]) -> str:
    """Compact context for non-price queries (dosage, general)."""
    lines = [
        f"  • {m.brand_name} — {m.generic_name} {m.strength or ''} ({m.dosage_form or 'tablet'}), ৳{m.price_per_unit}/{m.unit}, by {m.manufacturer}".rstrip()
        for m in medicines[:4]
    ]
    return "Bangladesh medicine DB:\n" + "\n".join(lines)


# ── SSE streaming ─────────────────────────────────────────────────────────────

async def _stream_envelope(envelope: ChatResponseEnvelope):
    """Stream message tokens first, then emit format + data payload."""
    words = envelope.message.split(" ")
    for i, word in enumerate(words):
        chunk = word + ("" if i == len(words) - 1 else " ")
        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'format': envelope.format, 'data': envelope.data})}\n\n"


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("")
async def chat_endpoint(
    request: ChatRequest,
    user_id: str | None = Depends(get_optional_user_id),
):
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )
    intent = _detect_intent(last_user_msg)
    lang   = _detect_language(last_user_msg)

    envelope: ChatResponseEnvelope

    # ── Price intent ─────────────────────────────────────────────────────────
    if intent == "price":
        brands = await _lookup_brands(last_user_msg)
        if len(brands) >= 2:
            context = _brands_to_context(brands)
            # LLM writes a warm 1-sentence message; we enforce real DB prices
            envelope = await llm_chat(request.messages, context_injection=context, language_hint=lang)
            envelope.format = "price_table"
            envelope.data   = _brands_to_data(brands)
        elif brands:
            # Single brand found — still inject context, let LLM answer
            envelope = await llm_chat(
                request.messages,
                context_injection=_medicine_context(brands),
                language_hint=lang,
            )
        else:
            # Not in DB — LLM answers from general knowledge
            envelope = await llm_chat(request.messages, language_hint=lang)

    # ── Interaction intent ────────────────────────────────────────────────────
    elif intent == "interaction":
        words = last_user_msg.replace(",", " ").split()
        candidates = [w.strip("?.,!") for w in words if len(w) > 4]
        cards = await check_interactions(candidates[:6])
        context = ""
        if cards:
            context = "Interaction data from database:\n" + "\n".join(
                f"  {c.drug_a} + {c.drug_b}: [{c.severity.upper()}] {c.description}"
                for c in cards
            )
        envelope = await llm_chat(request.messages, context_injection=context or None, language_hint=lang)

    # ── Dosage intent ─────────────────────────────────────────────────────────
    elif intent == "dosage":
        medicines = await search_medicines(" ".join(last_user_msg.split()[:4]), limit=4)
        context = _medicine_context(medicines) if medicines else None
        envelope = await llm_chat(request.messages, context_injection=context, language_hint=lang)

    # ── General ───────────────────────────────────────────────────────────────
    else:
        # Try to enrich with DB context if the query mentions a known medicine
        medicines = await search_medicines(" ".join(last_user_msg.split()[:4]), limit=3)
        context = _medicine_context(medicines) if medicines else None
        envelope = await llm_chat(request.messages, context_injection=context, language_hint=lang)

    return StreamingResponse(
        _stream_envelope(envelope),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
