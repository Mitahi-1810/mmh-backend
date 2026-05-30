"""
LLM service — Gemini 2.0 Flash for all text chat (same model used for prescription OCR).
This keeps the stack to a single LLM provider and uses the free tier generously.

All responses use the ChatResponseEnvelope JSON format so the frontend
knows which React component to render.
"""
import json
import re
import logging

import google.generativeai as genai
from app.config import get_settings
from app.models import ChatResponseEnvelope, ChatMessage

log = logging.getLogger(__name__)

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        genai.configure(api_key=get_settings().gemini_api_key)
        _configured = True


SYSTEM_PROMPT = """You are Sanjibani — the medical assistant on mmh.io, built for Bangladesh.

Personality: warm, direct, knowledgeable. Talk like a smart friend who knows medicine — not a database printout. Skip filler phrases like "It is important to note that", "I'd be happy to help", "Please be advised", "As an AI". Just answer.

You help with: medicine prices & generics in Bangladesh, drug interactions, dosage guidance, reading prescriptions.

─── RESPONSE FORMAT ───────────────────────────────────────────────────────────
Always return exactly this JSON — nothing outside it, no markdown fences:
{
  "message": "<your text>",
  "format": "<format name>",
  "data": <structured payload or null>
}

Formats:
  "text"              → general answers, explanations, greetings
  "price_table"       → comparing prices / finding generics
                        data = [{brand, generic, manufacturer, price (NUMBER not string), unit, is_cheapest}]
  "interaction_cards" → drug interaction check
                        data = [{drug_a, drug_b, severity ("major"|"moderate"|"minor"), description}]
  "dosage_card"       → how to take a medicine
                        data = {name, generic, dose, frequency, max_daily, warnings, with_food (bool)}
  "prescription_list" → listing drugs from a scanned prescription
                        data = [{brand, generic, dose, frequency}]
  "reminder_confirm"  → confirming a reminder
                        data = {medicine, remind_at (ISO string), note}

─── WRITING RULES ─────────────────────────────────────────────────────────────
• For "price_table", "dosage_card", "interaction_cards":
  Write a SHORT 1-sentence message — the card below shows details, don't repeat them.
  ✓ "Napa is the cheapest option — here's how the brands compare."
  ✗ "I found 5 brands of Paracetamol. The cheapest brand is Napa at ৳2.50 per tablet."

• For "text": write a full, conversational answer. No bullet-point walls.

• Prices in data must be plain numbers (2.5, not "2.5" or "৳2.50").

• Only add "consult a doctor" when the topic is genuinely risky — one short sentence, not a disclaimer paragraph.

• Never say "Found X results" or "Based on the data provided". Lead with the actual answer.
"""


def _parse_envelope(raw: str) -> ChatResponseEnvelope:
    """Extract the JSON envelope from the LLM output, tolerating minor formatting."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return ChatResponseEnvelope(**obj)
    except Exception:
        # Try to extract the outermost {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                return ChatResponseEnvelope(**obj)
            except Exception:
                pass
        return ChatResponseEnvelope(message=raw, format="text", data=None)


def _detect_language(text: str) -> str:
    """
    Detect if the user's message is Bangla, Banglish, or English.
    Returns 'bangla', 'banglish', or 'english'.
    """
    # Bengali Unicode range: U+0980–U+09FF
    bangla_chars = sum(1 for ch in text if 'ঀ' <= ch <= '৿')
    if bangla_chars > 2:
        return "bangla"

    # Banglish: common Bengali romanisation patterns
    banglish_words = {
        "ami", "tumi", "apni", "kemon", "achen", "acho", "ki", "kore", "khabo",
        "dao", "den", "nei", "ache", "hobe", "hoi", "bolo", "bolun", "dada",
        "bhai", "apu", "vai", "napa", "seclo", "osudh", "doctor", "daktar",
        "khawa", "khabar", "daam", "koto", "taka", "boro", "choto",
    }
    words_lower = set(text.lower().split())
    if words_lower & banglish_words:
        return "banglish"

    return "english"


async def chat(
    messages: list[ChatMessage],
    context_injection: str | None = None,
    language_hint: str | None = None,     # "bangla" | "banglish" | "english" | None (auto-detect)
) -> ChatResponseEnvelope:
    """
    Send messages to Gemini 2.0 Flash and return a parsed ChatResponseEnvelope.
    context_injection: extra context prepended to the last user message (e.g., DB results).
    """
    _ensure_configured()

    # Determine language (hint from voice takes priority, else auto-detect)
    last_user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
    lang = language_hint or _detect_language(last_user_text)

    # Language-specific instruction appended to system prompt
    if lang == "bangla":
        lang_instruction = (
            "\n\nIMPORTANT: The user is writing in Bengali. "
            "Write the 'message' field entirely in Bengali (বাংলা). "
            "However, ALL structured data values — medicine brand names, generic names, "
            "manufacturer names, dose amounts, units, price values, severity labels, "
            "field labels — MUST remain in English. "
            "Only the conversational 'message' prose should be in Bengali."
        )
    elif lang == "banglish":
        lang_instruction = (
            "\n\nIMPORTANT: The user is writing in Banglish (Bengali romanised). "
            "Write the 'message' field entirely in Bengali (বাংলা). "
            "However, ALL structured data values — medicine brand names, generic names, "
            "manufacturer names, dose amounts, units, price values, severity labels, "
            "field labels — MUST remain in English. "
            "Only the conversational 'message' prose should be in Bengali."
        )
    else:
        lang_instruction = "\n\nIMPORTANT: The user is writing in English. Respond in English."

    full_system = SYSTEM_PROMPT + lang_instruction

    # Build the conversation history for Gemini
    # Gemini uses "user" / "model" roles (not "assistant")
    gemini_history = []
    for i, msg in enumerate(messages[:-1]):  # all but the last
        role = "user" if msg.role == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg.content]})

    # Last message (the one we're responding to)
    last = messages[-1] if messages else ChatMessage(role="user", content="")
    last_content = last.content
    if context_injection:
        last_content = f"{context_injection}\n\nUser question: {last_content}"

    try:
        model = genai.GenerativeModel(
            model_name="gemini-flash-lite-latest",
            system_instruction=full_system,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(last_content)
        raw = response.text or ""
    except Exception as exc:
        log.error("Gemini chat error: %s", exc)
        return ChatResponseEnvelope(
            message="I'm having trouble connecting right now. Please try again in a moment.",
            format="text",
            data=None,
        )

    return _parse_envelope(raw)
