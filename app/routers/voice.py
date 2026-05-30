"""
/api/voice — Banglish / Bengali / English voice-note to chat response.
Browser sends audio blob → Gemini Flash transcribes → LLM answers in same language.
"""
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from app.auth import get_optional_user_id
from app.database import get_supabase
from app.models import ChatResponseEnvelope, ChatMessage
from app.services.gemini import transcribe_audio
from app.services.llm import chat as llm_chat
import uuid
import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/webm;codecs=opus", "audio/ogg", "audio/mpeg", "audio/wav", "audio/mp4"}


def _normalise_audio_mime(ct: str | None) -> str:
    ct = (ct or "audio/webm").lower().split(";")[0].strip()
    # Browsers vary — accept any audio subtype
    if ct.startswith("audio/"):
        return ct
    return "audio/webm"


@router.post("", response_model=ChatResponseEnvelope)
async def voice_query(
    file: UploadFile = File(...),
    user_id: str | None = Depends(get_optional_user_id),
):
    mime = _normalise_audio_mime(file.content_type)

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file received.")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio must be under 25 MB.")

    # Store in Supabase Storage only for logged-in users
    if user_id:
        try:
            sb = get_supabase()
            path = f"{uuid.uuid4()}.webm"
            sb.storage.from_("voice-notes").upload(
                path=path,
                file=data,
                file_options={"content-type": mime},
            )
        except Exception as exc:
            log.warning("Voice storage upload failed (non-fatal): %s", exc)

    # Transcribe with Gemini Flash — returns both transcript + detected language
    try:
        transcript_obj = await transcribe_audio_with_language(data, mime_type=mime)
    except Exception as exc:
        log.error("Voice transcription failed: %s", exc)
        return ChatResponseEnvelope(
            message="Sorry, I couldn't understand the audio. Please try again or type your question.",
            format="text",
            data=None,
        )

    transcript = transcript_obj.get("transcript", "")
    language   = transcript_obj.get("language", "english")   # "bangla" | "banglish" | "english"

    if not transcript.strip():
        return ChatResponseEnvelope(
            message="Sorry, I couldn't understand the audio. Please try again or type your question.",
            format="text",
            data=None,
        )

    # Build the user message — include language hint so LLM responds correctly
    messages = [ChatMessage(role="user", content=transcript)]
    envelope = await llm_chat(messages, language_hint=language)

    # Prepend what was heard so the user can verify
    heard_label = f'[শুনলাম: "{transcript}"]' if language in ("bangla", "banglish") else f'[Heard: "{transcript}"]'
    envelope.message = f"{heard_label}\n\n{envelope.message}"
    return envelope


# ── Extended transcription that also returns language ────────────────────────

import base64
import json
import re
import google.generativeai as genai
from app.config import get_settings

_configured = False

def _ensure():
    global _configured
    if not _configured:
        genai.configure(api_key=get_settings().gemini_api_key)
        _configured = True


async def transcribe_audio_with_language(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
    """
    Transcribe audio. Returns:
      {
        "transcript": "<original words>",
        "english":    "<english translation if not already English>",
        "language":   "bangla" | "banglish" | "english"
      }
    """
    _ensure()
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = """Transcribe this audio clip. The speaker may use Bengali, Banglish (Bengali + English mix), or English.

Return ONLY valid JSON with no markdown:
{
  "transcript": "<exact words spoken>",
  "english": "<English translation if Bengali/Banglish, else same as transcript>",
  "language": "<one of: bangla | banglish | english>"
}"""

    audio_part = {"mime_type": mime_type, "data": base64.b64encode(audio_bytes).decode()}
    try:
        response = model.generate_content([prompt, audio_part])
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        obj = json.loads(raw)
        return obj
    except Exception as exc:
        log.warning("Audio transcription parse error: %s", exc)
        # Return raw text as English fallback
        return {"transcript": raw if "raw" in dir() else "", "english": "", "language": "english"}
