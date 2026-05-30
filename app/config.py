from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_key: str
    supabase_jwt_secret: str

    groq_api_key: str
    gemini_api_key: str

    frontend_url: str = "http://localhost:3000"
    medex_scrape_delay_seconds: float = 1.0
    medex_max_retries: int = 3

    # Web Push (VAPID) — generate with: python -m pywebpush --gen-keys
    vapid_private_key: str = ""
    vapid_public_key: str = ""
    vapid_claims_email: str = "admin@mmh.io"

    # WhatsApp — provider: "callmebot" | "twilio" | "meta"
    whatsapp_enabled: bool = False
    whatsapp_provider: str = "callmebot"   # fastest for demos

    # CallMeBot (instant, free — recommended for demo)
    callmebot_api_key: str = ""

    # Twilio sandbox (free trial, 5-min setup)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = "whatsapp:+14155238886"  # Twilio sandbox default

    # Meta Cloud API (production — requires business verification)
    whatsapp_api_token: str = ""
    whatsapp_phone_number_id: str = ""

    # BD timezone offset for dose generation
    tz_offset_hours: int = 6  # BST = UTC+6

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
