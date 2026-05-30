"""
pgvector semantic drug name search.
Falls back to ilike fuzzy match when embedding call fails.
"""
import json
from app.database import get_supabase
from app.models import MedicineRow


async def search_medicines(query: str, limit: int = 10) -> list[MedicineRow]:
    """
    1. Embed the query via Supabase edge function (calls Gemini embeddings).
    2. Run pgvector cosine similarity search on medicines.name_embedding.
    3. Return top `limit` matches as MedicineRow objects.
    """
    sb = get_supabase()

    try:
        result = sb.rpc(
            "match_medicines",
            {"query_text": query, "match_count": limit},
        ).execute()
        rows = result.data or []
    except Exception:
        # Fallback: plain text search
        result = (
            sb.table("medicines")
            .select("*")
            .ilike("brand_name", f"%{query}%")
            .limit(limit)
            .execute()
        )
        rows = result.data or []

    return [MedicineRow(**r) for r in rows]


async def get_cheapest_generics(generic_name: str, limit: int = 10) -> list[MedicineRow]:
    """Return all brands for a generic, sorted cheapest first."""
    sb = get_supabase()
    result = (
        sb.table("medicines")
        .select("*")
        .ilike("generic_name", f"%{generic_name}%")
        .order("price_per_unit", desc=False)
        .limit(limit)
        .execute()
    )
    return [MedicineRow(**r) for r in (result.data or [])]
