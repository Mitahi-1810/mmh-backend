"""Drug-drug interaction lookup from the drug_interactions table."""
from app.database import get_supabase
from app.models import InteractionCard


async def check_interactions(generic_names: list[str]) -> list[InteractionCard]:
    """
    For every pair in generic_names, query drug_interactions.
    Returns a list of InteractionCard (may be empty if no interactions found).
    """
    if len(generic_names) < 2:
        return []

    sb = get_supabase()
    cards: list[InteractionCard] = []

    for i in range(len(generic_names)):
        for j in range(i + 1, len(generic_names)):
            a, b = generic_names[i].lower(), generic_names[j].lower()
            result = (
                sb.table("drug_interactions")
                .select("*")
                .or_(
                    f"and(drug_a.ilike.%{a}%,drug_b.ilike.%{b}%),"
                    f"and(drug_a.ilike.%{b}%,drug_b.ilike.%{a}%)"
                )
                .execute()
            )
            for row in result.data or []:
                cards.append(
                    InteractionCard(
                        drug_a=row["drug_a"],
                        drug_b=row["drug_b"],
                        severity=row["severity"],
                        description=row["description"],
                    )
                )

    return cards
