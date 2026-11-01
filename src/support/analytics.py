"""
Administrator analytics aggregation strictly derived from stored support records.

Carries no simulated data, no query-intent chat volume claims, and enforces
administrator authentication on the application layer.
"""
from typing import Dict, List, MutableMapping, Tuple

from .store import SupportStore


def require_admin_auth(state: MutableMapping) -> None:
    """Fail closed if the session is not authenticated as an administrator."""
    if not state.get("admin_authenticated"):
        raise PermissionError("Administrator authentication required.")


def get_admin_analytics(store: SupportStore, state: MutableMapping) -> Dict[str, object]:
    """
    Compute operational support analytics from the SQLite support store.

    Requires admin authentication. Unescalated student chat interactions are
    ephemeral and never retained; metrics reflect stored tickets and consented feedback.
    """
    require_admin_auth(state)
    return {
        "ticket_counts": store.ticket_counts(),
        "feedback_metrics": store.feedback_metrics(),
        "sentiment_distribution": store.sentiment_distribution(),
        "creation_trends": store.ticket_creation_trends(),
    }
