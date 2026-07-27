from __future__ import annotations

from typing import Any

import requests

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"


def match_species(scientific_name: str, timeout: int = 8) -> dict[str, Any]:
    """Match a scientific name against the GBIF taxonomic backbone.

    Returns a compact dictionary and never raises a network exception, allowing
    the dashboard to fall back to its bundled reference data when offline.
    """
    if not scientific_name:
        return {"ok": False, "error": "No scientific name is available."}

    try:
        response = requests.get(
            GBIF_MATCH_URL,
            params={"name": scientific_name, "kingdom": "Animalia"},
            timeout=timeout,
            headers={"User-Agent": "Ocean-Intelligence-Dashboard/1.0"},
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": bool(data.get("usageKey") or data.get("speciesKey")),
        "match_type": data.get("matchType", "Unknown"),
        "confidence": data.get("confidence"),
        "scientific_name": data.get("scientificName") or scientific_name,
        "canonical_name": data.get("canonicalName"),
        "status": data.get("status"),
        "rank": data.get("rank"),
        "kingdom": data.get("kingdom"),
        "phylum": data.get("phylum"),
        "class": data.get("class"),
        "order": data.get("order"),
        "family": data.get("family"),
        "genus": data.get("genus"),
        "species": data.get("species"),
        "taxon_key": data.get("usageKey") or data.get("speciesKey"),
    }
