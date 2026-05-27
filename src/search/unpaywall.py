"""Unpaywall - given a DOI, returns the open-access PDF URL if one is legally available.

This is the legal way to get paywalled papers: only fetches versions the authors
have legitimately made available (preprints, accepted manuscripts, gold OA).
"""
from __future__ import annotations
import httpx
from .. import config


def find_oa_pdf(doi: str) -> str | None:
    """Return an open-access PDF URL for this DOI, or None."""
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "").strip()
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": config.CONTACT_EMAIL}
    try:
        r = httpx.get(url, params=params, timeout=15.0)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception as e:
        print(f"[unpaywall] error for {doi}: {e}")
        return None

    best = data.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf") or best.get("url")
    return pdf
