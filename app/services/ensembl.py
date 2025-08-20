
import time
from typing import Any, Dict, Optional

import requests

from ..core.config import settings



def _release_int(release_tag: str | int) -> int:
    """
    Accepts 'v109' or 109 and returns 109 as int.
    """
    if isinstance(release_tag, int):
        return release_tag
    s = str(release_tag).lower().strip()
    if s.startswith("v"):
        s = s[1:]
    return int(s)

def _headers(release: Optional[int] = None) -> Dict[str, str]:
    """
    Ensembl REST accepts X-Ens-Release to pin a release on the main server.
    We also set a user-agent and JSON content-type.
    """
    h = {
        "Content-Type": "application/json",
        "User-Agent": settings.ENSEMBL_USER_AGENT or "GAVC/1.0 (FastAPI)"
    }
    if release is not None:
        h["X-Ens-Release"] = str(release)
    return h

def _base_url_for_release(release_int: int, force_grch37: bool = False) -> str:
    """
    Return which REST base to use.
    - For GRCh37 coordinate queries Ensembl uses a separate GRCh37 REST host.
    - For general lookups with release pinning, the main REST host works.
    We expose a 'force_grch37' toggle for callers that know they need GRCh37.
    """
    if force_grch37:
        return settings.ENSEMBL_REST_GRCH37_BASE
    return settings.ENSEMBL_REST_BASE

def _http_get_json(url: str, headers: Dict[str, str], params: Optional[dict] = None,
                   max_retries: int = 3, retry_delay: int = 5) -> dict:
    """
    Lightweight GET with retries that returns JSON or raises for errors.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=settings.ENSEMBL_REQUEST_TIMEOUT)
            if r.status_code == 404:
                return {"not_found": True}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e
    raise last_exc  

def lookup_gene_by_symbol(symbol: str, release_tag: str | int) -> dict | None:
    """
    Version-pinned lookup by gene symbol.
    GET /lookup/symbol/homo_sapiens/{symbol}?expand=1
    Uses X-Ens-Release header to pin to a specific Ensembl release.
    """
    release = _release_int(release_tag)
    base = _base_url_for_release(release)
    url = f"{base}/lookup/symbol/homo_sapiens/{symbol}"
    data = _http_get_json(url, headers=_headers(release), params={"expand": 1})
    if data.get("not_found"):
        return None
    return data

def lookup_gene_by_id(stable_id: str, release_tag: str | int) -> dict | None:
    """
    Version-pinned lookup by Ensembl stable ID.
    GET /lookup/id/{id}?expand=1
    """
    release = _release_int(release_tag)
    base = _base_url_for_release(release)
    url = f"{base}/lookup/id/{stable_id}"
    data = _http_get_json(url, headers=_headers(release), params={"expand": 1})
    if data.get("not_found"):
        return None
    return data

def summarize_gene(json_obj: dict, release_tag: str | int) -> dict:
    """
    Create a compact summary used by /compare-annotations response.
    """
    transcripts = []
    if isinstance(json_obj, dict) and isinstance(json_obj.get("Transcript"), list):
        transcripts = json_obj["Transcript"]

    return {
        "ensembl_release": _release_int(release_tag),
        "gene_id": json_obj.get("id"),
        "display_name": json_obj.get("display_name") or json_obj.get("external_name"),
        "seq_region_name": json_obj.get("seq_region_name"),
        "start": json_obj.get("start"),
        "end": json_obj.get("end"),
        "strand": json_obj.get("strand"),
        "transcript_count": len(transcripts),
    }

def map_region_between_assemblies(
    chrom: str, start: int, end: int, strand: int,
    source_assembly: str, target_assembly: str
) -> dict:
    """
    Ensembl REST:
      /map/human/{source_assembly}/{region}/{target_assembly}
    where region is {chrom}:{start}..{end}:{strand}
    """
    chrom_norm = chrom.replace("chr", "")
    region = f"{chrom_norm}:{start}..{end}:{strand}"

    base = settings.ENSEMBL_REST_BASE
    url = f"{base}/map/human/{source_assembly}/{region}/{target_assembly}"
    return _http_get_json(url, headers=_headers())

def get_gene_annotation_versioned(gene: str, release_tag: str | int) -> dict | None:
    """
    Tries symbol lookup first; if that fails, tries ID lookup.
    Returns the full expanded JSON object or None if not found.
    """
    data = lookup_gene_by_symbol(gene, release_tag)
    if data is None:
        data = lookup_gene_by_id(gene, release_tag)
    return data

def parse_gene_annotation(json_response: dict) -> dict:
    """
    Extract Mistral-style parsed summary for CLI-like comparisons.
    """
    parsed = {
        "transcript_count": len(json_response.get("Transcript", [])),
        "coordinates": {
            "chromosome": json_response.get("seq_region_name"),
            "start": json_response.get("start"),
            "end": json_response.get("end"),
            "strand": json_response.get("strand"),
        },
        "gene_type": json_response.get("biotype"),
        "transcript_details": [],
    }
    for tx in json_response.get("Transcript", []):
        parsed["transcript_details"].append({
            "transcript_id": tx.get("id"),
            "biotype": tx.get("biotype"),
            "exon_count": len(tx.get("Exon", []))
        })
    return parsed

def compare_gene_annotations(a: dict, b: dict) -> dict:
    """
    Simple, CLI-style comparison of two parsed gene objects (Mistral approach).
    """
    out = {
        "transcript_count_change": (a["transcript_count"], b["transcript_count"]),
        "coordinate_changes": {
            "chromosome_changed": a["coordinates"]["chromosome"] != b["coordinates"]["chromosome"],
            "start_changed": a["coordinates"]["start"] != b["coordinates"]["start"],
            "end_changed": a["coordinates"]["end"] != b["coordinates"]["end"],
            "strand_changed": a["coordinates"]["strand"] != b["coordinates"]["strand"],
        },
        "gene_type_change": a.get("gene_type") != b.get("gene_type"),
        "transcript_details": {
            "added": [],
            "removed": [],
            "changes": [],
        },
    }

    ids_a = {t["transcript_id"] for t in a["transcript_details"]}
    ids_b = {t["transcript_id"] for t in b["transcript_details"]}

    out["transcript_details"]["added"] = sorted(list(ids_b - ids_a))
    out["transcript_details"]["removed"] = sorted(list(ids_a - ids_b))

    for t in a["transcript_details"]:
        tid = t["transcript_id"]
        if tid in ids_b:
            t2 = next(x for x in b["transcript_details"] if x["transcript_id"] == tid)
            out["transcript_details"]["changes"].append({
                "transcript_id": tid,
                "biotype_change": (t.get("biotype") != t2.get("biotype")),
                "exon_count_change": (t.get("exon_count") != t2.get("exon_count")),
            })

    return out
