"""
Ensembl REST-based liftover wrapper with fallback to chain-file liftover.
Primary approach: use Ensembl REST `map` endpoint:
GET {base}/map/human/{from_assembly}/{region}/{to_assembly}
Example region: 7:140424943..140624564
"""
from typing import Dict, Optional
import requests
import time
from app.config import settings
import logging

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": settings.ENSEMBL_USER_AGENT
}


class EnsemblLiftover:
    def __init__(self, fallback=None):
        """
        fallback: object with convert_coordinate(chrom, pos, from_build, to_build) -> dict
        """
        self.base = settings.ENSEMBL_REST_BASE
        self.grch37_base = settings.ENSEMBL_REST_GRCH37_BASE
        self.timeout = settings.ENSEMBL_REQUEST_TIMEOUT
        self.fallback = fallback

    def _map_endpoint(self, from_build: str, chrom: str, pos: int, to_build: str) -> str:
        """
        Compose Ensembl map endpoint for a single base coordinate.
        We use a tiny window (pos..pos) which Ensembl accepts.
        """
        region = f"{chrom}:{pos}..{pos}"
        # If mapping from GRCh37 use the grch37 host
        base = self.grch37_base if "37" in from_build else self.base
        return f"{base}/map/human/{from_build}/{region}/{to_build}"

    def convert_coordinate(self, chrom: str, pos: int, from_build: str, to_build: str, strand: Optional[str] = "+"):
        """
        Attempt to convert via Ensembl REST. On error or empty result, call fallback.
        Returns a dictionary:
            {
              "success": True/False,
              "lifted_chrom": "chr17",
              "lifted_pos": 43044295,
              "confidence": 1.0,
              "ambiguous": False,
              "method": "ensembl" or "chain"
            }
        """
        url = self._map_endpoint(from_build, chrom, pos, to_build)
        headers = DEFAULT_HEADERS.copy()
        last_exc = None

        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=self.timeout)
                if r.status_code == 404:
                    # No mapping found via Ensembl
                    logger.debug("Ensembl returned 404 for %s", url)
                    break
                r.raise_for_status()
                data = r.json()
                # Example response shape includes "mappings": [...]
                mappings = data.get("mappings") or data.get("mapped") or []
                if not mappings:
                    logger.info("Ensembl returned no mappings for %s:%d (%s->%s)", chrom, pos, from_build, to_build)
                    break

                # Keep simple: take first mapping as primary; if more than one, mark ambiguous
                first = mappings[0]
                lifted_chrom = first.get("seq_region_name") or first.get("mapped_region", {}).get("seq_region_name")
                lifted_start = first.get("start") or first.get("mapped_region", {}).get("start")
                lifted_end = first.get("end") or first.get("mapped_region", {}).get("end")
                lifted_pos = int(lifted_start) if lifted_start else None

                ambiguous = len(mappings) > 1
                confidence = 1.0  # Ensembl doesn't provide explicit confidence; set to 1.0

                return {
                    "success": True if lifted_pos else False,
                    "lifted_chrom": f"chr{lifted_chrom}" if lifted_chrom and not str(lifted_chrom).startswith("chr") else lifted_chrom,
                    "lifted_pos": lifted_pos,
                    "confidence": confidence,
                    "ambiguous": ambiguous,
                    "method": "ensembl",
                    "raw": data
                }
            except Exception as e:
                last_exc = e
                logger.warning("Ensembl liftover attempt %d failed: %s", attempt + 1, e)
                time.sleep(1 + attempt * 2)

        # Fall back
        logger.info("Falling back to chain-file liftover for %s:%d", chrom, pos)
        if self.fallback:
            return self.fallback.convert_coordinate(chrom, pos, from_build, to_build)
        else:
            return {"success": False, "error": str(last_exc or "No mapping"), "method": "ensembl_fallback_failed"}