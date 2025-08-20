
import os
from typing import Dict, Any, Optional, Tuple

LIFTOVER_MODE = os.getenv("LIFTOVER_MODE", "ucsc").lower()
CHAIN_DIR = os.getenv("CHAIN_DIR", "./data/chains")

def _pick_chain_file(source_build: str, target_build: str) -> Optional[str]:
    """
    Maps build pair -> UCSC chain filename.
    Assumes human builds: GRCh37 ≈ hg19, GRCh38 ≈ hg38.
    """
    key = (source_build.upper(), target_build.upper())
    mapping = {
        ("GRCH37", "GRCH38"): "hg19ToHg38.over.chain.gz",
        ("GRCH38", "GRCH37"): "hg38ToHg19.over.chain.gz",
    }
    fname = mapping.get(key)
    if not fname:
        return None
    return os.path.join(CHAIN_DIR, fname)

def _ensure_chr_prefix(chrom: str) -> str:
    
    chrom = chrom.strip()
    if not chrom.lower().startswith("chr"):
        return "chr" + chrom
    return chrom

def _strip_chr_prefix(chrom: str) -> str:
    
    return chrom[3:] if chrom.lower().startswith("chr") else chrom

def _liftover_ucsc(source_build: str, target_build: str, chrom: str, start: int, end: int) -> Dict[str, Any]:
    try:
        chain_file = _pick_chain_file(source_build, target_build)
        if not chain_file or not os.path.exists(chain_file):
            return {"error": f"Missing chain file for {source_build}->{target_build}", "chain_dir": CHAIN_DIR}

        from pyliftover import LiftOver
        lo = LiftOver(chain_file)

        
        src_chrom = _ensure_chr_prefix(chrom)
        
        mapped_start_list = lo.convert_coordinate(src_chrom, start)
        mapped_end_list = lo.convert_coordinate(src_chrom, end)

        if not mapped_start_list or not mapped_end_list:
            return {"error": "No mapping found for given interval"}

        mstart = mapped_start_list[0]
        mend = mapped_end_list[0]
        tchrom, tstart, tstrand, tscore = mstart[0], int(mstart[1]), mstart[2], float(mstart[3])
        _, tend, _, _ = mend

        return {
            "source": {"build": source_build, "chrom": chrom, "start": start, "end": end},
            "target": {"build": target_build, "chrom": _strip_chr_prefix(tchrom), "start": tstart, "end": int(tend)},
            "tool": "UCSC liftOver (pyliftover)",
            "score": tscore
        }
    except Exception as e:
        return {"error": str(e)}

def _liftover_ensembl(source_build: str, target_build: str, chrom: str, start: int, end: int) -> Dict[str, Any]:
    """
    Placeholder for Ensembl REST 'map' endpoint.

    If your Perplexity doc has a working example, paste it here, something like:
      GET /map/homo_sapiens/{source_assembly}/{region}/{target_assembly}
    where region = '{chrom}:{start}..{end}'

    Then parse the JSON and return the same shape as _liftover_ucsc().
    """
    return {
        "error": "Ensembl REST liftover not configured. Paste your doc's code into _liftover_ensembl().",
        "hint": "Set LIFTOVER_MODE=ucsc (recommended) or provide Ensembl 'map' code here."
    }

def liftover_coords(source_build: str, target_build: str, chrom: str, start: int, end: int) -> Dict[str, Any]:
    """
    The function the router calls. It chooses the tool based on LIFTOVER_MODE.
    """
    mode = LIFTOVER_MODE
    if mode == "ucsc":
        return _liftover_ucsc(source_build, target_build, chrom, start, end)
    elif mode == "ensembl":
        return _liftover_ensembl(source_build, target_build, chrom, start, end)
    else:
        return {"error": f"Unknown LIFTOVER_MODE='{LIFTOVER_MODE}'. Use 'ucsc' or 'ensembl'."}
