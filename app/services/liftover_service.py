
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


import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import shutil

# Try pyliftover
try:
    from pyliftover import LiftOver
    HAS_PYLIFTOVER = True
except Exception:
    HAS_PYLIFTOVER = False

class LiftoverResult(BaseModel):
    chrom: str
    pos: int
    lift_chrom: Optional[str] = None
    lift_pos: Optional[int] = None
    status: str = "not_mapped"
    notes: Dict[str, Any] = {}

class LiftoverManager:
    def __init__(self, chain_dir: str = "./data/chains"):
        self.chain_dir = chain_dir
        self._lifter_cache = {}
        self._ucsc_path = shutil.which("liftOver")

    def _chain_filename_candidates(self, build_from: str, build_to: str):
        base = f"{build_from}To{build_to}.over.chain"
        return [base, base + ".gz"]

    def _get_chain_path(self, build_from: str, build_to: str):
        for cand in self._chain_filename_candidates(build_from, build_to):
            p = os.path.join(self.chain_dir, cand)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Missing chain file for {build_from} -> {build_to}. Expected files like {build_from}To{build_to}.over.chain(.gz) under {self.chain_dir}")

    def _load_pylift(self, build_from: str, build_to: str):
        if not HAS_PYLIFTOVER:
            return None
        key = f"{build_from}To{build_to}"
        if key in self._lifter_cache and isinstance(self._lifter_cache[key], LiftOver):
            return self._lifter_cache[key]
        chain_path = self._get_chain_path(build_from, build_to)
        lifter = LiftOver(chain_path)
        self._lifter_cache[key] = lifter
        return lifter

    def _call_ucsc_liftOver(self, chain_path: str, chrom: str, pos: int, strand: str = "+") -> List[Dict[str, Any]]:
        if not self._ucsc_path:
            raise FileNotFoundError("UCSC liftOver binary not found in PATH")
        bed_line = f"{chrom}\t{pos-1}\t{pos}\n"
        with tempfile.TemporaryDirectory() as td:
            in_bed = os.path.join(td, "in.bed")
            out_bed = os.path.join(td, "out.bed")
            unmap_bed = os.path.join(td, "unmap.bed")
            with open(in_bed, "w") as f:
                f.write(bed_line)
            cmd = [self._ucsc_path, in_bed, chain_path, out_bed, unmap_bed]
            try:
                subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                pass
            results = []
            if os.path.exists(out_bed):
                with open(out_bed) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            r_chrom, r_start, r_end = parts[0], int(parts[1]), int(parts[2])
                            results.append({"chrom": r_chrom, "pos": r_start + 1})
            return results

    def liftover_single(self, chrom: str, pos: int, build_from: str, build_to: str, strand: str = "+", fallback_to_ucsc: bool = False) -> LiftoverResult:
        try:
            lifter = self._load_pylift(build_from, build_to)
            if lifter:
                converted = lifter.convert_coordinate(chrom, pos - 1, strand)
                if converted:
                    if len(converted) == 1:
                        new_chrom, new_pos0, new_strand, frac = converted[0]
                        return LiftoverResult(chrom=chrom, pos=pos, lift_chrom=new_chrom, lift_pos=new_pos0 + 1, status="mapped", notes={"fractionMapped": frac, "mapped_strand": new_strand})
                    else:
                        candidates = [{"chrom": c[0], "pos": c[1] + 1, "strand": c[2], "fractionMapped": c[3]} for c in converted]
                        return LiftoverResult(chrom=chrom, pos=pos, status="ambiguous", notes={"candidates": candidates})
        except FileNotFoundError:
            raise
        except Exception:
            pass

        chain_path = self._get_chain_path(build_from, build_to)
        if self._ucsc_path:
            mappings = self._call_ucsc_liftOver(chain_path, chrom, pos, strand)
            if not mappings:
                return LiftoverResult(chrom=chrom, pos=pos, status="not_mapped")
            elif len(mappings) == 1:
                m = mappings[0]
                return LiftoverResult(chrom=chrom, pos=pos, lift_chrom=m["chrom"], lift_pos=m["pos"], status="mapped")
            else:
                return LiftoverResult(chrom=chrom, pos=pos, status="ambiguous", notes={"candidates": mappings})
        else:
            raise RuntimeError("No viable liftover backend available")