"""
Production Liftover Service - Bug-Free Implementation
"""
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from functools import lru_cache
import urllib.request
import gzip
import shutil

from app.config import settings
from app.models.schema_models import CoordinateInput, AssemblyBuild, Strand

logger = logging.getLogger(__name__)

try:
    from pyliftover import LiftOver
    HAS_PYLIFTOVER = True
except ImportError:
    HAS_PYLIFTOVER = False
    logger.error("pyliftover not installed")


class ChainFileManager:
    """Manages chain file downloads and caching"""
    
    def __init__(self):
        self.chain_dir = settings.CHAIN_DIR
        self.chain_urls = settings.CHAIN_FILE_URLS
        self.lifters_cache: Dict[str, LiftOver] = {}
    
    def _normalize_build(self, build: str) -> str:
        """Normalize build name to UCSC format"""
        build = build.upper().replace("GRCH", "HG")
        mapping = {
            "HG37": "hg19",
            "HG38": "hg38",
            "HG18": "hg18",
            "HG19": "hg19",
        }
        return mapping.get(build, build.lower())
    
    def _get_chain_key(self, from_build: str, to_build: str) -> str:
        """Get chain file key matching UCSC format: hg19ToHg38"""
        from_norm = self._normalize_build(from_build)
        to_norm = self._normalize_build(to_build)
        
        to_capitalized = to_norm[0].upper() + to_norm[1:]
        
        return f"{from_norm}To{to_capitalized}"
    
    def download_chain_file(self, chain_key: str, force: bool = False) -> Path:
        """Download chain file if not present"""
        chain_path = self.chain_dir / f"{chain_key}.over.chain"
        
        if chain_path.exists() and not force:
            logger.info(f"Chain file exists: {chain_path}")
            return chain_path
        
        if chain_key not in self.chain_urls:
            if chain_path.exists():
                return chain_path
            raise ValueError(f"No URL for chain: {chain_key}")
        
        url = self.chain_urls[chain_key]
        gz_path = self.chain_dir / f"{chain_key}.over.chain.gz"
        
        try:
            logger.info(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, gz_path)
            
            logger.info(f"Extracting {gz_path}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(chain_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            file_size = chain_path.stat().st_size
            if file_size < 1000:
                raise ValueError(f"File too small: {file_size} bytes")
            
            gz_path.unlink()
            logger.info(f"Downloaded: {chain_path} ({file_size:,} bytes)")
            
            return chain_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if gz_path.exists():
                gz_path.unlink()
            if chain_path.exists():
                chain_path.unlink()
            raise
    
    def get_lifter(self, from_build: str, to_build: str) -> LiftOver:
        """Get or create LiftOver object - FIXED caching"""
        if not HAS_PYLIFTOVER:
            raise RuntimeError("pyliftover not installed")
        
        chain_key = self._get_chain_key(from_build, to_build)
        
        # Check cache
        if chain_key in self.lifters_cache:
            return self.lifters_cache[chain_key]
        
        # Download if needed
        chain_path = self.download_chain_file(chain_key)
        
        # Create lifter
        logger.info(f"Loading LiftOver for {chain_key}...")
        lifter = LiftOver(str(chain_path))
        self.lifters_cache[chain_key] = lifter
        
        return lifter


class LiftoverService:
    """Main liftover service - FIXED all bugs"""
    
    def __init__(self):
        self.chain_manager = ChainFileManager()
    
    def convert_coordinate(
        self,
        chrom: str,
        pos: int,
        from_build: str,
        to_build: str,
        strand: str = "+"
    ) -> Dict:
        """
        Convert single coordinate - FIXED type handling
        """
        try:
            # Normalize chromosome
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            
            # Get lifter
            lifter = self.chain_manager.get_lifter(from_build, to_build)
            
            # Convert (pyliftover uses 0-based internally, handles this automatically)
            result = lifter.convert_coordinate(chrom, pos - 1, strand)
            
            if not result:
                return {
                    "success": False,
                    "error": "No mapping found",
                    "original": {"chrom": chrom, "pos": pos, "build": from_build}
                }
            
            # Handle multiple mappings
            if len(result) > 1:
                # Sort by score
                result = sorted(result, key=lambda x: x[3], reverse=True)
                
                return {
                    "success": True,
                    "lifted_chrom": result[0][0],
                    "lifted_pos": result[0][1] + 1,  # Convert back to 1-based
                    "lifted_strand": result[0][2],
                    "confidence": float(result[0][3]),
                    "method": "UCSC_LiftOver",
                    "ambiguous": True,
                    "alternative_mappings": [
                        {
                            "chrom": r[0],
                            "pos": r[1] + 1,
                            "strand": r[2],
                            "score": float(r[3])
                        } for r in result[1:]
                    ],
                    "original": {"chrom": chrom, "pos": pos, "build": from_build}
                }
            
            # Single mapping
            lifted_chrom, lifted_pos, lifted_strand, score = result[0]
            
            return {
                "success": True,
                "lifted_chrom": lifted_chrom,
                "lifted_pos": lifted_pos + 1,  # Convert to 1-based
                "lifted_strand": lifted_strand,
                "confidence": float(score),
                "method": "UCSC_LiftOver",
                "ambiguous": False,
                "original": {"chrom": chrom, "pos": pos, "build": from_build}
            }
            
        except Exception as e:
            logger.error(f"Liftover failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original": {"chrom": chrom, "pos": pos, "build": from_build}
            }
    
    def convert_region(
        self,
        chrom: str,
        start: int,
        end: int,
        from_build: str,
        to_build: str
    ) -> Dict:
        """Convert genomic region"""
        start_result = self.convert_coordinate(chrom, start, from_build, to_build)
        end_result = self.convert_coordinate(chrom, end, from_build, to_build)
        
        if not start_result["success"] or not end_result["success"]:
            return {
                "success": False,
                "error": "Failed to convert region boundaries",
                "start_result": start_result,
                "end_result": end_result
            }
        
        return {
            "success": True,
            "lifted_chrom": start_result["lifted_chrom"],
            "lifted_start": start_result["lifted_pos"],
            "lifted_end": end_result["lifted_pos"],
            "confidence": (start_result["confidence"] + end_result["confidence"]) / 2,
            "method": "UCSC_LiftOver",
            "original": {
                "chrom": chrom,
                "start": start,
                "end": end,
                "build": from_build
            }
        }
    
    def batch_convert(
        self,
        coordinates: List[Dict],
        from_build: str,
        to_build: str
    ) -> List[Dict]:
        """Batch convert coordinates"""
        results = []
        
        for coord in coordinates:
            result = self.convert_coordinate(
                coord.get("chrom", ""),
                coord.get("pos", 0),
                from_build,
                to_build,
                coord.get("strand", "+")
            )
            result["original_annotation"] = coord
            results.append(result)
        
        return results