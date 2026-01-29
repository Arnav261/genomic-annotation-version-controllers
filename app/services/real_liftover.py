"""
Real UCSC LiftOver implementation with chain file support
This actually works with real genomic coordinates!
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path as Pathlib
import urllib.request
import gzip
import shutil

logger = logging.getLogger(__name__)

# Try to import pyliftover
try:
    from pyliftover import LiftOver
    HAS_PYLIFTOVER = True
except ImportError:
    HAS_PYLIFTOVER = False
    logger.warning("pyliftover not installed - liftover will be limited")

class RealLiftoverService:
    """
    Production-ready liftover service using UCSC chain files.
    Downloads chain files automatically if needed.
    """
    
    def __init__(self, chain_dir: str = "./data/chains"):
        self.chain_dir = Pathlib(chain_dir)
        self.chain_dir.mkdir(parents=True, exist_ok=True)
        
        self.chain_urls = {
    "hg19ToHg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
    "hg38ToHg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz",
}

        
        if not HAS_PYLIFTOVER:
            logger.error("pyliftover not installed! Install with: pip install pyliftover")
    
    def _normalize_build_name(self, build: str) -> str:
        """Convert various build names to UCSC format"""
        build = build.upper().replace("GRCH", "HG")
        mapping = {
            "HG37": "hg19",
            "HG38": "hg38",
            "HG19": "hg19",
        }
        return mapping.get(build, build.lower())
    
    def _get_chain_filename(self, from_build: str, to_build: str) -> str:
        """Get chain filename for build pair"""
        from_norm = self._normalize_build_name(from_build)
        to_norm = self._normalize_build_name(to_build)
        
        # Capitalize first letter of each part
        from_cap = from_norm[0:2] + from_norm[2:].capitalize()
        to_cap = to_norm[0:2] + to_norm[2:].capitalize()
        
        return f"{from_cap}To{to_cap}"
    
    def _download_chain_file(self, chain_key: str) -> Pathlib:
        """Download chain file from UCSC if not present"""
        if chain_key not in self.chain_urls:
            # Try to see if local file exists in chain_dir
            local_chain = self.chain_dir / f"{chain_key}.over.chain"
            if local_chain.exists():
                logger.info(f"Using local chain file: {local_chain}")
                return local_chain
            
            else:
                raise ValueError(f"No URL configured for chain: {chain_key} and no local file found at {local_chain}")
        
        url = self.chain_urls[chain_key]
        gz_path = self.chain_dir / f"{chain_key}.over.chain.gz"
        chain_path = self.chain_dir / f"{chain_key}.over.chain"
        
        # Return if already downloaded and extracted
        if chain_path.exists():
            logger.info(f"Chain file already exists: {chain_path}")
            return chain_path
        
        try:
            logger.info(f"Downloading chain file from {url}...")
            urllib.request.urlretrieve(url, gz_path)
            
            logger.info(f"Extracting {gz_path}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(chain_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove gz file to save space
            gz_path.unlink()
            
            logger.info(f"Successfully downloaded and extracted: {chain_path}")
            return chain_path
            
        except Exception as e:
            logger.error(f"Failed to download chain file: {e}")
            raise
    
    def _get_lifter(self, from_build: str, to_build: str) -> LiftOver:
        """Get or create LiftOver object for build pair"""
        if not HAS_PYLIFTOVER:
            raise RuntimeError("pyliftover not installed")
        
        chain_key = self._get_chain_filename(from_build, to_build)
        
        # Return cached lifter if exists
        if chain_key in self.lifters:
            return self.lifters[chain_key]
        
        # Download chain file if needed
        chain_path = self._download_chain_file(chain_key)
        
        # Create and cache lifter
        logger.info(f"Loading LiftOver for {chain_key}...")
        lifter = LiftOver(str(chain_path))
        self.lifters[chain_key] = lifter
        
        return lifter
    
    def convert_coordinate(
        self, 
        chrom: str, 
        pos: int, 
        from_build: str = "GRCh37", 
        to_build: str = "GRCh38",
        strand: str = "+"
    ) -> Dict:
        """
        Convert a single coordinate using UCSC LiftOver.
        
        Args:
            chrom: Chromosome (e.g., "chr17" or "17")
            pos: Position (1-based)
            from_build: Source assembly (GRCh37, GRCh38, hg19, hg38)
            to_build: Target assembly
            strand: Strand ("+", "-", or None)
        
        Returns:
            Dict with conversion results including:
            - success: bool
            - lifted_chrom: str
            - lifted_pos: int
            - lifted_strand: str
            - confidence: float (based on chain score)
        """
        try:
            # Normalize chromosome name
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            
            # Get lifter
            lifter = self._get_lifter(from_build, to_build)
            
            # Convert coordinate (pyliftover uses 0-based, so subtract 1)
            result = lifter.convert_coordinate(chrom, pos - 1, strand)
            
            if not result:
                return {
                    "success": False,
                    "error": "No mapping found",
                    "original": {"chrom": chrom, "pos": pos, "build": from_build}
                }
            
            # Handle multiple mappings (usually means ambiguous)
            if len(result) > 1:
                # Return the one with highest score
                result = sorted(result, key=lambda x: x[3], reverse=True)
                
                return {
                    "success": True,
                    "lifted_chrom": result[0][0],
                    "lifted_pos": result[0][1] + 1,  # Convert back to 1-based
                    "lifted_strand": result[0][2],
                    "confidence": result[0][3],
                    "method": "UCSC_LiftOver",
                    "ambiguous": True,
                    "alternative_mappings": [
                        {
                            "chrom": r[0],
                            "pos": r[1] + 1,
                            "strand": r[2],
                            "score": r[3]
                        } for r in result[1:]
                    ],
                    "original": {"chrom": chrom, "pos": pos, "build": from_build}
                }
            
            # Single unambiguous mapping
            lifted_chrom, lifted_pos, lifted_strand, score = result[0]
            
            return {
                "success": True,
                "lifted_chrom": lifted_chrom,
                "lifted_pos": lifted_pos + 1,  # Convert to 1-based
                "lifted_strand": lifted_strand,
                "confidence": score,
                'confidence': float(max(0.0, min(1.0, score if score is not None else 0.0))),
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
        from_build: str = "GRCh37",
        to_build: str = "GRCh38"
    ) -> Dict:
        """Convert a genomic region (start and end coordinates)"""
        
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
        from_build: str = "GRCh37",
        to_build: str = "GRCh38"
    ) -> List[Dict]:
        """
        Convert multiple coordinates efficiently.
        
        Args:
            coordinates: List of dicts with 'chrom' and 'pos' keys
            from_build: Source assembly
            to_build: Target assembly
        
        Returns:
            List of conversion results
        """
        results = []
        
        for coord in coordinates:
            result = self.convert_coordinate(
                coord.get("chrom", coord.get("chr", "")),
                coord.get("pos", coord.get("start", 0)),
                from_build,
                to_build,
                coord.get("strand", "+")
            )
            
            # Add original annotation info
            result["original_annotation"] = coord
            results.append(result)
        
        return results