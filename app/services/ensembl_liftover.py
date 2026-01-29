"""
Ensembl REST API Liftover Service - FIXED
Handles assembly name conversion correctly (hg19 → GRCh37)
"""
import logging
import time
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class EnsemblLiftover:
    """
    Liftover using Ensembl REST API with proper assembly name handling.
    Falls back to chain-based liftover on failure.
    """
    
    def __init__(self, fallback=None, max_retries: int = 3, timeout: int = 10):
        self.base_url = "https://rest.ensembl.org"
        self.fallback = fallback
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Assembly name mapping: UCSC → Ensembl
        self.assembly_map = {
            'hg19': 'GRCh37',
            'hg38': 'GRCh38',
            'GRCh37': 'GRCh37',
            'GRCh38': 'GRCh38',
            'hg37': 'GRCh37',
        }
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _normalize_assembly(self, assembly: str) -> str:
        """Convert UCSC names (hg19) to Ensembl names (GRCh37)"""
        assembly_upper = assembly.upper().replace("HG", "hg")
        return self.assembly_map.get(assembly_upper, assembly)
    
    def _normalize_chrom(self, chrom: str) -> str:
        """Remove 'chr' prefix for Ensembl API"""
        return chrom.replace('chr', '')
    
    def convert_coordinate(
        self,
        chrom: str,
        pos: int,
        from_build: str = "GRCh37",
        to_build: str = "GRCh38"
    ) -> Dict:
        """
        Convert coordinate using Ensembl REST API.
        Falls back to chain-based liftover on failure.
        
        Args:
            chrom: Chromosome (e.g., "chr17" or "17")
            pos: Position (1-based)
            from_build: Source assembly (supports hg19, hg38, GRCh37, GRCh38)
            to_build: Target assembly
        
        Returns:
            Dict with conversion results
        """
        # Normalize assembly names
        from_assembly = self._normalize_assembly(from_build)
        to_assembly = self._normalize_assembly(to_build)
        
        # Normalize chromosome
        chrom_clean = self._normalize_chrom(chrom)
        
        # Build Ensembl API URL
        # Format: /map/human/{from_assembly}/{region}/{to_assembly}
        region = f"{chrom_clean}:{pos}..{pos}"
        url = f"{self.base_url}/map/human/{from_assembly}/{region}/{to_assembly}"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Try Ensembl API with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Parse Ensembl response
                if data and 'mappings' in data and len(data['mappings']) > 0:
                    mapping = data['mappings'][0]
                    mapped = mapping['mapped']
                    
                    # Extract coordinates
                    mapped_chrom = f"chr{mapped['seq_region_name']}"
                    mapped_pos = mapped['start']
                    mapped_strand = mapped.get('strand', 1)
                    strand_str = '+' if mapped_strand == 1 else '-'
                    
                    return {
                        "success": True,
                        "lifted_chrom": mapped_chrom,
                        "lifted_pos": mapped_pos,
                        "lifted_strand": strand_str,
                        "confidence": 0.95,  # Ensembl is highly reliable
                        "method": "Ensembl_REST_API",
                        "original": {
                            "chrom": chrom,
                            "pos": pos,
                            "build": from_build
                        }
                    }
                else:
                    logger.warning(f"No mapping found in Ensembl response for {chrom}:{pos}")
                    break  # No mapping, try fallback
                    
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Ensembl liftover attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except Exception as e:
                logger.error(f"Ensembl API error: {e}")
                break
        
        # Fall back to chain-based liftover
        if self.fallback:
            logger.info(f"Falling back to chain-file liftover for {chrom}:{pos}")
            try:
                return self.fallback.convert_coordinate(
                    chrom, pos, from_build, to_build
                )
            except Exception as e:
                logger.error(f"Fallback liftover also failed: {e}")
        
        # Complete failure
        return {
            "success": False,
            "error": "Both Ensembl API and chain-based liftover failed",
            "original": {
                "chrom": chrom,
                "pos": pos,
                "build": from_build
            }
        }
    
    def convert_region(
        self,
        chrom: str,
        start: int,
        end: int,
        from_build: str = "GRCh37",
        to_build: str = "GRCh38"
    ) -> Dict:
        """Convert a genomic region"""
        
        # Normalize names
        from_assembly = self._normalize_assembly(from_build)
        to_assembly = self._normalize_assembly(to_build)
        chrom_clean = self._normalize_chrom(chrom)
        
        region = f"{chrom_clean}:{start}..{end}"
        url = f"{self.base_url}/map/human/{from_assembly}/{region}/{to_assembly}"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data and 'mappings' in data and len(data['mappings']) > 0:
                mapping = data['mappings'][0]
                mapped = mapping['mapped']
                
                return {
                    "success": True,
                    "lifted_chrom": f"chr{mapped['seq_region_name']}",
                    "lifted_start": mapped['start'],
                    "lifted_end": mapped['end'],
                    "confidence": 0.95,
                    "method": "Ensembl_REST_API",
                    "original": {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "build": from_build
                    }
                }
            
        except Exception as e:
            logger.error(f"Region liftover failed: {e}")
        
        # Fallback
        if self.fallback:
            return self.fallback.convert_region(chrom, start, end, from_build, to_build)
        
        return {
            "success": False,
            "error": "Region liftover failed",
            "original": {"chrom": chrom, "start": start, "end": end, "build": from_build}
        }