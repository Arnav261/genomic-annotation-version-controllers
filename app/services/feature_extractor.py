"""
Genomic Feature Extractor for Confidence Prediction

This module extracts features that indicate liftover reliability:
- Repetitive element density
- Chain file consensus
- Structural variant overlap
- Assembly gap proximity
- Historical failure rates
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GenomicFeatures:
    """Container for extracted genomic features"""
    
    # Coordinate features
    chromosome: str
    position: int
    
    # Chain file features
    chain_score: float                    # UCSC chain alignment score
    chain_count: int                      # Number of chain files agreeing
    chain_gap_size: int                   # Size of nearest chain gap
    
    # Sequence context features
    gc_content: float                     # Local GC content (±1kb)
    repeat_density: float                 # RepeatMasker overlap fraction
    repeat_type: str                      # Type of repeat (SINE, LINE, etc.)
    low_complexity: bool                  # Low complexity region flag
    
    # Structural features
    sv_overlap: bool                      # Structural variant database overlap
    segdup_overlap: bool                  # Segmental duplication overlap
    assembly_gap_distance: int            # Distance to nearest assembly gap
    
    # Historical features
    historical_success_rate: float        # Success rate for this region
    cross_reference_agreement: float      # Agreement across databases
    
    # Metadata
    build_from: str
    build_to: str
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model"""
        return np.array([
            self.chain_score,
            float(self.chain_count),
            float(self.chain_gap_size),
            self.gc_content,
            self.repeat_density,
            float(self.low_complexity),
            float(self.sv_overlap),
            float(self.segdup_overlap),
            float(self.assembly_gap_distance),
            self.historical_success_rate,
            self.cross_reference_agreement
        ])

class FeatureExtractor:
    """
    Extract genomic features for ML confidence prediction.
    
    Uses multiple data sources:
    - UCSC chain files
    - RepeatMasker annotations
    - DGV structural variants
    - Assembly gap tracks
    - Historical liftover success rates
    """
    
    def __init__(self, data_dir: str = "./app/data/reference"):
        self.data_dir = data_dir
        self.repeat_cache = {}
        self.sv_cache = {}
        self.historical_stats = {}
        
        logger.info("Initializing FeatureExtractor")
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load reference annotation databases"""
        try:
            # Load pre-computed repeat densities by region
            # Format: {(chr, pos//10000): density}
            self.repeat_cache = self._load_repeat_densities()
            
            # Load structural variant intervals
            # Format: {chr: [(start, end, type)]}
            self.sv_cache = self._load_sv_intervals()
            
            # Load historical success rates
            # Format: {(chr, pos//100000): success_rate}
            self.historical_stats = self._load_historical_stats()
            
            logger.info("Reference data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load all reference data: {e}")
            logger.warning("Using simplified feature extraction")
    
    def _load_repeat_densities(self) -> Dict:
        """
        Load pre-computed RepeatMasker densities.
        
        In production, this would load from:
        - RepeatMasker.out files from UCSC
        - Pre-computed 10kb window densities
        
        For now, returns defaults.
        """
        # TODO: Load actual RepeatMasker data
        # wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/rmsk.txt.gz
        return {}
    
    def _load_sv_intervals(self) -> Dict:
        """
        Load structural variant intervals from DGV/gnomAD-SV.
        
        Format: chromosome -> list of (start, end, type) tuples
        """
        # TODO: Load actual SV databases
        # DGV: http://dgv.tcag.ca/dgv/app/downloads
        # gnomAD-SV: https://gnomad.broadinstitute.org/downloads
        return {}
    
    def _load_historical_stats(self) -> Dict:
        """
        Load historical liftover success rates by genomic region.
        
        This would be built from:
        - Previous successful/failed liftover attempts
        - NCBI Remap logs
        - Community-reported issues
        """
        # TODO: Implement historical tracking
        return {}
    
    def extract_features(
        self,
        chrom: str,
        pos: int,
        build_from: str,
        build_to: str,
        chain_result: Optional[Dict] = None
    ) -> GenomicFeatures:
        """
        Extract all features for a genomic position.
        
        Args:
            chrom: Chromosome (e.g., "chr17")
            pos: Position (1-based)
            build_from: Source assembly
            build_to: Target assembly
            chain_result: Optional result from chain file liftover
            
        Returns:
            GenomicFeatures object with all extracted features
        """
        
        # Chain file features
        chain_score = chain_result.get("confidence", 1.0) if chain_result else 0.0
        chain_count = 1 if chain_result and chain_result.get("success") else 0
        chain_gap_size = self._estimate_chain_gap_size(chrom, pos, build_from)
        
        # Sequence context
        gc_content = self._calculate_gc_content(chrom, pos, build_from)
        repeat_density, repeat_type = self._get_repeat_info(chrom, pos, build_from)
        low_complexity = repeat_density > 0.8
        
        # Structural features
        sv_overlap = self._check_sv_overlap(chrom, pos, build_from)
        segdup_overlap = self._check_segdup_overlap(chrom, pos, build_from)
        gap_distance = self._distance_to_assembly_gap(chrom, pos, build_from)
        
        # Historical features
        historical_success = self._get_historical_success_rate(chrom, pos, build_from, build_to)
        cross_ref_agreement = self._calculate_cross_reference_agreement(chrom, pos, build_from)
        
        return GenomicFeatures(
            chromosome=chrom,
            position=pos,
            chain_score=chain_score,
            chain_count=chain_count,
            chain_gap_size=chain_gap_size,
            gc_content=gc_content,
            repeat_density=repeat_density,
            repeat_type=repeat_type,
            low_complexity=low_complexity,
            sv_overlap=sv_overlap,
            segdup_overlap=segdup_overlap,
            assembly_gap_distance=gap_distance,
            historical_success_rate=historical_success,
            cross_reference_agreement=cross_ref_agreement,
            build_from=build_from,
            build_to=build_to
        )
    
    def _estimate_chain_gap_size(self, chrom: str, pos: int, build: str) -> int:
        """
        Estimate size of nearest gap in chain file alignment.
        
        Large gaps indicate problematic regions for liftover.
        """
        # Simplified: would query actual chain file structure
        # For now, return default
        return 0
    
    def _calculate_gc_content(self, chrom: str, pos: int, build: str, window: int = 1000) -> float:
        """
        Calculate GC content in local window.
        
        Extreme GC content correlates with assembly issues.
        """
        # Would query reference genome FASTA
        # For now, return genome-wide average
        return 0.41
    
    def _get_repeat_info(self, chrom: str, pos: int, build: str) -> Tuple[float, str]:
        """
        Get repeat density and type from RepeatMasker.
        
        High repeat density = less reliable liftover
        """
        # Query pre-loaded repeat cache
        region_key = (chrom, pos // 10000)
        density = self.repeat_cache.get(region_key, 0.0)
        
        # Default repeat type
        repeat_type = "none" if density < 0.1 else "unknown"
        
        return density, repeat_type
    
    def _check_sv_overlap(self, chrom: str, pos: int, build: str) -> bool:
        """
        Check if position overlaps known structural variants.
        
        SVs often have inconsistent coordinates across assemblies.
        """
        if chrom not in self.sv_cache:
            return False
        
        for start, end, sv_type in self.sv_cache[chrom]:
            if start <= pos <= end:
                return True
        
        return False
    
    def _check_segdup_overlap(self, chrom: str, pos: int, build: str) -> bool:
        """
        Check for segmental duplication overlap.
        
        Segmental duplications are difficult to map uniquely.
        """
        # Would query UCSC segmental duplication track
        # High-risk regions: pericentromeric, subtelomeric
        
        # Simple heuristic based on position
        chr_num = chrom.replace("chr", "")
        if chr_num.isdigit():
            chr_length = self._get_chromosome_length(chr_num)
            # Centromeric regions (rough estimate)
            if 0.4 * chr_length <= pos <= 0.6 * chr_length:
                return True
        
        return False
    
    def _distance_to_assembly_gap(self, chrom: str, pos: int, build: str) -> int:
        """
        Calculate distance to nearest assembly gap.
        
        Positions near gaps have uncertain coordinates.
        """
        # Would query UCSC gap track
        # For now, return large distance (no gap nearby)
        return 1000000
    
    def _get_historical_success_rate(
        self,
        chrom: str,
        pos: int,
        build_from: str,
        build_to: str
    ) -> float:
        """
        Get historical liftover success rate for this region.
        
        Based on previous attempts and known issues.
        """
        region_key = (chrom, pos // 100000, f"{build_from}_{build_to}")
        return self.historical_stats.get(region_key, 0.95)  # Default: assume good
    
    def _calculate_cross_reference_agreement(
        self,
        chrom: str,
        pos: int,
        build: str
    ) -> float:
        """
        Calculate agreement across reference databases.
        
        High agreement = more reliable coordinates
        """
        # Would compare NCBI, Ensembl, UCSC coordinates for genes in region
        # For now, return default
        return 0.90
    
    def _get_chromosome_length(self, chr_num: str) -> int:
        """Get approximate chromosome length for hg19/hg38"""
        # Rough estimates for human chromosomes
        lengths = {
            "1": 249000000, "2": 243000000, "3": 198000000,
            "4": 191000000, "5": 180000000, "6": 171000000,
            "7": 159000000, "8": 146000000, "9": 141000000,
            "10": 135000000, "11": 135000000, "12": 133000000,
            "13": 115000000, "14": 107000000, "15": 102000000,
            "16": 90000000, "17": 81000000, "18": 78000000,
            "19": 59000000, "20": 63000000, "21": 48000000,
            "22": 51000000, "X": 155000000, "Y": 59000000
        }
        return lengths.get(chr_num, 100000000)
    
    def batch_extract_features(
        self,
        coordinates: List[Tuple[str, int]],
        build_from: str,
        build_to: str,
        chain_results: Optional[List[Dict]] = None
    ) -> List[GenomicFeatures]:
        """
        Extract features for multiple coordinates efficiently.
        
        Args:
            coordinates: List of (chrom, pos) tuples
            build_from: Source assembly
            build_to: Target assembly
            chain_results: Optional list of chain file results
            
        Returns:
            List of GenomicFeatures objects
        """
        features = []
        
        for i, (chrom, pos) in enumerate(coordinates):
            chain_result = chain_results[i] if chain_results else None
            
            feature = self.extract_features(
                chrom, pos, build_from, build_to, chain_result
            )
            features.append(feature)
        
        return features