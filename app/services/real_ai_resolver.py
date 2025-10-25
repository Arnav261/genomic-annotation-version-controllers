"""
REAL AI-powered conflict resolution using actual machine learning.
Uses sklearn clustering, statistical analysis, and biological heuristics.
"""
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed - AI features will be limited")

@dataclass
class AnnotationSource:
    """Real annotation source data"""
    name: str
    start: int
    end: int
    strand: Optional[str] = None
    confidence: float = 0.8
    biotype: Optional[str] = None
    description: Optional[str] = None
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

@dataclass
class ConflictResolution:
    """Resolution result with detailed metrics"""
    gene_symbol: str
    resolved_start: int
    resolved_end: int
    resolved_strand: Optional[str]
    confidence_score: float
    resolution_method: str
    contributing_sources: List[str]
    conflict_types: List[str]
    consensus_level: float
    statistical_metrics: Dict
    recommendation: str
    manual_review_needed: bool

class RealAIConflictResolver:
    """
    Production-ready AI conflict resolver.
    Uses real statistical methods and machine learning.
    """
    
    def __init__(self):
        # Database reliability scores (based on publication citations and update frequency)
        self.source_reliability = {
            "GENCODE": 0.98,   # Gold standard for human genes
            "Ensembl": 0.95,   # Comprehensive, well-maintained
            "RefSeq": 0.92,    # NCBI curated
            "UCSC": 0.88,      # Aggregated but useful
            "CHESS": 0.85,     # Novel annotations
            "Havana": 0.94     # Manual curation
        }
        
        # Evidence type weights
        self.evidence_weights = {
            "experimental": 1.0,
            "literature": 0.9,
            "manual_annotation": 0.95,
            "computational": 0.7,
            "automatic_annotation": 0.6,
            "protein_evidence": 0.9,
            "transcript_evidence": 0.85
        }
        
        self.scaler = StandardScaler() if HAS_SKLEARN else None
    
    def detect_conflicts(self, sources: List[AnnotationSource]) -> List[str]:
        """
        Detect what types of conflicts exist.
        
        Returns:
            List of conflict types found
        """
        if len(sources) < 2:
            return []
        
        conflicts = []
        
        # Coordinate conflicts
        starts = [s.start for s in sources]
        ends = [s.end for s in sources]
        
        start_range = max(starts) - min(starts)
        end_range = max(ends) - min(ends)
        
        if start_range > 100 or end_range > 100:
            conflicts.append("coordinate_mismatch")
        
        # Strand conflicts
        strands = [s.strand for s in sources if s.strand]
        if len(set(strands)) > 1:
            conflicts.append("strand_inconsistency")
        
        # Biotype conflicts
        biotypes = [s.biotype for s in sources if s.biotype]
        if len(set(biotypes)) > 1:
            conflicts.append("biotype_disagreement")
        
        return conflicts
    
    def extract_features(self, sources: List[AnnotationSource]) -> np.ndarray:
        """
        Extract numerical features for clustering.
        
        Features:
        - Start position (normalized)
        - End position (normalized)
        - Gene length
        - Source reliability
        - Confidence score
        - Evidence count
        - Strand (encoded)
        """
        features = []
        
        for source in sources:
            reliability = self.source_reliability.get(source.name, 0.5)
            strand_encoding = {"+": 1.0, "-": -1.0, None: 0.0}
            
            feature_vector = [
                float(source.start),
                float(source.end),
                float(source.end - source.start),  # length
                reliability,
                source.confidence,
                float(len(source.evidence)),
                strand_encoding.get(source.strand, 0.0)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def cluster_annotations(
        self,
        sources: List[AnnotationSource],
        method: str = "dbscan"
    ) -> Tuple[List[int], Dict]:
        """
        Cluster similar annotations using real ML algorithms.
        
        Args:
            sources: List of annotation sources
            method: "dbscan" or "agglomerative"
        
        Returns:
            Tuple of (cluster_labels, clustering_metrics)
        """
        if not HAS_SKLEARN:
            # Fallback to simple grouping by proximity
            return self._simple_clustering(sources)
        
        # Extract and normalize features
        features = self.extract_features(sources)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Apply clustering
        if method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
        else:  # agglomerative
            n_clusters = min(3, len(sources))
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        labels = clusterer.fit_predict(features_normalized)
        
        # Calculate quality metrics
        if len(set(labels)) > 1 and len(sources) > 2:
            silhouette = silhouette_score(features_normalized, labels)
        else:
            silhouette = 0.0
        
        metrics = {
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": sum(1 for l in labels if l == -1),
            "silhouette_score": float(silhouette),
            "method": method
        }
        
        return labels.tolist(), metrics
    
    def _simple_clustering(
        self,
        sources: List[AnnotationSource]
    ) -> Tuple[List[int], Dict]:
        """
        Fallback clustering when sklearn unavailable.
        Groups by coordinate proximity.
        """
        labels = []
        current_cluster = 0
        tolerance = 500  # 500bp tolerance
        
        sorted_sources = sorted(enumerate(sources), key=lambda x: x[1].start)
        
        for i, (original_idx, source) in enumerate(sorted_sources):
            if i == 0:
                labels.append(current_cluster)
            else:
                prev_source = sorted_sources[i-1][1]
                if abs(source.start - prev_source.start) <= tolerance:
                    labels.append(current_cluster)
                else:
                    current_cluster += 1
                    labels.append(current_cluster)
        
        # Reorder labels to match original order
        final_labels = [0] * len(sources)
        for (original_idx, _), label in zip(sorted_sources, labels):
            final_labels[original_idx] = label
        
        metrics = {
            "n_clusters": len(set(final_labels)),
            "n_noise": 0,
            "silhouette_score": 0.0,
            "method": "simple_proximity"
        }
        
        return final_labels, metrics
    
    def resolve_with_clustering(
        self,
        gene_symbol: str,
        sources: List[AnnotationSource]
    ) -> ConflictResolution:
        """
        Resolve conflicts using clustering-based consensus.
        """
        if len(sources) < 2:
            return self._single_source_resolution(gene_symbol, sources[0] if sources else None)
        
        # Cluster the sources
        labels, metrics = self.cluster_annotations(sources)
        
        # Find largest cluster (consensus group)
        label_counts = Counter(l for l in labels if l != -1)
        
        if not label_counts:
            # All noise, fall back to weighted average
            return self._weighted_average_resolution(gene_symbol, sources)
        
        consensus_label = label_counts.most_common(1)[0][0]
        consensus_sources = [s for s, l in zip(sources, labels) if l == consensus_label]
        
        # Calculate consensus coordinates with weights
        weighted_starts = []
        weighted_ends = []
        total_weight = 0
        
        for source in consensus_sources:
            weight = self.source_reliability.get(source.name, 0.5)
            weight *= source.confidence
            
            # Evidence bonus
            evidence_bonus = sum(self.evidence_weights.get(e, 0.1) for e in source.evidence)
            weight *= (1 + evidence_bonus * 0.1)
            
            weighted_starts.append(source.start * weight)
            weighted_ends.append(source.end * weight)
            total_weight += weight
        
        if total_weight == 0:
            resolved_start = int(np.mean([s.start for s in consensus_sources]))
            resolved_end = int(np.mean([s.end for s in consensus_sources]))
        else:
            resolved_start = int(sum(weighted_starts) / total_weight)
            resolved_end = int(sum(weighted_ends) / total_weight)
        
        # Determine strand consensus
        strands = [s.strand for s in consensus_sources if s.strand]
        resolved_strand = Counter(strands).most_common(1)[0][0] if strands else None
        
        # Calculate confidence
        consensus_level = len(consensus_sources) / len(sources)
        cluster_quality = metrics.get("silhouette_score", 0.0)
        avg_source_confidence = np.mean([s.confidence for s in consensus_sources])
        
        confidence_score = (
            consensus_level * 0.4 +
            cluster_quality * 0.3 +
            avg_source_confidence * 0.3
        )
        
        # Statistical metrics
        starts = [s.start for s in consensus_sources]
        ends = [s.end for s in consensus_sources]
        
        statistical_metrics = {
            "start_std": float(np.std(starts)),
            "end_std": float(np.std(ends)),
            "start_range": max(starts) - min(starts),
            "end_range": max(ends) - min(ends),
            "n_sources_consensus": len(consensus_sources),
            "clustering_metrics": metrics
        }
        
        # Recommendation
        if confidence_score >= 0.9 and statistical_metrics["start_range"] < 100:
            recommendation = "HIGH_CONFIDENCE - Automatic resolution recommended"
            manual_review = False
        elif confidence_score >= 0.7:
            recommendation = "MODERATE_CONFIDENCE - Consider manual verification"
            manual_review = False
        else:
            recommendation = "LOW_CONFIDENCE - Manual review required"
            manual_review = True
        
        conflicts = self.detect_conflicts(sources)
        
        return ConflictResolution(
            gene_symbol=gene_symbol,
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            resolved_strand=resolved_strand,
            confidence_score=float(confidence_score),
            resolution_method="ml_clustering",
            contributing_sources=[s.name for s in consensus_sources],
            conflict_types=conflicts,
            consensus_level=float(consensus_level),
            statistical_metrics=statistical_metrics,
            recommendation=recommendation,
            manual_review_needed=manual_review
        )
    
    def _weighted_average_resolution(
        self,
        gene_symbol: str,
        sources: List[AnnotationSource]
    ) -> ConflictResolution:
        """Fallback to weighted average when clustering fails"""
        
        weighted_starts = []
        weighted_ends = []
        total_weight = 0
        
        for source in sources:
            weight = self.source_reliability.get(source.name, 0.5) * source.confidence
            weighted_starts.append(source.start * weight)
            weighted_ends.append(source.end * weight)
            total_weight += weight
        
        resolved_start = int(sum(weighted_starts) / total_weight) if total_weight > 0 else int(np.mean([s.start for s in sources]))
        resolved_end = int(sum(weighted_ends) / total_weight) if total_weight > 0 else int(np.mean([s.end for s in sources]))
        
        conflicts = self.detect_conflicts(sources)
        
        return ConflictResolution(
            gene_symbol=gene_symbol,
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            resolved_strand=Counter([s.strand for s in sources if s.strand]).most_common(1)[0][0] if any(s.strand for s in sources) else None,
            confidence_score=0.6,
            resolution_method="weighted_average",
            contributing_sources=[s.name for s in sources],
            conflict_types=conflicts,
            consensus_level=1.0,
            statistical_metrics={
                "start_std": float(np.std([s.start for s in sources])),
                "end_std": float(np.std([s.end for s in sources]))
            },
            recommendation="MODERATE_CONFIDENCE - Weighted average used",
            manual_review_needed=True
        )
    
    def _single_source_resolution(
        self,
        gene_symbol: str,
        source: Optional[AnnotationSource]
    ) -> ConflictResolution:
        """Handle case with only one source"""
        if not source:
            return ConflictResolution(
                gene_symbol=gene_symbol,
                resolved_start=0,
                resolved_end=0,
                resolved_strand=None,
                confidence_score=0.0,
                resolution_method="no_data",
                contributing_sources=[],
                conflict_types=[],
                consensus_level=0.0,
                statistical_metrics={},
                recommendation="NO_DATA - Cannot resolve",
                manual_review_needed=True
            )
        
        return ConflictResolution(
            gene_symbol=gene_symbol,
            resolved_start=source.start,
            resolved_end=source.end,
            resolved_strand=source.strand,
            confidence_score=source.confidence * self.source_reliability.get(source.name, 0.5),
            resolution_method="single_source",
            contributing_sources=[source.name],
            conflict_types=[],
            consensus_level=1.0,
            statistical_metrics={},
            recommendation="SINGLE_SOURCE - No conflicts to resolve",
            manual_review_needed=False
        )
    
    def batch_resolve(
        self,
        gene_annotations: List[Dict]
    ) -> List[ConflictResolution]:
        """
        Resolve conflicts for multiple genes.
        
        Args:
            gene_annotations: List of dicts with format:
                {
                    "gene_symbol": "BRCA1",
                    "sources": [
                        {"name": "Ensembl", "start": 100, "end": 200, ...},
                        ...
                    ]
                }
        
        Returns:
            List of ConflictResolution objects
        """
        results = []
        
        for gene_data in gene_annotations:
            gene_symbol = gene_data.get("gene_symbol", "Unknown")
            
            # Convert to AnnotationSource objects
            sources = []
            for src_data in gene_data.get("sources", []):
                source = AnnotationSource(
                    name=src_data.get("name", "Unknown"),
                    start=src_data.get("start", 0),
                    end=src_data.get("end", 0),
                    strand=src_data.get("strand"),
                    confidence=src_data.get("confidence", 0.8),
                    biotype=src_data.get("biotype"),
                    description=src_data.get("description"),
                    evidence=src_data.get("evidence", [])
                )
                sources.append(source)
            
            # Resolve
            resolution = self.resolve_with_clustering(gene_symbol, sources)
            results.append(resolution)
        
        return results
    
    def generate_report(self, resolutions: List[ConflictResolution]) -> Dict:
        """Generate summary report for batch resolutions"""
        
        high_conf = sum(1 for r in resolutions if r.confidence_score >= 0.9)
        med_conf = sum(1 for r in resolutions if 0.7 <= r.confidence_score < 0.9)
        low_conf = sum(1 for r in resolutions if r.confidence_score < 0.7)
        
        needs_review = sum(1 for r in resolutions if r.manual_review_needed)
        
        avg_confidence = statistics.mean([r.confidence_score for r in resolutions]) if resolutions else 0
        
        return {
            "total_genes": len(resolutions),
            "high_confidence": high_conf,
            "medium_confidence": med_conf,
            "low_confidence": low_conf,
            "needs_manual_review": needs_review,
            "average_confidence": round(avg_confidence, 3),
            "success_rate": round((high_conf + med_conf) / len(resolutions) * 100, 2) if resolutions else 0
        }