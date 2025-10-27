import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import time
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    COORDINATE_MISMATCH = "coordinate_mismatch"
    GENE_BOUNDARY_CONFLICT = "gene_boundary_conflict"
    STRAND_INCONSISTENCY = "strand_inconsistency"
    VERSION_CONFLICT = "version_conflict"
    BIOTYPE_DISAGREEMENT = "biotype_disagreement"
    ANNOTATION_GAP = "annotation_gap"
    OVERLAPPING_GENES = "overlapping_genes"

class ResolutionStatus(Enum):
    RESOLVED = "resolved"
    MANUAL_REVIEW = "manual_review"
    INSUFFICIENT_DATA = "insufficient_data"
    IRRECONCILABLE = "irreconcilable"

@dataclass
class AnnotationSource:
    name: str
    start: int
    end: int
    strand: Optional[str] = None
    version: Optional[str] = None
    confidence: float = 0.8
    evidence: List[str] = None
    biotype: Optional[str] = None
    description: Optional[str] = None
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

@dataclass
class ConflictResolution:
    gene_symbol: str
    resolved_coordinates: Dict
    confidence_score: float
    resolution_method: str
    status: ResolutionStatus
    contributing_sources: List[str]
    conflict_types: List[ConflictType]
    evidence_summary: Dict
    manual_review_notes: List[str]
    processing_time_ms: float
    ai_reasoning: str
    
    def to_dict(self):
        return {
            **asdict(self),
            'status': self.status.value,
            'conflict_types': [ct.value for ct in self.conflict_types]
        }

# Mock ML components for lightweight implementation
# Replace the mock NeuralConflictPredictor in ai_conflict_resolver.py

import numpy as np
from typing import List, Dict, Tuple
import math

class AnnotationClusterer:
    """Real k-means clustering implementation for grouping similar annotations"""
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def _initialize_centroids(self, data: List[List[float]], k: int) -> List[List[float]]:
        """Initialize centroids using k-means++ method"""
        if not data or k <= 0:
            return []
        
        centroids = []
        
        # Choose first centroid randomly
        import random
        centroids.append(data[random.randint(0, len(data) - 1)].copy())
        
        # Choose remaining centroids with probability proportional to squared distance
        for _ in range(1, k):
            distances = []
            for point in data:
                min_dist = min(self._calculate_distance(point, c) for c in centroids)
                distances.append(min_dist ** 2)
            
            total_dist = sum(distances)
            if total_dist > 0:
                # Weighted random selection
                r = random.random() * total_dist
                cumsum = 0
                for i, dist in enumerate(distances):
                    cumsum += dist
                    if cumsum >= r:
                        centroids.append(data[i].copy())
                        break
            else:
                centroids.append(data[random.randint(0, len(data) - 1)].copy())
        
        return centroids
    
    def fit_predict(self, data: List[List[float]], k: int) -> Tuple[List[int], List[List[float]]]:
        """Perform k-means clustering and return labels and centroids"""
        if not data or k <= 0:
            return [], []
        
        if k > len(data):
            k = len(data)
        
        # Initialize centroids
        centroids = self._initialize_centroids(data, k)
        
        for iteration in range(self.max_iterations):
            # Assign points to closest centroid
            labels = []
            for point in data:
                distances = [self._calculate_distance(point, c) for c in centroids]
                labels.append(distances.index(min(distances)))
            
            # Update centroids
            new_centroids = []
            converged = True
            
            for i in range(k):
                cluster_points = [data[j] for j, label in enumerate(labels) if label == i]
                
                if cluster_points:
                    # Calculate mean of cluster points
                    new_centroid = []
                    for dim in range(len(cluster_points[0])):
                        mean_val = sum(point[dim] for point in cluster_points) / len(cluster_points)
                        new_centroid.append(mean_val)
                    new_centroids.append(new_centroid)
                    
                    # Check convergence
                    if (len(centroids) > i and 
                        self._calculate_distance(new_centroid, centroids[i]) > self.tolerance):
                        converged = False
                else:
                    # Keep old centroid if no points assigned
                    new_centroids.append(centroids[i] if len(centroids) > i else data[0].copy())
            
            centroids = new_centroids
            
            if converged:
                break
        
        self.centroids = centroids
        self.labels = labels
        
        return labels, centroids

class RealConflictResolver:
    """Real ML-based conflict resolution using clustering"""
    
    def __init__(self):
        self.clusterer = AnnotationClusterer()
        self.source_reliability_scores = {
            "GENCODE": 0.98,
            "Ensembl": 0.95,
            "RefSeq": 0.92,
            "UCSC": 0.88,
        }
    
    def extract_features(self, sources: List[AnnotationSource]) -> List[List[float]]:
        """Extract numerical features from annotation sources"""
        features = []
        
        for source in sources:
            feature_vector = [
                float(source.start),
                float(source.end),
                float(source.end - source.start),  # length
                self.source_reliability_scores.get(source.name, 0.5),
                source.confidence,
                len(source.evidence),
                1.0 if source.strand == '+' else (-1.0 if source.strand == '-' else 0.0),
                1.0 if 'experimental' in source.evidence else 0.0,
                1.0 if 'literature' in source.evidence else 0.0,
            ]
            features.append(feature_vector)
        
        return features
    
    def resolve_with_clustering(self, sources: List[AnnotationSource]) -> Dict:
        """Resolve conflicts using k-means clustering"""
        if len(sources) < 2:
            return {"error": "Need at least 2 sources for clustering"}
        
        # Extract features
        features = self.extract_features(sources)
        
        # Determine optimal number of clusters (max 3 for genomic data)
        k = min(3, max(2, len(sources) // 2))
        
        # Perform clustering
        labels, centroids = self.clusterer.fit_predict(features, k)
        
        # Find the largest cluster (consensus)
        cluster_sizes = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        largest_cluster = max(cluster_sizes.keys(), key=cluster_sizes.get)
        consensus_sources = [sources[i] for i, label in enumerate(labels) 
                           if label == largest_cluster]
        
        # Calculate consensus coordinates
        if consensus_sources:
            avg_start = sum(s.start for s in consensus_sources) / len(consensus_sources)
            avg_end = sum(s.end for s in consensus_sources) / len(consensus_sources)
            
            # Weight by source reliability
            weighted_start = sum(s.start * self.source_reliability_scores.get(s.name, 0.5) 
                               for s in consensus_sources)
            weighted_end = sum(s.end * self.source_reliability_scores.get(s.name, 0.5) 
                             for s in consensus_sources)
            total_weight = sum(self.source_reliability_scores.get(s.name, 0.5) 
                             for s in consensus_sources)
            
            if total_weight > 0:
                final_start = int(weighted_start / total_weight)
                final_end = int(weighted_end / total_weight)
            else:
                final_start = int(avg_start)
                final_end = int(avg_end)
            
            confidence = len(consensus_sources) / len(sources)
            
            return {
                "resolved_coordinates": {
                    "start": final_start,
                    "end": final_end,
                    "method": "k_means_clustering"
                },
                "confidence": confidence,
                "consensus_sources": [s.name for s in consensus_sources],
                "cluster_analysis": {
                    "total_clusters_found": len(set(labels)),
                    "largest_cluster_size": len(consensus_sources),
                    "cluster_labels": labels,
                    "clustering_method": "k_means_plus_plus"
                }
            }
        
        return {"error": "No consensus cluster found"}

# Replace in ai_conflict_resolver.py
class AIConflictResolver:
    def __init__(self):
        # ... existing code ...
        self.real_resolver = RealConflictResolver()  # Add this line
    
    async def _ai_weighted_resolution(self, gene_symbol: str, sources: List[AnnotationSource], 
                                    conflicts: List[ConflictType], threshold: float) -> ConflictResolution:
        """AI-weighted resolution using real clustering algorithm"""
        
        # Use real clustering instead of mock prediction
        clustering_result = self.real_resolver.resolve_with_clustering(sources)
        
        if "error" not in clustering_result:
            coords = clustering_result["resolved_coordinates"]
            confidence = clustering_result["confidence"]
            
            status = (ResolutionStatus.RESOLVED if confidence >= threshold 
                     else ResolutionStatus.MANUAL_REVIEW)
            
            ai_reasoning = f"K-means clustering identified {clustering_result['cluster_analysis']['total_clusters_found']} distinct annotation groups. " \
                          f"Consensus from {len(clustering_result['consensus_sources'])} sources: {', '.join(clustering_result['consensus_sources'])}. " \
                          f"Clustering confidence: {confidence:.3f}"
            
            return ConflictResolution(
                gene_symbol=gene_symbol,
                resolved_coordinates=coords,
                confidence_score=confidence,
                resolution_method="k_means_clustering",
                status=status,
                contributing_sources=clustering_result['consensus_sources'],
                conflict_types=conflicts,
                evidence_summary=self._generate_evidence_summary(sources),
                manual_review_notes=[] if confidence >= threshold else ["Low clustering confidence"],
                processing_time_ms=0.0,
                ai_reasoning=ai_reasoning
            )
        
        # Fallback to original method if clustering fails
        return await self._consensus_voting_resolution(gene_symbol, sources, conflicts, threshold)

class StandardScaler:
    """Mock scaler for feature normalization"""
    
    def __init__(self):
        self.mean = 0
        self.std = 1
    
    def fit(self, X):
        if X:
            self.mean = np.mean(X, axis=0) if hasattr(np, 'mean') else 0
            self.std = np.std(X, axis=0) if hasattr(np, 'std') else 1
    
    def transform(self, X):
        return X  # Simple pass-through for mock

class AIConflictResolver:
    """Advanced AI system for genomic annotation conflict resolution"""
    
    def __init__(self):
        self.source_reliability_scores = {
            "GENCODE": 0.98,
            "Ensembl": 0.95,
            "RefSeq": 0.92,
            "UCSC": 0.88,
            "NCBI": 0.90,
            "Havana": 0.94,
            "CHESS": 0.85
        }
        
        self.evidence_weights = {
            "experimental": 1.0,
            "literature": 0.9,
            "computational": 0.7,
            "manual_annotation": 0.95,
            "automatic_annotation": 0.6,
            "protein_evidence": 0.9,
            "transcript_evidence": 0.85,
            "conservation_evidence": 0.8
        }
        
        self.conflict_predictor = NeuralConflictPredictor()
        self.scaler = StandardScaler()
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Generate some synthetic training data for the mock models
            X_train = self._generate_synthetic_training_data()
            self.scaler.fit(X_train)
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.warning(f"Model initialization warning: {e}")
    
    def _generate_synthetic_training_data(self):
        """Generate synthetic training data"""
        try:
            import random
            data = []
            for _ in range(100):
                data.append([
                    random.uniform(0, 1000),  # coordinate variance
                    random.uniform(0.5, 1.0), # confidence
                    random.randint(2, 5),     # source count
                    random.uniform(0, 2),     # conflict count
                    random.uniform(0.5, 1.0), # reliability
                ])
            return data
        except ImportError:
            return [[0.5, 0.8, 2, 1, 0.9]]  # Default data if random not available
    
    async def resolve_conflicts(
        self, 
        annotation_group: Dict, 
        strategy: str = "ai_weighted",
        threshold: float = 0.8
    ) -> ConflictResolution:
        """Main conflict resolution method using AI"""
        start_time = time.time()
        
        try:
            gene_symbol = annotation_group.get("gene_symbol", "Unknown")
            sources = annotation_group.get("sources", [])
            
            if len(sources) < 2:
                return ConflictResolution(
                    gene_symbol=gene_symbol,
                    resolved_coordinates={},
                    confidence_score=0.0,
                    resolution_method="insufficient_data",
                    status=ResolutionStatus.INSUFFICIENT_DATA,
                    contributing_sources=[],
                    conflict_types=[],
                    evidence_summary={},
                    manual_review_notes=["Less than 2 sources provided"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    ai_reasoning="Cannot resolve conflicts with less than 2 annotation sources"
                )

            annotation_sources = [
                AnnotationSource(
                    name=src.get("name", "Unknown"),
                    start=src.get("start", 0),
                    end=src.get("end", 0),
                    strand=src.get("strand"),
                    version=src.get("version"),
                    confidence=src.get("confidence", 0.8),
                    evidence=src.get("evidence", []),
                    biotype=src.get("biotype"),
                    description=src.get("description")
                ) for src in sources
            ]

            conflicts = self._detect_conflicts_in_sources(annotation_sources)

            if strategy == "ai_weighted":
                resolution = await self._ai_weighted_resolution(
                    gene_symbol, annotation_sources, conflicts, threshold
                )
            elif strategy == "consensus_voting":
                resolution = await self._consensus_voting_resolution(
                    gene_symbol, annotation_sources, conflicts, threshold
                )
            elif strategy == "evidence_based":
                resolution = await self._evidence_based_resolution(
                    gene_symbol, annotation_sources, conflicts, threshold
                )
            else:
                resolution = await self._ai_weighted_resolution(
                    gene_symbol, annotation_sources, conflicts, threshold
                )
            
            resolution.processing_time_ms = (time.time() - start_time) * 1000
            return resolution
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return ConflictResolution(
                gene_symbol=annotation_group.get("gene_symbol", "Unknown"),
                resolved_coordinates={},
                confidence_score=0.0,
                resolution_method="error",
                status=ResolutionStatus.IRRECONCILABLE,
                contributing_sources=[],
                conflict_types=[],
                evidence_summary={},
                manual_review_notes=[f"Error during resolution: {str(e)}"],
                processing_time_ms=(time.time() - start_time) * 1000,
                ai_reasoning=f"Resolution failed due to error: {str(e)}"
            )
    
    def _detect_conflicts_in_sources(self, sources: List[AnnotationSource]) -> List[ConflictType]:
        """Detect different types of conflicts between annotation sources"""
        conflicts = []
        
        if len(sources) < 2:
            return conflicts

        starts = [src.start for src in sources]
        ends = [src.end for src in sources]
        
        # Check coordinate conflicts
        if max(starts) - min(starts) > 100 or max(ends) - min(ends) > 100:
            conflicts.append(ConflictType.COORDINATE_MISMATCH)

        # Check strand conflicts
        strands = [src.strand for src in sources if src.strand]
        if len(set(strands)) > 1:
            conflicts.append(ConflictType.STRAND_INCONSISTENCY)

        # Check biotype conflicts
        biotypes = [src.biotype for src in sources if src.biotype]
        if len(set(biotypes)) > 1:
            conflicts.append(ConflictType.BIOTYPE_DISAGREEMENT)

        # Check boundary overlaps
        for i, src1 in enumerate(sources):
            for src2 in sources[i+1:]:
                if self._check_boundary_overlap(src1, src2):
                    conflicts.append(ConflictType.GENE_BOUNDARY_CONFLICT)
                    break
        
        return conflicts
    
    def _check_boundary_overlap(self, src1: AnnotationSource, src2: AnnotationSource) -> bool:
        """Check if two annotations have conflicting boundaries"""
        return not (src1.end < src2.start or src2.end < src1.start)
    
    async def _ai_weighted_resolution(
        self, 
        gene_symbol: str, 
        sources: List[AnnotationSource], 
        conflicts: List[ConflictType], 
        threshold: float
    ) -> ConflictResolution:
        """AI-weighted resolution using neural network confidence prediction"""

        features = self._extract_conflict_features(sources, conflicts)
        
        # Use mock predictor
        ai_confidence = self.conflict_predictor.predict(features)

        weighted_coords = self._calculate_weighted_coordinates(sources)

        if ai_confidence >= threshold and len(conflicts) <= 2:
            status = ResolutionStatus.RESOLVED
        elif ai_confidence >= threshold * 0.7:
            status = ResolutionStatus.MANUAL_REVIEW
        else:
            status = ResolutionStatus.IRRECONCILABLE

        evidence_summary = self._generate_evidence_summary(sources)
        ai_reasoning = self._generate_ai_reasoning(sources, conflicts, ai_confidence, weighted_coords)

        review_notes = []
        if ai_confidence < threshold:
            review_notes.append(f"Low AI confidence: {ai_confidence:.3f}")
        if len(conflicts) > 2:
            review_notes.append(f"Multiple conflicts detected: {[c.value for c in conflicts]}")
        
        return ConflictResolution(
            gene_symbol=gene_symbol,
            resolved_coordinates=weighted_coords,
            confidence_score=ai_confidence,
            resolution_method="ai_weighted_neural",
            status=status,
            contributing_sources=[src.name for src in sources],
            conflict_types=conflicts,
            evidence_summary=evidence_summary,
            manual_review_notes=review_notes,
            processing_time_ms=0.0,
            ai_reasoning=ai_reasoning
        )
    
    async def _consensus_voting_resolution(
        self, 
        gene_symbol: str, 
        sources: List[AnnotationSource], 
        conflicts: List[ConflictType], 
        threshold: float
    ) -> ConflictResolution:
        """Consensus voting resolution method"""

        coordinate_groups = self._group_similar_coordinates(sources)

        if coordinate_groups:
            largest_group = max(coordinate_groups, key=len)
            consensus_coords = self._calculate_group_average(largest_group)
            
            confidence = len(largest_group) / len(sources)
            
            status = ResolutionStatus.RESOLVED if confidence >= threshold else ResolutionStatus.MANUAL_REVIEW
            
            return ConflictResolution(
                gene_symbol=gene_symbol,
                resolved_coordinates=consensus_coords,
                confidence_score=confidence,
                resolution_method="consensus_voting",
                status=status,
                contributing_sources=[src.name for src in largest_group],
                conflict_types=conflicts,
                evidence_summary=self._generate_evidence_summary(largest_group),
                manual_review_notes=[] if confidence >= threshold else ["Low consensus agreement"],
                processing_time_ms=0.0,
                ai_reasoning=f"Consensus reached with {len(largest_group)}/{len(sources)} sources agreeing"
            )
        
        return ConflictResolution(
            gene_symbol=gene_symbol,
            resolved_coordinates={},
            confidence_score=0.0,
            resolution_method="consensus_voting",
            status=ResolutionStatus.IRRECONCILABLE,
            contributing_sources=[],
            conflict_types=conflicts,
            evidence_summary={},
            manual_review_notes=["No consensus could be reached"],
            processing_time_ms=0.0,
            ai_reasoning="No coordinate consensus found among sources"
        )
    
    async def _evidence_based_resolution(
        self, 
        gene_symbol: str, 
        sources: List[AnnotationSource], 
        conflicts: List[ConflictType], 
        threshold: float
    ) -> ConflictResolution:
        """Evidence-based resolution prioritizing sources with strong evidence"""

        evidence_scores = []
        for src in sources:
            score = self._calculate_evidence_score(src)
            evidence_scores.append((src, score))

        evidence_scores.sort(key=lambda x: x[1], reverse=True)
        best_source = evidence_scores[0][0]
        best_score = evidence_scores[0][1]

        resolved_coords = {
            "chromosome": getattr(best_source, 'chromosome', 'Unknown'),
            "start": best_source.start,
            "end": best_source.end,
            "strand": best_source.strand,
            "biotype": best_source.biotype
        }
        
        confidence = min(best_score, 1.0)
        status = ResolutionStatus.RESOLVED if confidence >= threshold else ResolutionStatus.MANUAL_REVIEW
        
        return ConflictResolution(
            gene_symbol=gene_symbol,
            resolved_coordinates=resolved_coords,
            confidence_score=confidence,
            resolution_method="evidence_based",
            status=status,
            contributing_sources=[best_source.name],
            conflict_types=conflicts,
            evidence_summary=self._generate_evidence_summary([best_source]),
            manual_review_notes=[] if confidence >= threshold else [f"Evidence score below threshold: {confidence:.3f}"],
            processing_time_ms=0.0,
            ai_reasoning=f"Selected {best_source.name} as primary source based on evidence strength (score: {best_score:.3f})"
        )
    
    def _extract_conflict_features(self, sources: List[AnnotationSource], conflicts: List[ConflictType]) -> List[float]:
        """Extract features for ML model"""
        if not sources:
            return [0.0] * 15

        starts = [src.start for src in sources]
        ends = [src.end for src in sources]
        confidences = [src.confidence for src in sources]
        
        try:
            coordinate_variance = np.var(starts) + np.var(ends) if hasattr(np, 'var') else sum((x - sum(starts)/len(starts))**2 for x in starts) / len(starts)
            avg_confidence = sum(confidences) / len(confidences)
        except:
            coordinate_variance = 0.0
            avg_confidence = 0.8
            
        source_count = len(sources)
        conflict_count = len(conflicts)

        max_reliability = max([self.source_reliability_scores.get(src.name, 0.5) for src in sources])
        avg_reliability = sum([self.source_reliability_scores.get(src.name, 0.5) for src in sources]) / len(sources)

        total_evidence = sum([len(src.evidence) for src in sources])
        has_experimental = any(['experimental' in src.evidence for src in sources])
        has_literature = any(['literature' in src.evidence for src in sources])

        strands = [src.strand for src in sources if src.strand]
        strand_agreement = 1.0 if len(set(strands)) <= 1 else 0.0
        
        biotypes = [src.biotype for src in sources if src.biotype]
        biotype_agreement = 1.0 if len(set(biotypes)) <= 1 else 0.0
        
        try:
            std_starts = np.std(starts) if hasattr(np, 'std') and len(starts) > 1 else 0.0
            std_ends = np.std(ends) if hasattr(np, 'std') and len(ends) > 1 else 0.0
        except:
            std_starts = 0.0
            std_ends = 0.0
        
        return [
            coordinate_variance,
            avg_confidence,
            source_count,
            conflict_count,
            max_reliability,
            avg_reliability,
            total_evidence,
            float(has_experimental),
            float(has_literature),
            strand_agreement,
            biotype_agreement,
            std_starts,
            std_ends,
            max(ends) - min(starts) if ends and starts else 0,
            len([c for c in conflicts if c == ConflictType.COORDINATE_MISMATCH])
        ]
    
    def _calculate_weighted_coordinates(self, sources: List[AnnotationSource]) -> Dict:
        """Calculate weighted coordinates based on source reliability and evidence"""
        if not sources:
            return {}
        
        total_weight = 0
        weighted_start = 0
        weighted_end = 0
        
        for src in sources:
            weight = self.source_reliability_scores.get(src.name, 0.5)

            # Evidence bonus
            evidence_bonus = 0
            for evidence_type in src.evidence:
                evidence_bonus += self.evidence_weights.get(evidence_type, 0.1)

            # Apply confidence multiplier
            weight *= src.confidence

            # Apply evidence bonus
            weight *= min(1.5, 1.0 + evidence_bonus * 0.1)
            
            weighted_start += src.start * weight
            weighted_end += src.end * weight
            total_weight += weight
        
        if total_weight == 0:
            return {}

        best_source = max(sources, key=lambda s: self.source_reliability_scores.get(s.name, 0) * s.confidence)
        
        return {
            "chromosome": getattr(best_source, 'chromosome', 'Unknown'),
            "start": int(round(weighted_start / total_weight)),
            "end": int(round(weighted_end / total_weight)),
            "strand": best_source.strand,
            "biotype": best_source.biotype,
            "source_weights": {src.name: self.source_reliability_scores.get(src.name, 0.5) * src.confidence for src in sources}
        }
    
    def _group_similar_coordinates(self, sources: List[AnnotationSource], tolerance: int = 50) -> List[List[AnnotationSource]]:
        """Group sources with similar coordinates"""
        if not sources:
            return []
        
        groups = []
        used_indices = set()
        
        for i, src1 in enumerate(sources):
            if i in used_indices:
                continue
                
            current_group = [src1]
            used_indices.add(i)
            
            for j, src2 in enumerate(sources[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if (abs(src1.start - src2.start) <= tolerance and 
                    abs(src1.end - src2.end) <= tolerance):
                    current_group.append(src2)
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_group_average(self, group: List[AnnotationSource]) -> Dict:
        """Calculate average coordinates for a group"""
        if not group:
            return {}
        
        avg_start = sum(src.start for src in group) / len(group)
        avg_end = sum(src.end for src in group) / len(group)

        strands = [src.strand for src in group if src.strand]
        biotypes = [src.biotype for src in group if src.biotype]
        
        most_common_strand = max(set(strands), key=strands.count) if strands else None
        most_common_biotype = max(set(biotypes), key=biotypes.count) if biotypes else None
        
        return {
            "chromosome": getattr(group[0], 'chromosome', 'Unknown'),
            "start": int(round(avg_start)),
            "end": int(round(avg_end)),
            "strand": most_common_strand,
            "biotype": most_common_biotype,
            "consensus_sources": [src.name for src in group]
        }
    
    def _calculate_evidence_score(self, source: AnnotationSource) -> float:
        """Calculate evidence strength score for a source"""
        base_score = self.source_reliability_scores.get(source.name, 0.5)

        evidence_score = 0
        for evidence_type in source.evidence:
            evidence_score += self.evidence_weights.get(evidence_type, 0.1)

        confidence_multiplier = source.confidence

        total_score = base_score * confidence_multiplier * (1.0 + evidence_score * 0.2)
        
        return min(total_score, 1.0)
    
    def _generate_evidence_summary(self, sources: List[AnnotationSource]) -> Dict:
        """Generate comprehensive evidence summary"""
        if not sources:
            return {}
        
        all_evidence = []
        for src in sources:
            all_evidence.extend(src.evidence)
        
        evidence_counts = {}
        for evidence in all_evidence:
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
        
        source_summary = {}
        for src in sources:
            source_summary[src.name] = {
                "reliability_score": self.source_reliability_scores.get(src.name, 0.5),
                "confidence": src.confidence,
                "evidence_types": src.evidence,
                "evidence_score": self._calculate_evidence_score(src)
            }
        
        return {
            "total_sources": len(sources),
            "evidence_distribution": evidence_counts,
            "strongest_evidence": max(evidence_counts.keys(), key=evidence_counts.get) if evidence_counts else None,
            "source_details": source_summary,
            "average_confidence": sum(src.confidence for src in sources) / len(sources),
            "highest_reliability_source": max(sources, key=lambda s: self.source_reliability_scores.get(s.name, 0)).name
        }
    
    def _generate_ai_reasoning(self, sources: List[AnnotationSource], conflicts: List[ConflictType], 
                             confidence: float, resolved_coords: Dict) -> str:
        """Generate human-readable AI reasoning"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Analyzed {len(sources)} annotation sources: {', '.join(src.name for src in sources)}")
        
        if conflicts:
            reasoning_parts.append(f"Detected {len(conflicts)} conflict types: {', '.join(c.value for c in conflicts)}")
        else:
            reasoning_parts.append("No major conflicts detected between sources")

        best_source = max(sources, key=lambda s: self.source_reliability_scores.get(s.name, 0))
        reasoning_parts.append(f"Highest reliability source: {best_source.name} (score: {self.source_reliability_scores.get(best_source.name, 0):.2f})")

        if resolved_coords and 'start' in resolved_coords:
            coordinate_range = resolved_coords.get('end', 0) - resolved_coords.get('start', 0)
            reasoning_parts.append(f"Resolved coordinates span {coordinate_range:,} bp")

        if confidence >= 0.9:
            reasoning_parts.append("High confidence resolution - automatic processing recommended")
        elif confidence >= 0.7:
            reasoning_parts.append("Moderate confidence - may benefit from expert review")
        else:
            reasoning_parts.append("Low confidence - manual validation strongly recommended")
        
        return ". ".join(reasoning_parts) + "."
    
    async def detect_conflicts(self, annotation: Dict, all_annotations: List[Dict], 
                             sensitivity: str = "high") -> List[Dict]:
        """Detect conflicts for a single annotation against a dataset"""
        conflicts = []
        
        try:
            target_chr = annotation.get("chromosome", "")
            target_start = annotation.get("start", 0)
            target_end = annotation.get("end", 0)
            target_gene = annotation.get("gene_symbol", "")
            
            sensitivity_thresholds = {
                "low": 1000,
                "medium": 500,
                "high": 100,
                "ultra": 50
            }
            threshold = sensitivity_thresholds.get(sensitivity, 100)
            
            for i, other_ann in enumerate(all_annotations):
                if other_ann == annotation:
                    continue
                
                other_chr = other_ann.get("chromosome", "")
                other_start = other_ann.get("start", 0)
                other_end = other_ann.get("end", 0)
                other_gene = other_ann.get("gene_symbol", "")

                # Same chromosome checks
                if target_chr == other_chr:
                    # Check for gene overlaps
                    if (target_gene != other_gene and target_gene and other_gene and
                        not (target_end < other_start or other_end < target_start)):
                        conflicts.append({
                            "type": ConflictType.OVERLAPPING_GENES.value,
                            "severity": "high",
                            "description": f"Gene overlap: {target_gene} and {other_gene}",
                            "conflicting_annotation_index": i,
                            "overlap_size": min(target_end, other_end) - max(target_start, other_start)
                        })

                    # Check for coordinate mismatches
                    if (target_gene == other_gene and target_gene != "" and
                        (abs(target_start - other_start) > threshold or 
                         abs(target_end - other_end) > threshold)):
                        conflicts.append({
                            "type": ConflictType.COORDINATE_MISMATCH.value,
                            "severity": "medium",
                            "description": f"Coordinate mismatch for {target_gene}",
                            "conflicting_annotation_index": i,
                            "coordinate_difference": max(abs(target_start - other_start), abs(target_end - other_end))
                        })
        
        except Exception as e:
            logger.error(f"Conflict detection error: {e}")
        
        return conflicts
    
    def categorize_conflicts(self, conflicts: List[Dict]) -> Dict:
        """Categorize detected conflicts by type and severity"""
        categories = {
            "coordinate_mismatch": 0,
            "overlapping_genes": 0,
            "strand_inconsistency": 0,
            "biotype_disagreement": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0
        }
        
        for conflict in conflicts:
            conflict_type = conflict.get("type", "unknown")
            severity = conflict.get("severity", "low")
            
            if conflict_type in categories:
                categories[conflict_type] += 1
            
            categories[f"{severity}_severity"] += 1
        
        return categories
    
    def generate_resolution_recommendations(self, conflicts: List[Dict]) -> List[str]:
        """Generate actionable recommendations for resolving conflicts"""
        recommendations = []
        
        if not conflicts:
            recommendations.append("No conflicts detected - annotations appear consistent")
            return recommendations
        
        conflict_types = set(c.get("type", "") for c in conflicts)
        
        if ConflictType.COORDINATE_MISMATCH.value in conflict_types:
            recommendations.append("Coordinate mismatches detected - consider using consensus coordinates or most reliable source")
        
        if ConflictType.OVERLAPPING_GENES.value in conflict_types:
            recommendations.append("Overlapping genes found - verify gene boundaries and consider alternative splicing")
        
        high_severity_count = sum(1 for c in conflicts if c.get("severity") == "high")
        if high_severity_count > 0:
            recommendations.append(f"{high_severity_count} high-severity conflicts require immediate attention")
        
        if len(conflicts) > 10:
            recommendations.append("Large number of conflicts detected - consider systematic annotation review")
        
        recommendations.append("Use AI conflict resolution for automated suggestions")
        
        return recommendations
    
    async def generate_conflict_analytics(self, resolution_results: List[Dict]) -> Dict:
        """Generate comprehensive analytics about conflict resolution patterns"""
        if not resolution_results:
            return {}
        
        try:
            total_resolutions = len(resolution_results)
            successful_resolutions = sum(1 for r in resolution_results 
                                       if r.get("resolution", {}).get("status") == "resolved")
            
            source_performance = {}
            conflict_patterns = {}
            
            for result in resolution_results:
                resolution = result.get("resolution", {})
                sources = resolution.get("contributing_sources", [])
                
                for source in sources:
                    if source not in source_performance:
                        source_performance[source] = {"count": 0, "success": 0}
                    
                    source_performance[source]["count"] += 1
                    if resolution.get("status") == "resolved":
                        source_performance[source]["success"] += 1

                for conflict_type in resolution.get("conflict_types", []):
                    if isinstance(conflict_type, str):
                        conflict_patterns[conflict_type] = conflict_patterns.get(conflict_type, 0) + 1

            # Calculate success rates
            for source, stats in source_performance.items():
                stats["success_rate"] = stats["success"] / max(stats["count"], 1)

            best_source = max(source_performance.items(), 
                            key=lambda x: x[1]["success_rate"]) if source_performance else ("Unknown", {})

            confidences = [r.get("resolution", {}).get("confidence_score", 0) for r in resolution_results]
            processing_times = [r.get("processing_time_ms", 0) for r in resolution_results]
            
            return {
                "resolution_success_rate": (successful_resolutions / max(total_resolutions, 1)) * 100,
                "average_confidence": sum(confidences) / max(len(confidences), 1),
                "average_processing_time_ms": sum(processing_times) / max(len(processing_times), 1),
                "source_performance": source_performance,
                "best_performing_source": best_source[0],
                "conflict_pattern_frequency": conflict_patterns,
                "most_common_conflict": max(conflict_patterns.keys(), key=conflict_patterns.get) if conflict_patterns else "None",
                "total_genes_processed": total_resolutions,
                "ai_recommendations": [
                    f"Primary source recommendation: {best_source[0]} ({best_source[1].get('success_rate', 0):.1%} success rate)",
                    f"Average resolution confidence: {sum(confidences) / max(len(confidences), 1):.1%}",
                    "Consider manual review for resolutions below 80% confidence"
                ]
            }
        
        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            return {"error": str(e)}
        


        from typing import List, Dict, Any
import math
import collections

def compute_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counts.values():
        p = v/total
        if p > 0:
            ent -= p * math.log2(p)
    return ent
def _resolve_cluster_votes(self, annotations: List[Dict[str, Any]], indices: List[int]) -> Dict[str, Any]:
        """
        Weighted vote within a cluster. Returns consensus and a confidence score in [0,1].
        Ensures the returned confidence_score is clamped to [0.0, 1.0].
        """
        weighted = {}
        sources = {}
        total_weight = 0.0
        for idx in indices:
            ann = annotations[idx]
            val = ann.get("value")
            if val is None:
                continue
            # evidence_score may be arbitrary positive number; we aggregate but final confidence is a ratio
            weight = float(ann.get("evidence_score", 1.0))
            if weight < 0:
                # defensively treat negative evidence as 0
                weight = 0.0
            src = ann.get("source", "unknown")
            weighted[val] = weighted.get(val, 0.0) + weight
            sources.setdefault(val, []).append(src)
            total_weight += weight

        if not weighted or total_weight <= 0:
            return {
                "consensus_value": None,
                "support_count": 0,
                "top_sources": [],
                "weighted_scores": {},
                "confidence_score": 0.0
            }

        sorted_vals = sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))
        top_val, top_weight = sorted_vals[0]
        support_count = len(sources.get(top_val, []))
        # confidence: normalized fraction [0,1]; clamp defensively
        raw_conf = (top_weight / total_weight) if total_weight > 0 else 0.0
        confidence_score = min(max(float(raw_conf), 0.0), 1.0)

        return {
            "consensus_value": top_val,
            "support_count": support_count,
            "top_sources": sources.get(top_val, [])[:10],
            "weighted_scores": weighted,
            "confidence_score": confidence_score
        }


evidence_conf = float(vote.get("confidence_score", 0.0)) if vote.get("confidence_score") is not None else 0.0
semantic_conf = float(vote.get("cluster_semantic_confidence", 1.0))
evidence_conf = min(max(evidence_conf, 0.0), 1.0)
semantic_conf = min(max(semantic_conf, 0.0), 1.0)
combined_confidence = evidence_conf * semantic_conf
combined_confidence = min(max(combined_confidence, 0.0), 1.0)

overall_consensus.append({
                "consensus_value": vote["consensus_value"],
                "combined_confidence": combined_confidence,
                "cluster_id": vote["cluster_id"],
                "cluster_size": vote["cluster_size"],
                "top_sources": vote["top_sources"]
            })