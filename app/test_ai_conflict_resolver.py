import pytest
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai_conflict_resolver import (
    AIConflictResolver, AnnotationSource, ConflictResolution, 
    ConflictType, ResolutionStatus
)

class TestAIConflictResolver:
    
    @pytest.fixture
    def resolver(self):
        """Create AI resolver instance for testing"""
        return AIConflictResolver()
    
    @pytest.fixture 
    def brca1_sources(self):
        """Real BRCA1 annotation sources with conflicts"""
        return [
            AnnotationSource(
                name="Ensembl",
                start=43044295,
                end=43125483,
                strand="-",
                version="110",
                confidence=0.95,
                evidence=["experimental", "computational", "literature"],
                biotype="protein_coding",
                description="BRCA1 DNA repair associated"
            ),
            AnnotationSource(
                name="RefSeq", 
                start=43044294, 
                end=43125482,   
                strand="-",
                version="109", 
                confidence=0.92,
                evidence=["literature", "computational"],
                biotype="protein_coding",
                description="BRCA1 DNA repair associated"
            ),
            AnnotationSource(
                name="GENCODE",
                start=43044295,
                end=43125483,
                strand="-",
                version="44",
                confidence=0.98,
                evidence=["experimental", "literature", "computational", "manual_annotation"],
                biotype="protein_coding",
                description="BRCA1 DNA repair associated"
            )
        ]
    
    @pytest.fixture
    def high_conflict_sources(self):
        """Sources with significant conflicts for stress testing"""
        return [
            AnnotationSource(
                name="Source_A",
                start=1000000,
                end=1010000,
                strand="+",
                confidence=0.85,
                evidence=["computational"],
                biotype="protein_coding"
            ),
            AnnotationSource(
                name="Source_B",
                start=1000500,
                end=1010500,
                strand="-",
                confidence=0.90,
                evidence=["experimental"],
                biotype="lncRNA"  
            ),
            AnnotationSource(
                name="Source_C",
                start=1001000,  
                end=1011000,
                strand="+",
                confidence=0.75,
                evidence=["literature"],
                biotype="protein_coding"
            )
        ]

    def test_resolver_initialization(self, resolver):
        """Test that AI resolver initializes properly"""
        assert resolver is not None
        assert hasattr(resolver, 'conflict_predictor')
        assert hasattr(resolver, 'source_reliability_scores')
        assert hasattr(resolver, 'evidence_weights')
        assert len(resolver.source_reliability_scores) > 0
    
    @pytest.mark.asyncio
    async def test_brca1_conflict_resolution(self, resolver, brca1_sources):
        """Test AI resolution of realistic BRCA1 conflicts"""
        annotation_group = {
            "gene_symbol": "BRCA1",
            "sources": [
                {
                    "name": src.name,
                    "start": src.start,
                    "end": src.end,
                    "strand": src.strand,
                    "version": src.version,
                    "confidence": src.confidence,
                    "evidence": src.evidence,
                    "biotype": src.biotype,
                    "description": src.description
                } for src in brca1_sources
            ]
        }
        
        resolution = await resolver.resolve_conflicts(annotation_group, "ai_weighted", 0.8)

        assert resolution.gene_symbol == "BRCA1"
        assert resolution.confidence_score > 0.8  
        assert resolution.status == ResolutionStatus.RESOLVED
        assert len(resolution.contributing_sources) == 3
        assert "resolved_coordinates" in resolution.to_dict()
        assert resolution.processing_time_ms > 0

        coords = resolution.resolved_coordinates
        assert 43000000 < coords["start"] < 43200000
        assert 43000000 < coords["end"] < 43200000
        assert coords["start"] < coords["end"]
    
    @pytest.mark.asyncio
    async def test_high_conflict_resolution(self, resolver, high_conflict_sources):
        """Test AI handling of high-conflict scenarios"""
        annotation_group = {
            "gene_symbol": "HIGH_CONFLICT_GENE",
            "sources": [
                {
                    "name": src.name,
                    "start": src.start, 
                    "end": src.end,
                    "strand": src.strand,
                    "confidence": src.confidence,
                    "evidence": src.evidence,
                    "biotype": src.biotype
                } for src in high_conflict_sources
            ]
        }
        
        resolution = await resolver.resolve_conflicts(annotation_group, "ai_weighted", 0.8)

        assert resolution.gene_symbol == "HIGH_CONFLICT_GENE"
        assert resolution.status in [ResolutionStatus.MANUAL_REVIEW, ResolutionStatus.IRRECONCILABLE]
        assert len(resolution.conflict_types) > 0
        assert ConflictType.COORDINATE_MISMATCH in resolution.conflict_types
        assert ConflictType.STRAND_INCONSISTENCY in resolution.conflict_types
        assert ConflictType.BIOTYPE_DISAGREEMENT in resolution.conflict_types

        assert len(resolution.manual_review_notes) > 0
        assert len(resolution.ai_reasoning) > 50 
    
    @pytest.mark.asyncio
    async def test_consensus_voting_strategy(self, resolver, brca1_sources):
        """Test consensus voting resolution strategy"""
        annotation_group = {
            "gene_symbol": "BRCA1",
            "sources": [
                {
                    "name": src.name,
                    "start": src.start,
                    "end": src.end,
                    "strand": src.strand,
                    "confidence": src.confidence,
                    "evidence": src.evidence
                } for src in brca1_sources
            ]
        }
        
        resolution = await resolver.resolve_conflicts(
            annotation_group, "consensus_voting", 0.7
        )
        
        assert resolution.resolution_method == "consensus_voting"
        assert resolution.confidence_score > 0.5
        assert "consensus_sources" in resolution.resolved_coordinates or len(resolution.contributing_sources) > 0
    
    @pytest.mark.asyncio
    async def test_evidence_based_strategy(self, resolver, brca1_sources):
        """Test evidence-based resolution strategy"""
        annotation_group = {
            "gene_symbol": "BRCA1",
            "sources": [
                {
                    "name": src.name,
                    "start": src.start,
                    "end": src.end,
                    "strand": src.strand,
                    "confidence": src.confidence,
                    "evidence": src.evidence
                } for src in brca1_sources
            ]
        }
        
        resolution = await resolver.resolve_conflicts(
            annotation_group, "evidence_based", 0.8
        )
        
        assert resolution.resolution_method == "evidence_based"
        assert "GENCODE" in resolution.contributing_sources
        assert resolution.confidence_score > 0.9  
    
    def test_conflict_detection(self, resolver):
        """Test conflict detection capabilities"""
        sources = [
            AnnotationSource(name="A", start=1000, end=2000, strand="+"),
            AnnotationSource(name="B", start=1500, end=2500, strand="-")  
        ]
        
        conflicts = resolver._detect_conflicts_in_sources(sources)
        
        assert ConflictType.STRAND_INCONSISTENCY in conflicts
    
    def test_weighted_coordinates_calculation(self, resolver, brca1_sources):
        """Test weighted coordinate calculation"""
        weighted_coords = resolver._calculate_weighted_coordinates(brca1_sources)
        
        assert "start" in weighted_coords
        assert "end" in weighted_coords
        assert "strand" in weighted_coords
        assert "source_weights" in weighted_coords

        assert 43000000 < weighted_coords["start"] < 43200000
        assert 43000000 < weighted_coords["end"] < 43200000
        assert weighted_coords["start"] < weighted_coords["end"]

        weights = weighted_coords["source_weights"]
        assert weights["GENCODE"] > weights["Ensembl"]
        assert weights["GENCODE"] > weights["RefSeq"]
    
    def test_evidence_scoring(self, resolver):
        """Test evidence strength scoring"""
        strong_evidence_source = AnnotationSource(
            name="GENCODE",
            start=1000,
            end=2000,
            confidence=0.95,
            evidence=["experimental", "literature", "manual_annotation"]
        )
        
        weak_evidence_source = AnnotationSource(
            name="Unknown",
            start=1000,
            end=2000,
            confidence=0.75,
            evidence=["computational"]
        )
        
        strong_score = resolver._calculate_evidence_score(strong_evidence_source)
        weak_score = resolver._calculate_evidence_score(weak_evidence_source)
        
        assert strong_score > weak_score
        assert strong_score > 0.9
        assert weak_score < 0.8
    
    @pytest.mark.asyncio
    async def test_conflict_analytics_generation(self, resolver):
        """Test comprehensive analytics generation"""
        mock_results = [
            {
                "gene_symbol": "GENE1",
                "resolution": {
                    "status": "resolved",
                    "confidence_score": 0.95,
                    "contributing_sources": ["Ensembl", "GENCODE"],
                    "conflict_types": ["coordinate_mismatch"]
                },
                "processing_time_ms": 150
            },
            {
                "gene_symbol": "GENE2", 
                "resolution": {
                    "status": "manual_review",
                    "confidence_score": 0.65,
                    "contributing_sources": ["RefSeq"],
                    "conflict_types": ["strand_inconsistency", "biotype_disagreement"]
                },
                "processing_time_ms": 200
            }
        ]
        
        analytics = await resolver.generate_conflict_analytics(mock_results)
        
        assert "resolution_success_rate" in analytics
        assert "average_confidence" in analytics
        assert "source_performance" in analytics
        assert "best_performing_source" in analytics
        assert "conflict_pattern_frequency" in analytics
        assert "ai_recommendations" in analytics
        
        assert analytics["resolution_success_rate"] == 50.0 
        assert analytics["average_confidence"] == 0.8  
    
    def test_edge_cases(self, resolver):
        """Test edge cases and error handling"""

        empty_group = {"gene_symbol": "EMPTY", "sources": []}
        
        async def test_empty():
            resolution = await resolver.resolve_conflicts(empty_group)
            assert resolution.status == ResolutionStatus.INSUFFICIENT_DATA
            assert len(resolution.manual_review_notes) > 0

        single_source_group = {
            "gene_symbol": "SINGLE",
            "sources": [{"name": "Only", "start": 1000, "end": 2000}]
        }
        
        async def test_single():
            resolution = await resolver.resolve_conflicts(single_source_group)
            assert resolution.status == ResolutionStatus.INSUFFICIENT_DATA

        asyncio.run(test_empty())
        asyncio.run(test_single())
    
    def test_performance_benchmarks(self, resolver, brca1_sources):
        """Test performance benchmarks for research-grade requirements"""

        large_dataset = []
        for i in range(50): 
            gene_sources = []
            for src in brca1_sources:
                gene_sources.append({
                    "name": src.name,
                    "start": src.start + i * 1000, 
                    "end": src.end + i * 1000,
                    "strand": src.strand,
                    "confidence": src.confidence,
                    "evidence": src.evidence
                })
            
            large_dataset.append({
                "gene_symbol": f"GENE_{i:03d}",
                "sources": gene_sources
            })
        
        async def performance_test():
            start_time = asyncio.get_event_loop().time()
            
            resolutions = []
            for gene_data in large_dataset[:10]: 
                resolution = await resolver.resolve_conflicts(gene_data)
                resolutions.append(resolution)
            
            end_time = asyncio.get_event_loop().time()
            
            processing_time = end_time - start_time
            genes_per_second = 10 / processing_time

            assert genes_per_second > 5  
            assert processing_time < 5  

            high_confidence_count = sum(1 for r in resolutions if r.confidence_score > 0.8)
            assert high_confidence_count >= 8  
            
            print(f"Performance: {genes_per_second:.2f} genes/second")
            print(f"High confidence results: {high_confidence_count}/10")
        
        asyncio.run(performance_test())

class TestIntegrationScenarios:
    """Test real-world genomic scenarios"""
    
    @pytest.mark.asyncio
    async def test_clinical_variant_scenario(self):
        """Test clinically relevant variant annotation conflicts"""
        resolver = AIConflictResolver()

        clinical_conflict = {
            "gene_symbol": "BRCA1",
            "sources": [
                {
                    "name": "ClinVar",
                    "start": 43045802,  
                    "end": 43045803,
                    "confidence": 0.98,
                    "evidence": ["experimental", "literature", "clinical"],
                    "description": "Pathogenic variant"
                },
                {
                    "name": "COSMIC", 
                    "start": 43045803,  
                    "end": 43045804,
                    "confidence": 0.93,
                    "evidence": ["experimental", "literature"],
                    "description": "Somatic mutation"
                }
            ]
        }
        
        resolution = await resolver.resolve_conflicts(clinical_conflict, "ai_weighted", 0.9)

        assert resolution.confidence_score > 0.9
        assert resolution.status == ResolutionStatus.RESOLVED
        assert "clinical" in str(resolution.evidence_summary)
    
    def test_large_scale_batch_processing(self):
        """Test batch processing capabilities for large genomic studies"""
        resolver = AIConflictResolver()
        
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])