
import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

# Try to import NLP libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available - semantic features limited")

@dataclass
class SemanticAnnotation:
    """Structured annotation with semantic information"""
    gene_symbol: str
    description: str
    source: str
    biological_process: Optional[List[str]] = None
    molecular_function: Optional[List[str]] = None
    cellular_component: Optional[List[str]] = None
    protein_domains: Optional[List[str]] = None
    synonyms: Optional[List[str]] = None
    confidence: float = 0.8
    
    def __post_init__(self):
        if self.biological_process is None:
            self.biological_process = []
        if self.molecular_function is None:
            self.molecular_function = []
        if self.cellular_component is None:
            self.cellular_component = []
        if self.protein_domains is None:
            self.protein_domains = []
        if self.synonyms is None:
            self.synonyms = []

class BiologicalTermExtractor:
    """Extract and normalize biological terms from text"""
    
    def __init__(self):
        # Common biological term patterns
        self.patterns = {
            'kinase': r'\b\w+\s*kinase\b',
            'receptor': r'\b\w+\s*receptor\b',
            'factor': r'\b\w+\s*factor\b',
            'protein': r'\b\w+\s*protein\b',
            'enzyme': r'\b\w+\s*(?:ase|dehydrogenase|transferase|synthase)\b',
            'domain': r'\b(?:SH[23]|PH|PDZ|WD40|zinc finger|kinase|catalytic)\s*domain\b',
        }
        
        # Biological process keywords
        self.process_keywords = {
            'regulation', 'signaling', 'pathway', 'activation', 'inhibition',
            'metabolism', 'biosynthesis', 'degradation', 'transport', 'binding',
            'transcription', 'translation', 'replication', 'repair', 'apoptosis',
            'differentiation', 'proliferation', 'development', 'growth'
        }
        
        # Molecular function keywords
        self.function_keywords = {
            'kinase', 'phosphatase', 'ligase', 'transferase', 'hydrolase',
            'oxidoreductase', 'isomerase', 'binding', 'activity', 'catalytic',
            'receptor', 'channel', 'transporter', 'regulator'
        }
    
    def extract_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract biological terms from description text"""
        text_lower = text.lower()
        
        extracted = {
            'protein_types': [],
            'processes': [],
            'functions': [],
            'domains': []
        }
        
        # Extract protein types and domains
        for term_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                if term_type == 'domain':
                    extracted['domains'].extend(matches)
                else:
                    extracted['protein_types'].extend(matches)
        
        # Extract process keywords
        words = set(re.findall(r'\b\w+\b', text_lower))
        extracted['processes'] = list(words & self.process_keywords)
        extracted['functions'] = list(words & self.function_keywords)
        
        return extracted
    
    def normalize_term(self, term: str) -> str:
        """Normalize biological term to canonical form"""
        term = term.lower().strip()
        
        # Remove common prefixes/suffixes
        term = re.sub(r'\b(human|mouse|rat|homo sapiens)\b', '', term)
        term = re.sub(r'\s+', ' ', term).strip()
        
        # Standardize common abbreviations
        abbreviations = {
            'dna': 'deoxyribonucleic acid',
            'rna': 'ribonucleic acid',
            'mrna': 'messenger rna',
            'atp': 'adenosine triphosphate',
            'gtp': 'guanosine triphosphate'
        }
        
        for abbrev, full in abbreviations.items():
            if abbrev in term:
                term = term.replace(abbrev, full)
        
        return term

class SemanticReconciliationEngine:
    """
    Main engine for semantic reconciliation of gene annotations.
    
    Reconciles conflicting descriptions by:
    1. Computing semantic similarity using TF-IDF
    2. Extracting and comparing biological terms
    3. Identifying consensus concepts
    4. Generating unified descriptions
    """
    
    def __init__(self):
        self.term_extractor = BiologicalTermExtractor()
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english'
        ) if HAS_SKLEARN else None
        
        # Database credibility scores
        self.source_credibility = {
            'NCBI': 0.95,
            'UniProt': 0.98,
            'Ensembl': 0.93,
            'GENCODE': 0.97,
            'RefSeq': 0.92,
            'HGNC': 0.96,
            'MGI': 0.91
        }
    
    def compute_text_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise semantic similarity between descriptions.
        
        Args:
            texts: List of description texts
        
        Returns:
            Similarity matrix (n x n)
        """
        if not HAS_SKLEARN or len(texts) < 2:
            # Fallback: simple word overlap
            return self._compute_simple_similarity(texts)
        
        try:
            # TF-IDF vectorization
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return self._compute_simple_similarity(texts)
    
    def _compute_simple_similarity(self, texts: List[str]) -> np.ndarray:
        """Fallback similarity using word overlap"""
        n = len(texts)
        similarity = np.zeros((n, n))
        
        # Tokenize texts
        tokenized = [set(re.findall(r'\b\w+\b', t.lower())) for t in texts]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # Jaccard similarity
                    intersection = len(tokenized[i] & tokenized[j])
                    union = len(tokenized[i] | tokenized[j])
                    similarity[i, j] = intersection / union if union > 0 else 0.0
        
        return similarity
    
    def extract_semantic_features(
        self, 
        annotation: SemanticAnnotation
    ) -> Dict[str, any]:
        """Extract semantic features from annotation"""
        
        # Extract terms from description
        extracted_terms = self.term_extractor.extract_terms(annotation.description)
        
        # Combine with existing structured data
        features = {
            'description': annotation.description,
            'protein_types': extracted_terms['protein_types'],
            'biological_processes': (
                annotation.biological_process + extracted_terms['processes']
            ),
            'molecular_functions': (
                annotation.molecular_function + extracted_terms['functions']
            ),
            'protein_domains': (
                annotation.protein_domains + extracted_terms['domains']
            ),
            'source': annotation.source,
            'credibility': self.source_credibility.get(annotation.source, 0.5)
        }
        
        return features
    
    def reconcile_annotations(
        self,
        gene_symbol: str,
        annotations: List[SemanticAnnotation]
    ) -> Dict:
        """
        Reconcile multiple annotations for a gene.
        
        Args:
            gene_symbol: Gene identifier
            annotations: List of annotations from different sources
        
        Returns:
            Dict with reconciled information and confidence metrics
        """
        if len(annotations) == 0:
            return {
                'gene_symbol': gene_symbol,
                'status': 'no_data',
                'reconciled_description': None
            }
        
        if len(annotations) == 1:
            return {
                'gene_symbol': gene_symbol,
                'status': 'single_source',
                'reconciled_description': annotations[0].description,
                'source': annotations[0].source,
                'confidence': annotations[0].confidence
            }
        
        # Extract features from all annotations
        features_list = [self.extract_semantic_features(ann) for ann in annotations]
        
        # Compute description similarity
        descriptions = [ann.description for ann in annotations]
        similarity_matrix = self.compute_text_similarity(descriptions)
        
        # Find consensus description (most similar to others)
        avg_similarity = similarity_matrix.mean(axis=1)
        consensus_idx = np.argmax(avg_similarity)
        consensus_annotation = annotations[consensus_idx]
        
        # Aggregate biological terms
        all_processes = []
        all_functions = []
        all_domains = []
        
        for features in features_list:
            all_processes.extend(features['biological_processes'])
            all_functions.extend(features['molecular_functions'])
            all_domains.extend(features['protein_domains'])
        
        # Count term frequencies
        process_counts = Counter(all_processes)
        function_counts = Counter(all_functions)
        domain_counts = Counter(all_domains)
        
        # Select terms appearing in multiple sources
        min_sources = max(2, len(annotations) // 2)
        
        consensus_processes = [
            term for term, count in process_counts.items() 
            if count >= min_sources
        ]
        consensus_functions = [
            term for term, count in function_counts.items() 
            if count >= min_sources
        ]
        consensus_domains = [
            term for term, count in domain_counts.items() 
            if count >= min_sources
        ]
        
        # Calculate reconciliation confidence
        text_similarity = float(avg_similarity[consensus_idx])
        source_agreement = len(consensus_processes) / max(len(process_counts), 1)
        credibility = self.source_credibility.get(consensus_annotation.source, 0.5)
        
        reconciliation_confidence = (
            text_similarity * 0.4 +
            source_agreement * 0.3 +
            credibility * 0.3
        )
        
        # Generate enhanced description
        enhanced_description = self._generate_enhanced_description(
            consensus_annotation.description,
            consensus_processes,
            consensus_functions,
            consensus_domains
        )
        
        return {
            'gene_symbol': gene_symbol,
            'status': 'reconciled',
            'reconciled_description': consensus_annotation.description,
            'enhanced_description': enhanced_description,
            'consensus_source': consensus_annotation.source,
            'contributing_sources': [ann.source for ann in annotations],
            'n_sources': len(annotations),
            'confidence': float(reconciliation_confidence),
            'semantic_similarity': float(text_similarity),
            'source_agreement': float(source_agreement),
            'consensus_biological_processes': consensus_processes,
            'consensus_molecular_functions': consensus_functions,
            'consensus_protein_domains': consensus_domains,
            'all_sources_similarity': similarity_matrix.tolist(),
            'recommendation': self._generate_recommendation(reconciliation_confidence)
        }
    
    def _generate_enhanced_description(
        self,
        base_description: str,
        processes: List[str],
        functions: List[str],
        domains: List[str]
    ) -> str:
        """Generate enhanced description with consensus terms"""
        
        enhanced = base_description
        
        if processes:
            process_str = ', '.join(processes[:3])
            enhanced += f" Biological processes: {process_str}."
        
        if functions:
            function_str = ', '.join(functions[:3])
            enhanced += f" Molecular functions: {function_str}."
        
        if domains:
            domain_str = ', '.join(domains[:3])
            enhanced += f" Protein domains: {domain_str}."
        
        return enhanced
    
    def _generate_recommendation(self, confidence: float) -> str:
        """Generate recommendation based on confidence"""
        if confidence >= 0.9:
            return "High confidence reconciliation - suitable for automated use"
        elif confidence >= 0.75:
            return "Moderate confidence - recommended for most applications"
        elif confidence >= 0.6:
            return "Acceptable confidence - consider manual review for critical applications"
        else:
            return "Low confidence - manual curation recommended"
    
    def batch_reconcile(
        self,
        gene_annotations: Dict[str, List[SemanticAnnotation]]
    ) -> Dict[str, Dict]:
        """
        Reconcile annotations for multiple genes.
        
        Args:
            gene_annotations: Dict mapping gene symbols to annotation lists
        
        Returns:
            Dict mapping gene symbols to reconciliation results
        """
        results = {}
        
        for gene_symbol, annotations in gene_annotations.items():
            try:
                result = self.reconcile_annotations(gene_symbol, annotations)
                results[gene_symbol] = result
            except Exception as e:
                logger.error(f"Reconciliation failed for {gene_symbol}: {e}")
                results[gene_symbol] = {
                    'gene_symbol': gene_symbol,
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def generate_reconciliation_report(self, results: Dict[str, Dict]) -> Dict:
        """Generate summary report for batch reconciliation"""
        
        total_genes = len(results)
        reconciled = sum(1 for r in results.values() if r.get('status') == 'reconciled')
        single_source = sum(1 for r in results.values() if r.get('status') == 'single_source')
        errors = sum(1 for r in results.values() if r.get('status') == 'error')
        
        confidences = [
            r['confidence'] for r in results.values() 
            if 'confidence' in r
        ]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        high_conf = sum(1 for c in confidences if c >= 0.9)
        med_conf = sum(1 for c in confidences if 0.75 <= c < 0.9)
        low_conf = sum(1 for c in confidences if c < 0.75)
        
        return {
            'summary': {
                'total_genes': total_genes,
                'reconciled': reconciled,
                'single_source': single_source,
                'errors': errors,
                'success_rate': (reconciled / total_genes * 100) if total_genes > 0 else 0
            },
            'confidence_distribution': {
                'high_confidence': high_conf,
                'medium_confidence': med_conf,
                'low_confidence': low_conf,
                'average_confidence': round(avg_confidence, 3)
            },
            'quality_metrics': {
                'suitable_for_automated_use': high_conf,
                'recommended_for_use': high_conf + med_conf,
                'requires_manual_review': low_conf
            }
        }