"""
ML-Based Confidence Predictor for Liftover Reliability

This module provides machine learning models to predict whether
a liftover result is reliable based on genomic features.

Models:
- Random Forest (baseline)
- Gradient Boosting (production)
- Calibrated probabilities for confidence scores
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed - confidence prediction disabled")

class ConfidencePredictor:
    """
    Predict liftover confidence using machine learning.
    
    The model predicts probability that a liftover result is correct
    based on genomic features extracted from the region.
    
    Training:
        - Positive examples: Validated correct liftover (NCBI, Ensembl agreement)
        - Negative examples: Known incorrect liftover, failed mappings
        
    Features:
        - Chain file quality (score, gap size)
        - Sequence context (GC content, repeats)
        - Structural features (SVs, segmental duplications)
        - Historical success rates
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "app/data/models/confidence_model.pkl"
        self.model = None
        self.scaler = None
        self.feature_names = [
            "chain_score",
            "chain_count",
            "chain_gap_size",
            "gc_content",
            "repeat_density",
            "low_complexity",
            "sv_overlap",
            "segdup_overlap",
            "assembly_gap_distance",
            "historical_success_rate",
            "cross_reference_agreement"
        ]
        
        if not HAS_SKLEARN:
            logger.error("Cannot use ConfidencePredictor without scikit-learn")
            return
        
        # Try to load existing model
        self._load_model()
        
        # If no model exists, create default
        if self.model is None:
            logger.warning("No trained model found - using default heuristics")
            self._create_default_model()
    
    def _load_model(self) -> bool:
        """Load trained model from disk"""
        model_file = Path(self.model_path)
        
        if not model_file.exists():
            logger.info(f"Model file not found: {self.model_path}")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_names = saved_data.get('feature_names', self.feature_names)
            
            logger.info(f"Loaded model from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_default_model(self):
        """Create simple heuristic-based model"""
        if not HAS_SKLEARN:
            return
        
        # Create a simple gradient boosting classifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        
        logger.info("Created default model (not trained on real data)")
    
    def predict_confidence(self, features: np.ndarray) -> float:
        """
        Predict confidence score for liftover.
        
        Args:
            features: Numpy array of genomic features (11 dimensions)
            
        Returns:
            Confidence score in [0, 1] where:
            - 1.0 = Very reliable liftover
            - 0.5 = Uncertain, manual review recommended
            - 0.0 = Likely incorrect liftover
        """
        if not HAS_SKLEARN or self.model is None:
            # Fallback: simple heuristic
            return self._heuristic_confidence(features)
        
        try:
            # Reshape for single prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict probability (calibrated)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Return probability of positive class (reliable liftover)
            confidence = probabilities[0, 1]
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._heuristic_confidence(features)
    
    def _heuristic_confidence(self, features: np.ndarray) -> float:
        """
        Fallback heuristic when ML model unavailable.
        
        Uses simple rules based on known risk factors.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Extract key features (indices match feature_names)
        chain_score = features[0, 0]
        repeat_density = features[0, 4]
        sv_overlap = features[0, 6]
        segdup_overlap = features[0, 7]
        historical_success = features[0, 9]
        cross_ref_agreement = features[0, 10]
        
        # Start with base confidence
        confidence = 0.8
        
        # Adjust based on risk factors
        if chain_score < 0.5:
            confidence -= 0.3
        elif chain_score > 0.95:
            confidence += 0.1
        
        if repeat_density > 0.7:
            confidence -= 0.2
        
        if sv_overlap > 0.5:
            confidence -= 0.15
        
        if segdup_overlap > 0.5:
            confidence -= 0.25
        
        # Weight by historical success
        confidence = confidence * (0.5 + 0.5 * historical_success)
        
        # Weight by cross-reference agreement
        confidence = confidence * (0.7 + 0.3 * cross_ref_agreement)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def predict_batch(self, features_list: List[np.ndarray]) -> List[float]:
        """
        Predict confidence for multiple liftover results.
        
        Args:
            features_list: List of feature arrays
            
        Returns:
            List of confidence scores
        """
        # Stack features into matrix
        features_matrix = np.vstack(features_list)
        
        if not HAS_SKLEARN or self.model is None:
            return [self._heuristic_confidence(f.reshape(1, -1)) 
                    for f in features_matrix]
        
        try:
            # Scale all at once
            features_scaled = self.scaler.transform(features_matrix)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(features_scaled)
            
            # Extract confidence scores
            confidences = probabilities[:, 1]
            
            return [float(np.clip(c, 0.0, 1.0)) for c in confidences]
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [self._heuristic_confidence(f.reshape(1, -1)) 
                    for f in features_matrix]
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train confidence prediction model.
        
        Args:
            X_train: Training features (N x 11)
            y_train: Training labels (N,) - 1 for correct, 0 for incorrect
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for training")
        
        logger.info(f"Training model on {len(X_train)} examples")
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train gradient boosting model
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        base_model.fit(X_train_scaled, y_train)
        
        # Calibrate probabilities for better confidence estimates
        self.model = CalibratedClassifierCV(
            base_model,
            method='sigmoid',
            cv=5
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_preds = self.model.predict(X_train_scaled)
        train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
        
        metrics = {
            'train_accuracy': np.mean(train_preds == y_train),
            'train_auc': roc_auc_score(y_train, train_probs)
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.model.predict(X_val_scaled)
            val_probs = self.model.predict_proba(X_val_scaled)[:, 1]
            
            metrics['val_accuracy'] = np.mean(val_preds == y_val)
            metrics['val_auc'] = roc_auc_score(y_val, val_probs)
            
            logger.info(f"Validation AUC: {metrics['val_auc']:.4f}")
        
        logger.info(f"Training complete - AUC: {metrics['train_auc']:.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model to disk"""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        save_path = path or self.model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not HAS_SKLEARN or self.model is None:
            return {}
        
        try:
            # Get base estimator (before calibration)
            if hasattr(self.model, 'base_estimator'):
                base_model = self.model.base_estimator
            else:
                base_model = self.model
            
            if hasattr(base_model, 'feature_importances_'):
                importances = base_model.feature_importances_
                
                return {
                    name: float(imp)
                    for name, imp in zip(self.feature_names, importances)
                }
        except Exception as e:
            logger.error(f"Could not extract feature importance: {e}")
        
        return {}
    
    def interpret_confidence(self, confidence: float) -> Dict:
        """
        Provide human-readable interpretation of confidence score.
        
        Args:
            confidence: Confidence score in [0, 1]
            
        Returns:
            Dictionary with interpretation and recommendations
        """
        if confidence >= 0.95:
            level = "VERY_HIGH"
            interpretation = "Liftover highly reliable - safe for automated use"
            recommendation = "Proceed with confidence"
            color = "green"
            
        elif confidence >= 0.85:
            level = "HIGH"
            interpretation = "Liftover reliable - suitable for most applications"
            recommendation = "Acceptable for publication"
            color = "green"
            
        elif confidence >= 0.70:
            level = "MODERATE"
            interpretation = "Liftover likely correct but has some uncertainty"
            recommendation = "Verify for critical applications"
            color = "yellow"
            
        elif confidence >= 0.50:
            level = "LOW"
            interpretation = "Liftover uncertain - multiple risk factors present"
            recommendation = "Manual verification strongly recommended"
            color = "orange"
            
        else:
            level = "VERY_LOW"
            interpretation = "Liftover likely incorrect or unreliable"
            recommendation = "Do not use - manual curation required"
            color = "red"
        
        return {
            'confidence_score': float(confidence),
            'confidence_level': level,
            'interpretation': interpretation,
            'recommendation': recommendation,
            'color_code': color,
            'threshold_clinical': confidence >= 0.90,
            'threshold_research': confidence >= 0.70,
            'threshold_exploratory': confidence >= 0.50
        }