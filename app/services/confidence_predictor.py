"""
ML-Based Confidence Predictor - FIXED
"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed")


class ConfidencePredictor:
    """Predict liftover confidence using ML - FIXED"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(settings.MODEL_DIR / "confidence_model.pkl")
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = [
            "chain_score", "chain_count", "chain_gap_size",
            "gc_content", "repeat_density", "low_complexity",
            "sv_overlap", "segdup_overlap", "assembly_gap_distance",
            "historical_success_rate", "cross_reference_agreement"
        ]
        
        if not HAS_SKLEARN:
            logger.error("Cannot use ConfidencePredictor without scikit-learn")
            return
        
        # Try to load existing model
        self._load_model()
        
        # If no model, create untrained one
        if self.model is None:
            self._initialize_model()
    
    def _load_model(self) -> bool:
        """Load trained model from disk"""
        model_file = Path(self.model_path)
        
        if not model_file.exists():
            logger.info(f"No model file found at {self.model_path}")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_names = saved_data.get('feature_names', self.feature_names)
                self.is_trained = True
            
            logger.info(f"Loaded trained model from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _initialize_model(self):
        """Initialize untrained model - FIXED"""
        if not HAS_SKLEARN:
            return
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("Initialized untrained model")
    
    def predict_confidence(self, features: np.ndarray) -> float:
        """Predict confidence score - FIXED to handle untrained model"""
        if not HAS_SKLEARN or self.model is None:
            return self._heuristic_confidence(features)
        
        if not self.is_trained:
            logger.warning("Model not trained, using heuristics")
            return self._heuristic_confidence(features)
        
        try:
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Validate dimensions
            if features.shape[1] != 11:
                logger.error(f"Wrong feature dimensions: {features.shape[1]}, expected 11")
                return self._heuristic_confidence(features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict probability
            probabilities = self.model.predict_proba(features_scaled)
            confidence = float(probabilities[0, 1])
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._heuristic_confidence(features)
    
    def _heuristic_confidence(self, features: np.ndarray) -> float:
        """Fallback heuristic when ML unavailable - FIXED"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Extract key features by index
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
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """Train model - FIXED"""
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for training")
        
        logger.info(f"Training model on {len(X_train)} examples")
        
        # Initialize scaler and fit
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train base model
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
        
        # Calibrate probabilities
        self.model = CalibratedClassifierCV(
            base_model,
            method='sigmoid',
            cv=5
        )
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_preds = self.model.predict(X_train_scaled)
        train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
        
        metrics = {
            'train_accuracy': float(np.mean(train_preds == y_train)),
            'train_auc': float(roc_auc_score(y_train, train_probs))
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.model.predict(X_val_scaled)
            val_probs = self.model.predict_proba(X_val_scaled)[:, 1]
            
            metrics['val_accuracy'] = float(np.mean(val_preds == y_val))
            metrics['val_auc'] = float(roc_auc_score(y_val, val_probs))
        
        logger.info(f"Training complete - AUC: {metrics['train_auc']:.4f}")
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if not self.is_trained:
            logger.warning("Model not trained, nothing to save")
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
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, no feature importance available")
            return None
        
        try:
            # For CalibratedClassifierCV, access the base estimator
            if hasattr(self.model, 'calibrated_classifiers_'):
                base_estimator = self.model.calibrated_classifiers_[0].estimator
            else:
                base_estimator = self.model
            
            if hasattr(base_estimator, 'feature_importances_'):
                importances = base_estimator.feature_importances_
                
                importance_dict = {}
                for i, importance in enumerate(importances):
                    if i < len(self.feature_names):
                        importance_dict[self.feature_names[i]] = float(importance)
                
                return importance_dict
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None

def predict_batch(self, features_list: List[np.ndarray]) -> List[float]:
    """Predict confidence scores for batch of features"""
    return [self.predict_confidence(features) for features in features_list]

def interpret_confidence(self, confidence: float) -> Dict[str, Any]:
    """
    Interpret ML confidence score with clinical recommendations
        
    Args:
        confidence: Confidence score between 0 and 1
            
    Returns:
        Dictionary with interpretation details
    """
    # Normalize to ensure it's between 0 and 1
    confidence = float(confidence)
    confidence = max(0.0, min(1.0, confidence))
        
    if confidence >= 0.90:
            level = "VERY_HIGH"
            interpretation = "Liftover highly reliable for clinical use"
            recommendation = "Suitable for clinical-grade applications"
            color = "green"
    elif confidence >= 0.70:
            level = "HIGH"
            interpretation = "Liftover reliable for research use"
            recommendation = "Suitable for research applications with standard validation"
            color = "blue"
    elif confidence >= 0.50:
            level = "MODERATE"
            interpretation = "Liftover acceptable but requires verification"
            recommendation = "Verify with additional validation methods"
            color = "yellow"
    else:
            level = "LOW"
            interpretation = "Liftover uncertain and requires manual review"
            recommendation = "Manual review and alternative validation required"
            color = "red"
    return {
            'confidence_score': confidence,
            'confidence_level': level,
            'interpretation': interpretation,
            'recommendation': recommendation,
            'color_code': color,
            'threshold_clinical': confidence >= 0.90,
            'threshold_research': confidence >= 0.70,
            'threshold_exploratory': confidence >= 0.50
        }