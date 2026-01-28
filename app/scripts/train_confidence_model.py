#!/usr/bin/env python3
"""
Train ML Confidence Prediction Model

This script trains the gradient boosting model for liftover confidence prediction.

Training Data Sources:
1. Validated NCBI RefSeq coordinates (positive examples)
2. Known problematic regions (negative examples)
3. Historical liftover failures

Usage:
    # Initial training with validation data
    python scripts/train_confidence_model.py --mode initial
    
    # Retrain with additional data
    python scripts/train_confidence_model.py --mode update --data validation_results/
    
    # Evaluate existing model
    python scripts/train_confidence_model.py --mode evaluate
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.error("scikit-learn required: pip install scikit-learn matplotlib")

from app.services.feature_extractor import FeatureExtractor, GenomicFeatures
from app.services.confidence_predictor import ConfidencePredictor
from app.services.validation_engine import ValidationEngine
from app.services.real_liftover import RealLiftoverService


class ModelTrainer:
    """Train and evaluate confidence prediction model"""
    
    def __init__(self, data_dir: str = "./app/data"):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.confidence_predictor = ConfidencePredictor()
        self.liftover_service = RealLiftoverService()
        self.validation_engine = ValidationEngine()
    
    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from validation results.
        
        Returns:
            X: Feature matrix (N x 11)
            y: Labels (N,) - 1 for correct, 0 for incorrect
        """
        logger.info("Generating training data from validation...")
        
        # Run validation to get labeled examples
        records = self.validation_engine.validate_against_ncbi(
            self.liftover_service,
            sample_size=None  # Use all available genes
        )
        
        logger.info(f"Generated {len(records)} validation records")
        
        # Extract features and labels
        X_list = []
        y_list = []
        
        for record in records:
            # Extract features
            features = self.feature_extractor.extract_features(
                record.expected_chrom,
                record.expected_pos,
                record.expected_build,
                "hg19",  # Reverse direction for training
                None
            )
            
            X_list.append(features.to_array())
            
            # Label: 1 if successful (within 100bp), 0 otherwise
            y_list.append(1 if record.success else 0)
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Positive examples: {y.sum()}")
        logger.info(f"Negative examples: {len(y) - y.sum()}")
        
        return X, y
    
    def add_synthetic_negatives(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add synthetic negative examples for problematic regions.
        
        This helps the model learn failure patterns:
        - High repeat density
        - Low chain scores
        - Near assembly gaps
        """
        logger.info("Adding synthetic negative examples...")
        
        n_positives = y.sum()
        n_synthetics = max(100, n_positives // 2)  # Add 50% more negatives
        
        synthetic_X = []
        
        for _ in range(n_synthetics):
            # Create feature vector for problematic region
            features = np.random.rand(11)
            
            # Adjust features to represent failure patterns
            features[0] = np.random.uniform(0.0, 0.5)  # Low chain score
            features[4] = np.random.uniform(0.7, 1.0)  # High repeat density
            features[5] = 1.0  # Low complexity
            features[6] = np.random.choice([0, 1])  # SV overlap
            features[7] = np.random.choice([0, 1])  # Segdup overlap
            features[8] = np.random.uniform(0, 1000)  # Near gap
            features[9] = np.random.uniform(0.0, 0.5)  # Low historical success
            
            synthetic_X.append(features)
        
        synthetic_X = np.vstack(synthetic_X)
        synthetic_y = np.zeros(n_synthetics)
        
        # Combine with real data
        X_combined = np.vstack([X, synthetic_X])
        y_combined = np.concatenate([y, synthetic_y])
        
        logger.info(f"Added {n_synthetics} synthetic negatives")
        logger.info(f"Total training examples: {len(y_combined)}")
        
        return X_combined, y_combined
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """Train gradient boosting model"""
        logger.info("Training confidence prediction model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} examples")
        logger.info(f"Val set: {len(X_val)} examples")
        
        # Train model
        metrics = self.confidence_predictor.train(X_train, y_train, X_val, y_val)
        
        logger.info("Training complete!")
        logger.info(f"  Train AUC: {metrics['train_auc']:.4f}")
        logger.info(f"  Val AUC: {metrics['val_auc']:.4f}")
        
        return metrics
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        # Predict
        confidences = [self.confidence_predictor.predict_confidence(x) for x in X]
        predictions = [1 if c >= 0.5 else 0 for c in confidences]
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == y)
        auc = roc_auc_score(y, confidences)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"AUC-ROC: {auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=['Incorrect', 'Correct']))
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        importance = self.confidence_predictor.get_feature_importance()
        if importance:
            print("\nFeature Importance:")
            for feature, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
                print(f"  {feature}: {imp:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'feature_importance': importance
        }
    
    def save_model(self):
        """Save trained model"""
        output_path = self.models_dir / "confidence_model.pkl"
        self.confidence_predictor.save_model(str(output_path))
        logger.info(f"Model saved to {output_path}")
    
    def run_initial_training(self):
        """Run complete initial training pipeline"""
        logger.info("=" * 60)
        logger.info("Initial Model Training")
        logger.info("=" * 60)
        
        # 1. Generate training data
        X, y = self.generate_training_data()
        
        # 2. Add synthetic negatives
        X, y = self.add_synthetic_negatives(X, y)
        
        # 3. Train model
        self.train_model(X, y)
        
        # 4. Evaluate
        self.evaluate_model(X, y)
        
        # 5. Save
        self.save_model()
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train ML confidence prediction model"
    )
    
    parser.add_argument(
        '--mode',
        choices=['initial', 'update', 'evaluate'],
        default='initial',
        help='Training mode'
    )
    
    parser.add_argument(
        '--data-dir',
        default='./app/data',
        help='Data directory'
    )
    
    args = parser.parse_args()
    
    if not HAS_SKLEARN:
        logger.error("scikit-learn required: pip install scikit-learn")
        return
    
    trainer = ModelTrainer(args.data_dir)
    
    if args.mode == 'initial':
        trainer.run_initial_training()
    
    elif args.mode == 'evaluate':
        # Load existing model and evaluate
        X, y = trainer.generate_training_data()
        trainer.evaluate_model(X, y)
    
    else:
        logger.error("Mode not implemented yet")


if __name__ == "__main__":
    main()