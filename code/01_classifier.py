#!/usr/bin/env python3
"""
Counterfeit Risk via Fake Reviews - File 1: Classifier Training
Trains fake-review classifier on Kaggle data and scores Trustpilot reviews.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_versions():
    """Log package versions for reproducibility."""
    import sklearn
    import pandas as pd
    import numpy as np
    
    logger.info("Package versions:")
    logger.info(f"  scikit-learn: {sklearn.__version__}")
    logger.info(f"  pandas: {pd.__version__}")
    logger.info(f"  numpy: {np.__version__}")
    logger.info(f"  random seed: 42")

def clean_text(text: str) -> str:
    """Clean text: strip HTML, URLs, emojis; lowercase; lemmatize; remove stopwords."""
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove emojis and special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def create_tfidf_features(texts: pd.Series, max_features: int = 50000, min_df: int = 5) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Create TF-IDF features with word n-grams (1-2) for memory efficiency."""
    
    # Use only word n-grams to reduce memory usage
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Fit and transform
    features = vectorizer.fit_transform(texts)
    
    return vectorizer, features

def train_models(X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
    """Train multiple models with cross-validation."""
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000
        ),
        'LinearSVM': LinearSVC(
            random_state=42,
            class_weight='balanced'
        ),
        'RandomForest': RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced'
        )
    }
    
    # Try to add XGBoost if available
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=1.0,  # Will be adjusted for class imbalance
            eval_metric='logloss'
        )
        logger.info("XGBoost available - included in training")
    except (ImportError, Exception) as e:
        logger.info(f"XGBoost not available - skipping: {e}")
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"  {name} - CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results

def calibrate_probabilities(model, X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    """Calibrate model probabilities using CalibratedClassifierCV."""
    
    # Choose calibration method based on sample size
    # Use shape[0] for sparse matrices
    n_samples = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
    method = 'isotonic' if n_samples >= 10000 else 'sigmoid'
    
    calibrated_model = CalibratedClassifierCV(
        model, 
        cv=5, 
        method=method,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train, y_train)
    logger.info(f"Calibrated model using {method} method")
    
    return calibrated_model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate model performance with given threshold."""
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    return metrics, y_pred, y_pred_proba

def plot_roc_curves(models_results: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, save_path: Path):
    """Plot ROC curves for all models."""
    
    plt.figure(figsize=(10, 8))
    
    for name, result in models_results.items():
        if hasattr(result['model'], 'predict_proba'):
            y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curves saved to {save_path}")

def plot_pr_curves(models_results: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, save_path: Path):
    """Plot Precision-Recall curves for all models."""
    
    plt.figure(figsize=(10, 8))
    
    for name, result in models_results.items():
        if hasattr(result['model'], 'predict_proba'):
            y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.plot(recall, precision, label=f'{name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curves saved to {save_path}")

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Train fake review classifier')
    parser.add_argument('--kaggle_csv', required=True, help='Path to Kaggle dataset CSV')
    parser.add_argument('--trustpilot_csv', required=True, help='Path to Trustpilot dataset CSV')
    parser.add_argument('--out_dir', default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    (out_dir / 'artifacts').mkdir(exist_ok=True)
    (out_dir / 'outputs').mkdir(exist_ok=True)
    (out_dir / 'reports').mkdir(exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)
    
    logger.info("Starting fake review classifier training...")
    log_versions()
    
    # Load Kaggle dataset
    logger.info(f"Loading Kaggle dataset from {args.kaggle_csv}")
    kaggle_df = pd.read_csv(args.kaggle_csv)
    
    # Check required columns
    required_columns = ['category', 'rating', 'label', 'text_']
    missing_columns = [col for col in required_columns if col not in kaggle_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in Kaggle dataset: {missing_columns}")
    
    logger.info(f"Kaggle dataset loaded: {len(kaggle_df)} reviews")
    logger.info(f"Label distribution: {kaggle_df['label'].value_counts().to_dict()}")
    
    # Load Trustpilot dataset
    logger.info(f"Loading Trustpilot dataset from {args.trustpilot_csv}")
    trustpilot_df = pd.read_csv(args.trustpilot_csv)
    
    # Check Trustpilot columns
    trustpilot_required = ['seller', 'product_category', 'rating', 'text', 'date']
    missing_trustpilot = [col for col in trustpilot_required if col not in trustpilot_df.columns]
    if missing_trustpilot:
        raise ValueError(f"Missing required columns in Trustpilot dataset: {missing_trustpilot}")
    
    logger.info(f"Trustpilot dataset loaded: {len(trustpilot_df)} reviews")
    
    # Preprocess text
    logger.info("Preprocessing text...")
    kaggle_df['text_cleaned'] = kaggle_df['text_'].apply(clean_text)
    trustpilot_df['text_cleaned'] = trustpilot_df['text'].apply(clean_text)
    
    # Create labels: CG = 1 (fake), others = 0 (human)
    y = (kaggle_df['label'] == 'CG').astype(int)
    logger.info(f"Label mapping: CG (fake) = 1, others = 0")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Create TF-IDF features
    logger.info("Creating TF-IDF features...")
    
    # Sample data if too large to prevent memory issues
    if len(kaggle_df) > 20000:
        logger.info(f"Dataset is large ({len(kaggle_df)} samples), sampling 20000 for training...")
        sample_indices = np.random.choice(len(kaggle_df), 20000, replace=False)
        kaggle_sample = kaggle_df.iloc[sample_indices].copy()
        y_sample = y[sample_indices]
    else:
        kaggle_sample = kaggle_df
        y_sample = y
    
    vectorizer, X = create_tfidf_features(kaggle_sample['text_cleaned'])
    
    # Split data (before any feature engineering to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train models
    logger.info("Training models...")
    models_results = train_models(X_train, y_train)
    
    # Select best model based on CV performance
    best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['cv_mean'])
    best_model = models_results[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name} (CV ROC-AUC: {models_results[best_model_name]['cv_mean']:.4f})")
    
    # Calibrate best model
    logger.info("Calibrating best model...")
    calibrated_model = calibrate_probabilities(best_model, X_train, y_train)
    
    # Evaluate calibrated model
    logger.info("Evaluating calibrated model...")
    metrics, y_pred, y_pred_proba = evaluate_model(calibrated_model, X_test, y_test)
    
    logger.info("Final model performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Evaluate at different thresholds
    thresholds = [0.4, 0.5, 0.6]
    threshold_results = {}
    
    for t in thresholds:
        t_metrics, t_pred, _ = evaluate_model(calibrated_model, X_test, y_test, threshold=t)
        threshold_results[f't_{int(t*100)}'] = t_metrics
    
    # Save artifacts
    logger.info("Saving artifacts...")
    
    # Save vectorizer
    joblib.dump(vectorizer, out_dir / 'artifacts' / 'vectorizer.joblib')
    
    # Save calibrated model
    joblib.dump(calibrated_model, out_dir / 'artifacts' / 'model.joblib')
    
    # Save thresholds
    thresholds_dict = {
        "t_default": 0.5,
        "t_lo": 0.4,
        "t_hi": 0.6
    }
    with open(out_dir / 'artifacts' / 'thresholds.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    # Save label mapping
    label_mapping = {"CG": 1, "Human": 0}
    with open(out_dir / 'artifacts' / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save model summary
    model_summary = {}
    for name, result in models_results.items():
        model_summary[name] = {
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std'])
        }
    
    # Add final calibrated model metrics
    model_summary['calibrated_final'] = {k: float(v) for k, v in metrics.items()}
    
    with open(out_dir / 'reports' / 'model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    # Save confusion matrices
    confusion_matrices = {}
    for t in thresholds:
        _, t_pred, _ = evaluate_model(calibrated_model, X_test, y_test, threshold=t)
        cm = confusion_matrix(y_test, t_pred)
        confusion_matrices[f't_{int(t*100)}'] = cm.tolist()
    
    with open(out_dir / 'reports' / 'confusion_matrices.json', 'w') as f:
        json.dump(confusion_matrices, f, indent=2)
    
    # Create plots
    logger.info("Creating plots...")
    plot_roc_curves(models_results, X_test, y_test, out_dir / 'figures' / 'roc_curve.png')
    plot_pr_curves(models_results, X_test, y_test, out_dir / 'figures' / 'pr_curve.png')
    
    # Score Trustpilot reviews
    logger.info("Scoring Trustpilot reviews...")
    
    # Use the same vectorizer (don't refit)
    trustpilot_features = vectorizer.transform(trustpilot_df['text_cleaned'])
    
    # Get probabilities
    trustpilot_fake_probs = calibrated_model.predict_proba(trustpilot_features)[:, 1]
    
    # Create scored dataset
    scored_df = trustpilot_df.copy()
    scored_df['fake_prob'] = trustpilot_fake_probs
    scored_df['fake_pred_t05'] = (trustpilot_fake_probs >= 0.5).astype(int)
    scored_df['fake_pred_t04'] = (trustpilot_fake_probs >= 0.4).astype(int)
    scored_df['fake_pred_t06'] = (trustpilot_fake_probs >= 0.6).astype(int)
    
    # Save scored Trustpilot data
    scored_df.to_csv(out_dir / 'outputs' / 'trustpilot_scored.csv', index=False)
    
    logger.info(f"Scored Trustpilot reviews saved to {out_dir / 'outputs' / 'trustpilot_scored.csv'}")
    logger.info("Classifier training completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFIER TRAINING COMPLETED")
    print("="*60)
    print(f"Best model: {best_model_name}")
    print(f"CV ROC-AUC: {models_results[best_model_name]['cv_mean']:.4f} ± {models_results[best_model_name]['cv_std']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Artifacts saved to: {out_dir / 'artifacts'}")
    print(f"Scored Trustpilot data: {out_dir / 'outputs' / 'trustpilot_scored.csv'}")

if __name__ == "__main__":
    main()
