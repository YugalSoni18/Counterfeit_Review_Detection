#!/usr/bin/env python3
"""
Counterfeit Risk via Fake Reviews - File 2: Counterfeit Risk Analysis
Analyzes scored Trustpilot reviews to identify counterfeit product hotspots.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('counterfeit_risk_analysis.log'),
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

def load_crs_weights(weights_path: Path = None) -> Dict[str, float]:
    """Load CRS weights from JSON or use defaults."""
    if weights_path and weights_path.exists():
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        logger.info(f"Loaded custom CRS weights: {weights}")
    else:
        weights = {
            'fake_ratio': 0.45,
            'rating_polarization': 0.20,
            'review_burstiness': 0.15,
            'sentiment_rating_gap': 0.10,
            'lexical_redundancy': 0.10
        }
        logger.info(f"Using default CRS weights: {weights}")
    
    return weights

def clean_text_for_similarity(text: str) -> str:
    """Clean text for TF-IDF similarity calculation (same as file 1)."""
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

def calculate_fake_review_features(group: pd.DataFrame) -> Dict[str, float]:
    """Calculate fake review prevalence features for an entity."""
    return {
        'fake_ratio': group['fake_prob'].mean(),
        'fake_share_t05': (group['fake_pred_t05'] == 1).mean(),
        'n_reviews': len(group)
    }

def calculate_rating_features(group: pd.DataFrame) -> Dict[str, float]:
    """Calculate rating shape and burst features for an entity."""
    ratings = group['rating'].dropna()
    
    if len(ratings) == 0:
        return {
            'rating_polarization': 0.0,
            'rating_variance': 0.0
        }
    
    # Rating polarization: P(1★) + P(5★)
    rating_polarization = ((ratings == 1).sum() + (ratings == 5).sum()) / len(ratings)
    
    # Rating variance
    rating_variance = ratings.var()
    
    return {
        'rating_polarization': rating_polarization,
        'rating_variance': rating_variance
    }

def calculate_burst_features(group: pd.DataFrame) -> Dict[str, float]:
    """Calculate review burstiness features for an entity."""
    if 'date' not in group.columns:
        return {
            'review_burstiness': 1.0,
            'first_30d_burst': 0.0
        }
    
    # Parse dates
    try:
        dates = pd.to_datetime(group['date'], errors='coerce').dropna()
    except:
        return {
            'review_burstiness': 1.0,
            'first_30d_burst': 0.0
        }
    
    if len(dates) == 0:
        return {
            'review_burstiness': 1.0,
            'first_30d_burst': 0.0
        }
    
    # Daily review counts
    daily_counts = dates.dt.date.value_counts()
    
    # Review burstiness: (max reviews/day) / (median reviews/day + ε)
    max_daily = daily_counts.max()
    median_daily = daily_counts.median()
    review_burstiness = max_daily / (median_daily + 1e-6)
    
    # Winsorize at 99th percentile
    review_burstiness = min(review_burstiness, daily_counts.quantile(0.99))
    
    # First 30 days burst
    min_date = dates.min()
    first_30d = dates[dates <= min_date + timedelta(days=30)]
    first_30d_burst = len(first_30d) / len(dates)
    
    return {
        'review_burstiness': review_burstiness,
        'first_30d_burst': first_30d_burst
    }

def calculate_sentiment_features(group: pd.DataFrame) -> Dict[str, float]:
    """Calculate sentiment and style inconsistency features for an entity."""
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each review
    sentiments = []
    for text in group['text'].dropna():
        sentiment = analyzer.polarity_scores(str(text))
        sentiments.append(sentiment['compound'])
    
    if not sentiments:
        return {
            'sentiment_rating_gap': 0.0,
            'lexical_redundancy': 0.0
        }
    
    # Sentiment rating gap: mean(|scaled_sentiment - scaled_rating|)
    # Scale both to 0-1 range
    scaled_sentiments = [(s + 1) / 2 for s in sentiments]  # -1..1 -> 0..1
    scaled_ratings = [(r - 1) / 4 for r in group['rating'].dropna()]  # 1..5 -> 0..1
    
    # Align lengths
    min_len = min(len(scaled_sentiments), len(scaled_ratings))
    if min_len > 0:
        sentiment_rating_gap = np.mean([
            abs(s - r) for s, r in zip(scaled_sentiments[:min_len], scaled_ratings[:min_len])
        ])
    else:
        sentiment_rating_gap = 0.0
    
    # Lexical redundancy: average pairwise cosine similarity of TF-IDF vectors
    # For efficiency, sample up to 500 reviews
    texts = group['text'].dropna().head(500)
    if len(texts) < 2:
        lexical_redundancy = 0.0
    else:
        # Clean texts
        cleaned_texts = [clean_text_for_similarity(text) for text in texts]
        
        # Create TF-IDF vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # Calculate pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(tfidf_matrix)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        
        if len(upper_triangle) > 0:
            lexical_redundancy = upper_triangle.mean()
        else:
            lexical_redundancy = 0.0
    
    return {
        'sentiment_rating_gap': sentiment_rating_gap,
        'lexical_redundancy': lexical_redundancy
    }

def calculate_crs(entity_features: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    """Calculate Counterfeit Risk Score (CRS) for entities."""
    
    # Features to use in CRS
    crs_features = [
        'fake_ratio', 'rating_polarization', 'review_burstiness',
        'sentiment_rating_gap', 'lexical_redundancy'
    ]
    
    # Standardize features within each product_category
    crs_scores = []
    
    for category in entity_features['product_category'].unique():
        category_mask = entity_features['product_category'] == category
        category_data = entity_features[category_mask].copy()
        
        if len(category_data) < 2:
            # If only one entity in category, use global standardization
            category_data = entity_features.copy()
        
        # Select features for CRS
        feature_matrix = category_data[crs_features].fillna(0)
        
        # Standardize (z-score)
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Calculate weighted sum
        crs_raw = np.zeros(len(feature_matrix_scaled))
        for i, feature in enumerate(crs_features):
            if feature in weights:
                crs_raw += weights[feature] * feature_matrix_scaled[:, i]
        
        # Map to 0-100 using standard normal CDF
        crs = 100 * norm.cdf(crs_raw)
        
        # Store scores
        if category_mask.sum() == 1:
            # Single entity in category
            crs_scores.extend(crs)
        else:
            # Multiple entities in category
            crs_scores.extend(crs)
    
    return np.array(crs_scores)

def detect_anomalies(entity_features: pd.DataFrame) -> np.ndarray:
    """Detect anomalies using Isolation Forest."""
    
    # Select features for anomaly detection
    anomaly_features = [
        'fake_ratio', 'rating_polarization', 'review_burstiness',
        'sentiment_rating_gap', 'lexical_redundancy'
    ]
    
    feature_matrix = entity_features[anomaly_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(feature_matrix_scaled)
    
    # Convert to positive scores (higher = more anomalous)
    anomaly_scores = -anomaly_scores  # -1 becomes 1, 1 becomes -1
    
    return anomaly_scores

def cluster_entities(entity_features: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
    """Cluster entities using K-means."""
    
    # Select features for clustering
    cluster_features = [
        'fake_ratio', 'rating_polarization', 'review_burstiness',
        'sentiment_rating_gap', 'lexical_redundancy'
    ]
    
    feature_matrix = entity_features[cluster_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
    
    return cluster_labels

def create_visualizations(entity_features: pd.DataFrame, out_dir: Path):
    """Create all required visualizations."""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Top 20 CRS bar chart
    plt.figure(figsize=(15, 10))
    top_20 = entity_features.nlargest(20, 'CRS')
    
    # Create stacked bar chart showing feature contributions
    x_pos = np.arange(len(top_20))
    bottom = np.zeros(len(top_20))
    
    feature_colors = ['red', 'orange', 'yellow', 'green', 'blue']
    crs_features = ['fake_ratio', 'rating_polarization', 'review_burstiness', 'sentiment_rating_gap', 'lexical_redundancy']
    
    for i, feature in enumerate(crs_features):
        values = top_20[feature] * 100  # Convert to percentage
        plt.bar(x_pos, values, bottom=bottom, label=feature, color=feature_colors[i])
        bottom += values
    
    plt.xlabel('Seller/Entity')
    plt.ylabel('CRS Contribution (%)')
    plt.title('Top 20 Counterfeit Risk Scores - Feature Contributions')
    plt.xticks(x_pos, top_20['seller'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'top20_crs_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. CRS histogram by category
    plt.figure(figsize=(15, 10))
    categories = entity_features['product_category'].unique()
    n_categories = len(categories)
    
    for i, category in enumerate(categories):
        category_data = entity_features[entity_features['product_category'] == category]
        if len(category_data) > 0:
            plt.subplot(2, (n_categories + 1) // 2, i + 1)
            plt.hist(category_data['CRS'], bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(category_data['CRS'].quantile(0.95), color='red', linestyle='--', label='95th percentile')
            plt.xlabel('CRS')
            plt.ylabel('Frequency')
            plt.title(f'{category}\n(n={len(category_data)})')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'crs_hist_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Rating distribution: high vs low risk
    plt.figure(figsize=(12, 8))
    
    high_risk = entity_features[entity_features['CRS'] >= entity_features['CRS'].quantile(0.95)]
    low_risk = entity_features[entity_features['CRS'] <= entity_features['CRS'].quantile(0.05)]
    
    if len(high_risk) > 0 and len(low_risk) > 0:
        plt.subplot(1, 2, 1)
        plt.hist(high_risk['rating_polarization'], bins=20, alpha=0.7, label='High Risk', color='red')
        plt.xlabel('Rating Polarization')
        plt.ylabel('Frequency')
        plt.title('High Risk Entities')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(low_risk['rating_polarization'], bins=20, alpha=0.7, label='Low Risk', color='blue')
        plt.xlabel('Rating Polarization')
        plt.ylabel('Frequency')
        plt.title('Low Risk Entities')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'rating_dist_high_vs_low.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. UMAP clusters by CRS
    try:
        from sklearn.manifold import TSNE
        
        # Select features for visualization
        viz_features = ['fake_ratio', 'rating_polarization', 'review_burstiness', 'sentiment_rating_gap', 'lexical_redundancy']
        feature_matrix = entity_features[viz_features].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix_scaled) - 1))
        tsne_result = tsne.fit_transform(feature_matrix_scaled)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                            c=entity_features['CRS'], cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='CRS')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Entity Clusters by Counterfeit Risk Score (t-SNE)')
        plt.tight_layout()
        plt.savefig(out_dir / 'figures' / 'umap_clusters_by_crs.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create t-SNE visualization: {e}")

def generate_risk_summary(entity_features: pd.DataFrame, out_dir: Path):
    """Generate risk summary report."""
    
    # Top 10 flagged entities
    top_flagged = entity_features.nlargest(10, 'CRS')
    
    report = "# Counterfeit Risk Analysis Summary\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Top 10 Flagged Entities\n\n"
    
    for i, (_, entity) in enumerate(top_flagged.iterrows(), 1):
        report += f"### {i}. {entity['seller']}\n"
        report += f"- **CRS**: {entity['CRS']:.2f}\n"
        report += f"- **Product Category**: {entity['product_category']}\n"
        report += f"- **Number of Reviews**: {entity['n_reviews']}\n"
        report += f"- **Fake Review Ratio**: {entity['fake_ratio']:.3f}\n"
        report += f"- **Rating Polarization**: {entity['rating_polarization']:.3f}\n"
        report += f"- **Review Burstiness**: {entity['review_burstiness']:.3f}\n"
        report += f"- **Sentiment-Rating Gap**: {entity['sentiment_rating_gap']:.3f}\n"
        report += f"- **Lexical Redundancy**: {entity['lexical_redundancy']:.3f}\n"
        report += f"- **Anomaly Score**: {entity['anomaly_score']:.3f}\n"
        report += f"- **Risk Flag**: {'YES' if entity['risk_flag'] else 'NO'}\n"
        report += f"- **Consensus Flag**: {'YES' if entity['consensus_flag'] else 'NO'}\n\n"
    
    # Save report
    with open(out_dir / 'reports' / 'risk_summary.md', 'w') as f:
        f.write(report)
    
    logger.info(f"Risk summary saved to {out_dir / 'reports' / 'risk_summary.md'}")

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Analyze counterfeit risk from scored reviews')
    parser.add_argument('--scored_csv', required=True, help='Path to scored Trustpilot CSV')
    parser.add_argument('--out_dir', default='.', help='Output directory')
    parser.add_argument('--weights_json', help='Path to custom CRS weights JSON')
    
    args = parser.parse_args()
    
    # Create output directories
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    (out_dir / 'outputs').mkdir(exist_ok=True)
    (out_dir / 'reports').mkdir(exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)
    
    logger.info("Starting counterfeit risk analysis...")
    log_versions()
    
    # Load CRS weights
    weights_path = Path(args.weights_json) if args.weights_json else None
    crs_weights = load_crs_weights(weights_path)
    
    # Load scored data
    logger.info(f"Loading scored data from {args.scored_csv}")
    scored_df = pd.read_csv(args.scored_csv)
    
    # Check required columns
    required_columns = ['seller', 'product_category', 'rating', 'text', 'date', 'fake_prob']
    missing_columns = [col for col in required_columns if col not in scored_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in scored data: {missing_columns}")
    
    logger.info(f"Scored data loaded: {len(scored_df)} reviews")
    
    # Group by entity (seller + product_id if available)
    logger.info("Grouping reviews by entity...")
    
    # Check if product_id exists
    if 'product_id' in scored_df.columns:
        entity_groups = scored_df.groupby(['seller', 'product_id', 'product_category'])
        logger.info("Grouping by seller + product_id + category")
    else:
        entity_groups = scored_df.groupby(['seller', 'product_category'])
        logger.info("Grouping by seller + category")
    
    # Calculate entity-level features
    logger.info("Calculating entity-level features...")
    
    entity_features = []
    
    for entity_key, group in entity_groups:
        if len(entity_key) == 3:  # seller, product_id, category
            seller, product_id, category = entity_key
        else:  # seller, category
            seller, category = entity_key
            product_id = None
        
        # Calculate features
        fake_features = calculate_fake_review_features(group)
        rating_features = calculate_rating_features(group)
        burst_features = calculate_burst_features(group)
        sentiment_features = calculate_sentiment_features(group)
        
        # Combine all features
        entity_feature = {
            'seller': seller,
            'product_id': product_id,
            'product_category': category,
            **fake_features,
            **rating_features,
            **burst_features,
            **sentiment_features
        }
        
        entity_features.append(entity_feature)
    
    entity_features_df = pd.DataFrame(entity_features)
    logger.info(f"Calculated features for {len(entity_features_df)} entities")
    
    # Calculate CRS
    logger.info("Calculating Counterfeit Risk Scores...")
    crs_scores = calculate_crs(entity_features_df, crs_weights)
    entity_features_df['CRS'] = crs_scores
    
    # Detect anomalies
    logger.info("Detecting anomalies...")
    anomaly_scores = detect_anomalies(entity_features_df)
    entity_features_df['anomaly_score'] = anomaly_scores
    
    # Cluster entities
    logger.info("Clustering entities...")
    cluster_labels = cluster_entities(entity_features_df)
    entity_features_df['cluster'] = cluster_labels
    
    # Calculate risk ranks and flags
    logger.info("Calculating risk ranks and flags...")
    
    # Risk rank within each category
    entity_features_df['risk_rank'] = entity_features_df.groupby('product_category')['CRS'].rank(ascending=False)
    
    # High risk flag (top 5% within each category)
    entity_features_df['risk_flag'] = entity_features_df.groupby('product_category')['CRS'].transform(
        lambda x: x >= x.quantile(0.95)
    )
    
    # Consensus flag (high risk by CRS AND top decile anomaly)
    entity_features_df['consensus_flag'] = (
        entity_features_df['risk_flag'] & 
        (entity_features_df['anomaly_score'] >= entity_features_df['anomaly_score'].quantile(0.9))
    )
    
    # Save entity risk table
    logger.info("Saving entity risk table...")
    entity_features_df.to_csv(out_dir / 'outputs' / 'entity_risk_table.csv', index=False)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(entity_features_df, out_dir)
    
    # Generate risk summary
    logger.info("Generating risk summary...")
    generate_risk_summary(entity_features_df, out_dir)
    
    # Calculate sensitivity analysis
    logger.info("Calculating sensitivity analysis...")
    
    # Weight sensitivity
    alt_weights_1 = crs_weights.copy()
    alt_weights_1['fake_ratio'] = 0.35
    alt_weights_1['rating_polarization'] = 0.30
    
    alt_weights_2 = crs_weights.copy()
    alt_weights_2['fake_ratio'] = 0.55
    alt_weights_2['rating_polarization'] = 0.10
    
    crs_alt_1 = calculate_crs(entity_features_df, alt_weights_1)
    crs_alt_2 = calculate_crs(entity_features_df, alt_weights_2)
    
    # Calculate Kendall's tau between rankings
    from scipy.stats import kendalltau
    
    tau_1, _ = kendalltau(entity_features_df['CRS'], crs_alt_1)
    tau_2, _ = kendalltau(entity_features_df['CRS'], crs_alt_2)
    
    # Threshold sensitivity
    threshold_overlap = {}
    for threshold in [0.4, 0.5, 0.6]:
        if f'fake_pred_t{int(threshold*100)}' in scored_df.columns:
            # Calculate fake share at different thresholds
            fake_share = scored_df.groupby(['seller', 'product_category'])[f'fake_pred_t{int(threshold*100)}'].mean()
            threshold_overlap[f't_{int(threshold*100)}'] = fake_share.to_dict()
    
    sensitivity_results = {
        'weight_sensitivity': {
            'tau_alt_1': float(tau_1),
            'tau_alt_2': float(tau_2)
        },
        'threshold_sensitivity': threshold_overlap
    }
    
    with open(out_dir / 'reports' / 'sensitivity.json', 'w') as f:
        json.dump(sensitivity_results, f, indent=2)
    
    logger.info("Counterfeit risk analysis completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("COUNTERFEIT RISK ANALYSIS COMPLETED")
    print("="*60)
    print(f"Entities analyzed: {len(entity_features_df)}")
    print(f"High risk entities: {entity_features_df['risk_flag'].sum()}")
    print(f"Consensus flagged: {entity_features_df['consensus_flag'].sum()}")
    print(f"CRS range: {entity_features_df['CRS'].min():.2f} - {entity_features_df['CRS'].max():.2f}")
    print(f"Entity risk table: {out_dir / 'outputs' / 'entity_risk_table.csv'}")
    print(f"Risk summary: {out_dir / 'reports' / 'risk_summary.md'}")

if __name__ == "__main__":
    main()

