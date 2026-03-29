"""
Main entry point for the Phishing & Smishing Detection System.

This script provides a command-line interface for:
- Training models
- Making predictions
- Running the full pipeline
- Launching the web application
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}


def train_pipeline(data_type: str = 'sms', config: dict = None):
    """
    Run the training pipeline.

    Args:
        data_type: Type of data ('sms' or 'email')
        config: Configuration dictionary
    """
    from src.training.pipeline import TrainingPipeline
    
    pipeline = TrainingPipeline(config=config)
    results = pipeline.run(data_type=data_type)
    
    return results


def predict(text: str, model_name: str = 'random_forest', data_type: str = 'sms'):
    """
    Make prediction on a single text using combined TF-IDF + URL features.

    Args:
        text: Text to classify
        model_name: Name of model to use
        data_type: Type of data ('sms' or 'email')

    Returns:
        Prediction result dictionary
    """
    import pickle
    from scipy.sparse import hstack
    from src.data.preprocessor import get_ml_preprocessor
    from src.features.sms_features import SMSFeatureExtractor
    from src.features.email_features import EmailFeatureExtractor

    # Load model and vectorizer
    models_dir = Path('models')

    # Determine paths based on data type
    if data_type == 'email':
        # Add 'email_' prefix if not already present
        if not model_name.startswith('email_'):
            model_path = models_dir / f'email_{model_name}.pkl'
        else:
            model_path = models_dir / f'{model_name}.pkl'
        vectorizer_path = models_dir / 'email_tfidf_vectorizer.pkl'
        url_config_path = models_dir / 'email_url_features_config.pkl'
    else:
        model_path = models_dir / f'{model_name}.pkl'
        vectorizer_path = models_dir / 'tfidf_vectorizer.pkl'
        url_config_path = models_dir / 'url_features_config.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer_data = pickle.load(f)

    # Handle both dict format and raw vectorizer
    if isinstance(vectorizer_data, dict):
        word_vectorizer = vectorizer_data['word_vectorizer']
        char_vectorizer = vectorizer_data.get('char_vectorizer')
    else:
        word_vectorizer = vectorizer_data
        char_vectorizer = None

    # Load URL features config if available (new models)
    url_config = None
    if url_config_path.exists():
        with open(url_config_path, 'rb') as f:
            url_config = pickle.load(f)

    # Preprocess using centralized preprocessor (matches training)
    preprocessor = get_ml_preprocessor()
    clean_text = preprocessor.preprocess(text)

    # 1. Extract TF-IDF features from preprocessed text
    X_word = word_vectorizer.transform([clean_text])
    if char_vectorizer is not None:
        X_char = char_vectorizer.transform([clean_text])
        X_tfidf = hstack([X_word, X_char])
    else:
        X_tfidf = X_word

    # Convert to dense if needed
    if hasattr(X_tfidf, "toarray"):
        X_tfidf = X_tfidf.toarray()

    # 2. Extract URL/structural features from RAW text
    if url_config is not None:
        if data_type == 'sms':
            feature_extractor = SMSFeatureExtractor()
            features = feature_extractor.extract_all_features(text)
        else:
            feature_extractor = EmailFeatureExtractor()
            features = feature_extractor.extract_all_features({
                'body': text, 'subject': '', 'headers': {}
            })

        # Build features array in the same order as training
        feature_names = url_config['feature_names']
        X_url = np.array([[features.get(name, 0) for name in feature_names]])

        # 3. Combine TF-IDF + URL features
        X = np.hstack([X_tfidf, X_url])
    else:
        # Fallback for old models without URL features
        X = X_tfidf
        features = {}

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    result = {
        'text': text,
        'clean_text': clean_text,
        'prediction': int(prediction),
        'label': 'Phishing/Smishing' if prediction == 1 else 'Legitimate',
        'probability': float(proba[1]) * 100,
        'confidence': float(max(proba)) * 100,
        'uses_url_features': url_config is not None,
        'url_features': features if url_config else {}
    }

    return result


def run_app():
    """Launch the Streamlit web application."""
    import subprocess
    app_path = Path(__file__).parent.parent / 'app' / 'streamlit_app.py'
    subprocess.run(['streamlit', 'run', str(app_path)])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phishing & Smishing Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main train --type sms
  python -m src.main predict "Click here to claim your prize!"
  python -m src.main app
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument(
        '--type', '-t',
        choices=['sms', 'email'],
        default='sms',
        help='Type of data to train on'
    )
    train_parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction')
    predict_parser.add_argument(
        'text',
        help='Text to classify'
    )
    predict_parser.add_argument(
        '--model', '-m',
        default='random_forest',
        help='Model to use'
    )
    predict_parser.add_argument(
        '--type', '-t',
        choices=['sms', 'email'],
        default='sms',
        help='Type of text'
    )

    # App command
    app_parser = subparsers.add_parser('app', help='Launch web application')

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'train':
        config = load_config(args.config)
        train_pipeline(data_type=args.type, config=config)

    elif args.command == 'predict':
        try:
            result = predict(args.text, args.model, args.type)
            print("\n" + "="*50)
            print("PREDICTION RESULT")
            print("="*50)
            print(f"Text: {result['text'][:100]}...")
            print(f"Prediction: {result['label']}")
            print(f"Probability of fraud: {result['probability']:.1f}%")
            print(f"Confidence: {result['confidence']:.1f}%")
            print("="*50)
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info("Please run 'python -m src.main train' first to train models.")
            sys.exit(1)

    elif args.command == 'app':
        run_app()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
