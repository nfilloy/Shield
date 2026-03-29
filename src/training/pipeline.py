
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Import project modules
from src.data.sms_loader import SMSLoader
from src.data.email_loader import EmailLoader
from src.data.preprocessor import TextPreprocessor, get_ml_preprocessor
from src.features.text_features import TextFeatureExtractor
from src.features.sms_features import SMSFeatureExtractor
from src.features.email_features import EmailFeatureExtractor
from src.models.classical import ClassicalModels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Unified training pipeline for Phishing and Smishing detection.
    Handles data loading, preprocessing, feature extraction, training, and evaluation.
    """

    def __init__(self, config: Dict[str, Any] = None, models_dir: str = 'models'):
        """
        Initialize the pipeline.

        Args:
            config: Configuration dictionary (optional)
            models_dir: Directory to save trained models
        """
        self.config = config or {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components placeholder
        self.preprocessor = None
        self.extractor = None
        self.models = None

        # URL/structural feature extractors
        self.sms_feature_extractor = SMSFeatureExtractor()
        self.email_feature_extractor = EmailFeatureExtractor()
        self.url_feature_names = None  # Will store feature names for inference

    def _load_data(self, data_type: str) -> pd.DataFrame:
        """Load dataset based on type.

        Uses phishing/smishing-specific datasets:
        - SMS: Mendeley + Hugging Face smishing datasets
        - Email: Hugging Face phishing dataset
        """
        logger.info(f"Loading {data_type.upper()} data...")

        if data_type == 'sms':
            loader = SMSLoader()
            # Load combined smishing dataset (Mendeley + HuggingFace)
            df = loader.load_combined_smishing()

            if df.empty:
                logger.error("No SMS smishing data could be loaded!")
                logger.info("Please check your internet connection or install 'datasets' library:")
                logger.info("  pip install datasets")
                return pd.DataFrame()

            logger.info(f"Loaded SMISHING dataset: {len(df)} messages")

        elif data_type == 'email':
            loader = EmailLoader()
            # Load phishing dataset from HuggingFace
            df = loader.load_huggingface_phishing()

            if df.empty:
                logger.error("No email phishing data could be loaded!")
                logger.info("Please check your internet connection or install 'datasets' library:")
                logger.info("  pip install datasets")
                return pd.DataFrame()

            # For email, use body directly as full_text (no subject in HF dataset)
            df['full_text'] = df['body'].fillna('')
            # Filter short texts
            df = df[df['full_text'].str.len() > 10]

            logger.info(f"Loaded PHISHING dataset: {len(df)} emails")

        else:
            raise ValueError(f"Unknown data type: {data_type}")

        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")

        return df

    def _preprocess(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Preprocess text data."""
        logger.info("Preprocessing texts...")

        # Use centralized preprocessor to ensure consistency with inference
        self.preprocessor = get_ml_preprocessor()

        # Use batch processing if available or list comprehension
        texts = df[text_col].astype(str).tolist()
        df['text_clean'] = [self.preprocessor.preprocess(t) for t in texts]

        return df

    def _extract_url_features(self, texts: pd.Series, data_type: str) -> np.ndarray:
        """
        Extract URL and structural features from RAW text (before preprocessing).

        These features capture important signals like:
        - URL count, shortened URLs, suspicious TLDs, IP-based URLs
        - Urgency/phishing keywords
        - Character patterns (uppercase ratio, special chars)
        - For emails: header analysis, link-text mismatches

        Args:
            texts: Raw text data (NOT preprocessed)
            data_type: 'sms' or 'email'

        Returns:
            Numpy array of extracted features
        """
        logger.info(f"Extracting URL/structural features for {data_type}...")

        if data_type == 'sms':
            features_df = self.sms_feature_extractor.extract_features_batch(texts)
            self.url_feature_names = features_df.columns.tolist()
        else:
            # For email, we need to convert texts to expected format
            email_data_list = [{'body': str(t), 'subject': '', 'headers': {}} for t in texts]
            features_df = self.email_feature_extractor.extract_features_batch(email_data_list)
            self.url_feature_names = features_df.columns.tolist()

        logger.info(f"Extracted {len(self.url_feature_names)} URL/structural features")

        return features_df.values

    def _extract_features(
        self,
        X_train_clean: pd.Series,
        X_test_clean: pd.Series,
        X_train_raw: pd.Series,
        X_test_raw: pd.Series,
        data_type: str,
        prefix: str = ''
    ) -> tuple:
        """
        Extract combined features: TF-IDF + URL/structural features.

        Args:
            X_train_clean: Preprocessed training texts (for TF-IDF)
            X_test_clean: Preprocessed test texts (for TF-IDF)
            X_train_raw: Raw training texts (for URL features)
            X_test_raw: Raw test texts (for URL features)
            data_type: 'sms' or 'email'
            prefix: Prefix for saved files

        Returns:
            Tuple of combined feature matrices (train, test)
        """
        logger.info("Extracting features (TF-IDF + URL/structural)...")

        # 1. Extract TF-IDF features from preprocessed text
        self.extractor = TextFeatureExtractor(
            method='tfidf',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2 if prefix == '' else 3,
            max_df=0.95,
            use_char_ngrams=True,
            char_ngram_range=(2, 5),
            char_max_features=5000,
            sublinear_tf=True
        )

        X_train_tfidf = self.extractor.fit_transform(X_train_clean)
        X_test_tfidf = self.extractor.transform(X_test_clean)
        logger.info(f"TF-IDF features shape: {X_train_tfidf.shape}")

        # 2. Extract URL/structural features from RAW text
        X_train_url = self._extract_url_features(X_train_raw, data_type)
        X_test_url = self._extract_url_features(X_test_raw, data_type)
        logger.info(f"URL/structural features shape: {X_train_url.shape}")

        # 3. Combine features
        # Convert TF-IDF to dense if sparse
        if hasattr(X_train_tfidf, 'toarray'):
            X_train_tfidf = X_train_tfidf.toarray()
        if hasattr(X_test_tfidf, 'toarray'):
            X_test_tfidf = X_test_tfidf.toarray()

        X_train_combined = np.hstack([X_train_tfidf, X_train_url])
        X_test_combined = np.hstack([X_test_tfidf, X_test_url])
        logger.info(f"Combined features shape: {X_train_combined.shape}")

        # 4. Save vectorizer and URL feature names
        vec_filename = f"{prefix}tfidf_vectorizer.pkl" if prefix else "tfidf_vectorizer.pkl"
        self.extractor.save(str(self.models_dir / vec_filename))

        # Save URL feature configuration for inference
        url_config_filename = f"{prefix}url_features_config.pkl" if prefix else "url_features_config.pkl"
        with open(self.models_dir / url_config_filename, 'wb') as f:
            pickle.dump({
                'feature_names': self.url_feature_names,
                'data_type': data_type,
                'n_features': len(self.url_feature_names)
            }, f)
        logger.info(f"Saved URL features config to {url_config_filename}")

        return X_train_combined, X_test_combined

    def run(self, data_type: str = 'sms'):
        """
        Run the complete training pipeline.

        Args:
            data_type: 'sms' or 'email'
        """
        logger.info(f"=== Starting Training Pipeline for {data_type.upper()} ===")

        # 1. Load Data
        df = self._load_data(data_type)
        if df.empty:
            logger.error("No data loaded. Aborting.")
            return

        # 2. Preprocess
        text_col = 'body' if data_type == 'sms' else 'full_text'
        df = self._preprocess(df, text_col)

        # 3. Split - Keep both raw and clean text for feature extraction
        X_clean = df['text_clean']
        X_raw = df[text_col].astype(str)  # Raw text for URL features
        y = df['label'].values

        # Split maintaining index alignment
        indices = df.index.tolist()
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_clean = X_clean.loc[train_idx].reset_index(drop=True)
        X_test_clean = X_clean.loc[test_idx].reset_index(drop=True)
        X_train_raw = X_raw.loc[train_idx].reset_index(drop=True)
        X_test_raw = X_raw.loc[test_idx].reset_index(drop=True)

        logger.info(f"Train size: {len(X_train_clean)}, Test size: {len(X_test_clean)}")

        # 4. Feature Extraction (TF-IDF + URL/structural features)
        prefix = 'email_' if data_type == 'email' else ''
        X_train_vec, X_test_vec = self._extract_features(
            X_train_clean, X_test_clean,
            X_train_raw, X_test_raw,
            data_type=data_type,
            prefix=prefix
        )
        
        # 5. Initialize Models
        self.models = ClassicalModels(model_dir=str(self.models_dir))
        
        # Define models to train
        model_names = ['naive_bayes', 'logistic_regression', 'random_forest']
        # Add LinearSVM (más rápido y apropiado para datos de texto de alta dimensionalidad)
        if data_type == 'sms':
            model_names.append('linear_svm')
            
        try:
            import xgboost
            model_names.append('xgboost')
        except ImportError:
            pass
            
        # 6. Train & Evaluate
        results = {}
        
        # Calcular scale_pos_weight dinámico para XGBoost (desbalance de clases)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Class balance: neg={n_neg}, pos={n_pos}, scale_pos_weight={scale_pos_weight:.2f}")

        for name in model_names:
            full_model_name = f"{prefix}{name}" if prefix else name
            logger.info(f"Training {full_model_name}...")

            # Train con parámetros específicos por modelo
            if name == 'naive_bayes':
                 # Use alpha=0.1 as determined to be better for this task
                 self.models.train(name, X_train_vec, y_train, params={'alpha': 0.1})
            elif name == 'xgboost':
                 # XGBoost con scale_pos_weight ajustado por desbalance
                 xgb_params = {
                     'n_estimators': 100,
                     'learning_rate': 0.1,
                     'max_depth': 6,
                     'min_child_weight': 1,
                     'subsample': 0.8,
                     'colsample_bytree': 0.8,
                     'scale_pos_weight': scale_pos_weight,
                     'random_state': 42,
                     'use_label_encoder': False,
                     'eval_metric': 'logloss'
                 }
                 self.models.train(name, X_train_vec, y_train, params=xgb_params)
            else:
                 self.models.train(name, X_train_vec, y_train)
                 
            # Save under the specific name
            self.models.save_model(name, filename=f"{full_model_name}.pkl")
            
            # Evaluate
            y_pred = self.models.predict(name, X_test_vec)
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            results[full_model_name] = {'f1': f1, 'accuracy': acc}
            
            logger.info(f"Results for {full_model_name}:")
            logger.info(f"  F1: {f1:.4f}")
            logger.info(f"  Acc: {acc:.4f}")
            
        # 7. Summary
        best_model = max(results, key=lambda x: results[x]['f1'])
        logger.info(f"\nTraining Complete. Best model: {best_model} (F1: {results[best_model]['f1']:.4f})")
        
        return results

if __name__ == "__main__":
    # Test run
    pipeline = TrainingPipeline()
    # pipeline.run('sms')
