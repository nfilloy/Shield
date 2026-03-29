"""
Text feature extraction module.

Provides TF-IDF, n-grams, word embeddings, and other text-based
features for phishing/smishing detection.
"""

import logging
from typing import List, Optional, Union, Tuple, Dict, Any
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, hstack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """
    Extract text-based features for machine learning models.

    Supports:
    - TF-IDF (unigrams, bigrams, trigrams)
    - Count Vectorizer (Bag of Words)
    - Character n-grams
    - Word embeddings (Word2Vec, FastText)
    - Dimensionality reduction (SVD/LSA)
    """

    def __init__(
        self,
        method: str = 'tfidf',
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: float = 0.95,
        use_char_ngrams: bool = False,
        char_ngram_range: Tuple[int, int] = (2, 5),
        char_max_features: int = 5000,
        use_svd: bool = False,
        svd_components: int = 300,
        sublinear_tf: bool = True
    ):
        """
        Initialize the TextFeatureExtractor.

        Args:
            method: Feature extraction method ('tfidf', 'count', 'both')
            max_features: Maximum number of word features
            ngram_range: Range of n-grams (1,1)=unigrams, (1,2)=uni+bi
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_char_ngrams: Include character-level n-grams
            char_ngram_range: Range for character n-grams
            char_max_features: Maximum character features
            use_svd: Apply SVD for dimensionality reduction
            svd_components: Number of SVD components
            sublinear_tf: Use sublinear TF scaling
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_char_ngrams = use_char_ngrams
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.use_svd = use_svd
        self.svd_components = svd_components
        self.sublinear_tf = sublinear_tf

        # Vectorizers (initialized in fit)
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.svd = None

        self._is_fitted = False

    def _create_word_vectorizer(self) -> Union[TfidfVectorizer, CountVectorizer]:
        """Create word-level vectorizer based on method."""
        if self.method in ['tfidf', 'both']:
            return TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=self.sublinear_tf,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b|<[A-Z]+>'  # Words + special tokens
            )
        else:
            return CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b|<[A-Z]+>'
            )

    def _create_char_vectorizer(self) -> TfidfVectorizer:
        """Create character-level vectorizer."""
        return TfidfVectorizer(
            max_features=self.char_max_features,
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer='char_wb',  # Character n-grams within word boundaries
            sublinear_tf=self.sublinear_tf
        )

    def fit(self, texts: Union[List[str], pd.Series]) -> 'TextFeatureExtractor':
        """
        Fit vectorizers on training texts.

        Args:
            texts: Training texts

        Returns:
            Self for chaining
        """
        if isinstance(texts, pd.Series):
            texts = texts.fillna('').tolist()

        logger.info(f"Fitting {self.method} vectorizer on {len(texts)} texts...")

        # Fit word vectorizer
        self.word_vectorizer = self._create_word_vectorizer()
        word_features = self.word_vectorizer.fit_transform(texts)
        logger.info(f"Word features shape: {word_features.shape}")

        # Fit character vectorizer if enabled
        if self.use_char_ngrams:
            self.char_vectorizer = self._create_char_vectorizer()
            char_features = self.char_vectorizer.fit_transform(texts)
            logger.info(f"Char features shape: {char_features.shape}")

            # Combine for SVD fitting
            combined = hstack([word_features, char_features])
        else:
            combined = word_features

        # Fit SVD if enabled
        if self.use_svd:
            n_components = min(self.svd_components, combined.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(combined)
            logger.info(f"SVD variance explained: {self.svd.explained_variance_ratio_.sum():.2%}")

        self._is_fitted = True
        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts to feature vectors.

        Args:
            texts: Texts to transform

        Returns:
            Feature matrix
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        if isinstance(texts, pd.Series):
            texts = texts.fillna('').tolist()

        # Transform with word vectorizer
        word_features = self.word_vectorizer.transform(texts)

        # Transform with char vectorizer if enabled
        if self.use_char_ngrams and self.char_vectorizer:
            char_features = self.char_vectorizer.transform(texts)
            features = hstack([word_features, char_features])
        else:
            features = word_features

        # Apply SVD if enabled
        if self.use_svd and self.svd:
            features = self.svd.transform(features)

        # Convert to dense if needed (for some models)
        if isinstance(features, csr_matrix):
            features = features.toarray()

        return features

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            texts: Texts to fit and transform

        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Get feature names from vectorizers."""
        if not self._is_fitted:
            raise ValueError("Vectorizer not fitted.")

        names = list(self.word_vectorizer.get_feature_names_out())

        if self.use_char_ngrams and self.char_vectorizer:
            char_names = [f"char_{n}" for n in self.char_vectorizer.get_feature_names_out()]
            names.extend(char_names)

        if self.use_svd:
            names = [f"svd_{i}" for i in range(self.svd.n_components_)]

        return names

    def get_top_features(self, n: int = 20) -> Dict[str, float]:
        """
        Get top features by IDF score (for TF-IDF).

        Args:
            n: Number of top features

        Returns:
            Dictionary of feature names and scores
        """
        if not self._is_fitted or not hasattr(self.word_vectorizer, 'idf_'):
            return {}

        feature_names = self.word_vectorizer.get_feature_names_out()
        idf_scores = self.word_vectorizer.idf_

        # Sort by IDF score (lower = more common)
        indices = np.argsort(idf_scores)[:n]

        return {feature_names[i]: idf_scores[i] for i in indices}

    def save(self, filepath: str):
        """Save fitted vectorizers to file."""
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted vectorizer.")

        data = {
            'word_vectorizer': self.word_vectorizer,
            'char_vectorizer': self.char_vectorizer,
            'svd': self.svd,
            'config': {
                'method': self.method,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'use_char_ngrams': self.use_char_ngrams,
                'use_svd': self.use_svd
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved vectorizer to {filepath}")

    def load(self, filepath: str):
        """Load fitted vectorizers from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.word_vectorizer = data['word_vectorizer']
        self.char_vectorizer = data.get('char_vectorizer')
        self.svd = data.get('svd')
        self._is_fitted = True

        logger.info(f"Loaded vectorizer from {filepath}")


class WordEmbeddingExtractor:
    """
    Extract word embedding features using pre-trained models.

    Supports Word2Vec, FastText, and custom embeddings.
    """

    def __init__(
        self,
        embedding_type: str = 'word2vec',
        embedding_dim: int = 300,
        aggregation: str = 'mean',
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize the WordEmbeddingExtractor.

        Args:
            embedding_type: Type of embeddings ('word2vec', 'fasttext', 'glove')
            embedding_dim: Embedding dimension
            aggregation: How to aggregate word vectors ('mean', 'max', 'concat')
            pretrained_path: Path to pre-trained embeddings
        """
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.pretrained_path = pretrained_path

        self.model = None
        self._is_fitted = False

    def fit(self, texts: Union[List[str], pd.Series]) -> 'WordEmbeddingExtractor':
        """
        Fit or load embedding model.

        Args:
            texts: Training texts (used if training from scratch)

        Returns:
            Self for chaining
        """
        if self.pretrained_path and Path(self.pretrained_path).exists():
            self._load_pretrained()
        else:
            self._train_embeddings(texts)

        self._is_fitted = True
        return self

    def _load_pretrained(self):
        """Load pre-trained embeddings."""
        try:
            from gensim.models import KeyedVectors

            logger.info(f"Loading pre-trained embeddings from {self.pretrained_path}")

            if self.pretrained_path.endswith('.bin'):
                self.model = KeyedVectors.load_word2vec_format(
                    self.pretrained_path, binary=True
                )
            else:
                self.model = KeyedVectors.load_word2vec_format(
                    self.pretrained_path, binary=False
                )

            self.embedding_dim = self.model.vector_size
            logger.info(f"Loaded embeddings with dimension {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def _train_embeddings(self, texts: Union[List[str], pd.Series]):
        """Train embeddings from scratch."""
        try:
            from gensim.models import Word2Vec, FastText

            if isinstance(texts, pd.Series):
                texts = texts.fillna('').tolist()

            # Tokenize texts
            sentences = [text.split() for text in texts]

            logger.info(f"Training {self.embedding_type} embeddings on {len(sentences)} texts...")

            if self.embedding_type == 'fasttext':
                model = FastText(
                    sentences=sentences,
                    vector_size=self.embedding_dim,
                    window=5,
                    min_count=2,
                    workers=4,
                    epochs=10
                )
            else:  # word2vec
                model = Word2Vec(
                    sentences=sentences,
                    vector_size=self.embedding_dim,
                    window=5,
                    min_count=2,
                    workers=4,
                    epochs=10
                )

            self.model = model.wv
            logger.info(f"Trained embeddings with vocabulary size {len(self.model)}")

        except ImportError:
            logger.error("gensim not installed. Install with: pip install gensim")
            raise

    def _get_text_vector(self, text: str) -> np.ndarray:
        """Get aggregated vector for a text."""
        words = text.split()
        vectors = []

        for word in words:
            if word in self.model:
                vectors.append(self.model[word])

        if not vectors:
            return np.zeros(self.embedding_dim)

        vectors = np.array(vectors)

        if self.aggregation == 'mean':
            return np.mean(vectors, axis=0)
        elif self.aggregation == 'max':
            return np.max(vectors, axis=0)
        elif self.aggregation == 'concat':
            mean_vec = np.mean(vectors, axis=0)
            max_vec = np.max(vectors, axis=0)
            min_vec = np.min(vectors, axis=0)
            return np.concatenate([mean_vec, max_vec, min_vec])
        else:
            return np.mean(vectors, axis=0)

    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts to embedding vectors.

        Args:
            texts: Texts to transform

        Returns:
            Feature matrix
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(texts, pd.Series):
            texts = texts.fillna('').tolist()

        vectors = [self._get_text_vector(text) for text in texts]
        return np.array(vectors)

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)


class CombinedFeatureExtractor:
    """
    Combine multiple feature extraction methods.

    Useful for creating rich feature representations that include
    both statistical features (TF-IDF) and semantic features (embeddings).
    """

    def __init__(
        self,
        use_tfidf: bool = True,
        use_char_ngrams: bool = True,
        use_embeddings: bool = False,
        tfidf_config: Optional[Dict] = None,
        embedding_config: Optional[Dict] = None
    ):
        """
        Initialize combined feature extractor.

        Args:
            use_tfidf: Include TF-IDF features
            use_char_ngrams: Include character n-grams
            use_embeddings: Include word embeddings
            tfidf_config: Configuration for TF-IDF extractor
            embedding_config: Configuration for embedding extractor
        """
        self.use_tfidf = use_tfidf
        self.use_char_ngrams = use_char_ngrams
        self.use_embeddings = use_embeddings

        # Initialize extractors
        if use_tfidf:
            tfidf_config = tfidf_config or {}
            tfidf_config['use_char_ngrams'] = use_char_ngrams
            self.tfidf_extractor = TextFeatureExtractor(**tfidf_config)

        if use_embeddings:
            embedding_config = embedding_config or {}
            self.embedding_extractor = WordEmbeddingExtractor(**embedding_config)

        self._is_fitted = False

    def fit(self, texts: Union[List[str], pd.Series]) -> 'CombinedFeatureExtractor':
        """Fit all extractors."""
        if self.use_tfidf:
            self.tfidf_extractor.fit(texts)

        if self.use_embeddings:
            self.embedding_extractor.fit(texts)

        self._is_fitted = True
        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts with all extractors and concatenate."""
        features_list = []

        if self.use_tfidf:
            tfidf_features = self.tfidf_extractor.transform(texts)
            if isinstance(tfidf_features, csr_matrix):
                tfidf_features = tfidf_features.toarray()
            features_list.append(tfidf_features)

        if self.use_embeddings:
            embedding_features = self.embedding_extractor.transform(texts)
            features_list.append(embedding_features)

        return np.hstack(features_list)

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "urgent verify your account immediately click here",
        "meeting tomorrow at 3pm please confirm",
        "you won a free iphone click the link now",
        "project update attached for your review",
        "your bank account has been suspended verify now"
    ]

    # TF-IDF features
    tfidf_extractor = TextFeatureExtractor(
        method='tfidf',
        max_features=1000,
        ngram_range=(1, 2),
        use_char_ngrams=True
    )

    features = tfidf_extractor.fit_transform(sample_texts)
    print(f"TF-IDF features shape: {features.shape}")
    print(f"Top features: {tfidf_extractor.get_top_features(10)}")
