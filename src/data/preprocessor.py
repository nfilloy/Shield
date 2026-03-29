"""
Text preprocessing module for phishing and smishing detection.

Provides comprehensive text cleaning and normalization utilities
for both email and SMS data.
"""

import re
import string
import logging
from typing import List, Optional, Union, Callable
from html import unescape

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading for NLP libraries
_nlp = None
_nltk_initialized = False


def _get_spacy_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Downloading spaCy model 'en_core_web_sm'...")
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not available. Some preprocessing features will be limited.")
            _nlp = None
    return _nlp


def _init_nltk():
    """Initialize NLTK resources."""
    global _nltk_initialized
    if not _nltk_initialized:
        try:
            import nltk
            for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            _nltk_initialized = True
        except ImportError:
            logger.warning("NLTK not available.")
    return _nltk_initialized


class TextPreprocessor:
    """
    Comprehensive text preprocessor for phishing/smishing detection.

    Supports multiple preprocessing strategies and can be customized
    for different use cases (emails vs SMS).
    """

    # Common URL patterns
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    # Email pattern
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

    # Phone number patterns
    PHONE_PATTERN = re.compile(
        r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|'
        r'\d{10,}|'
        r'(?:\+\d{1,3}[-.\s]?)?\d{6,}'
    )

    # Money/currency patterns
    MONEY_PATTERN = re.compile(r'[$£€¥]\s*\d+(?:[,\.]\d+)*|\d+(?:[,\.]\d+)*\s*(?:dollars?|euros?|pounds?)')

    # Short URL domains
    SHORT_URL_DOMAINS = [
        'bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'j.mp', 'tr.im', 'cli.gs', 'short.to'
    ]

    # Suspicious/urgency words for phishing detection
    URGENCY_WORDS = [
        'urgent', 'immediately', 'asap', 'expire', 'suspended', 'locked',
        'verify', 'confirm', 'update', 'secure', 'alert', 'warning',
        'limited', 'act now', 'deadline', 'final', 'last chance',
        'winner', 'won', 'congratulations', 'free', 'prize', 'reward',
        'click', 'login', 'password', 'account', 'bank', 'credit card'
    ]

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = False,
        replace_urls: bool = True,
        remove_emails: bool = False,
        replace_emails: bool = True,
        remove_numbers: bool = False,
        replace_numbers: bool = True,
        remove_punctuation: bool = False,
        remove_extra_whitespace: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        stem: bool = False,
        min_word_length: int = 1,
        max_word_length: int = 50,
        use_spacy: bool = False
    ):
        """
        Initialize the TextPreprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_html: Remove HTML tags
            remove_urls: Completely remove URLs
            replace_urls: Replace URLs with <URL> token (if not removing)
            remove_emails: Completely remove email addresses
            replace_emails: Replace emails with <EMAIL> token (if not removing)
            remove_numbers: Completely remove numbers
            replace_numbers: Replace numbers with <NUM> token (if not removing)
            remove_punctuation: Remove punctuation characters
            remove_extra_whitespace: Collapse multiple whitespaces
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            stem: Apply stemming (mutually exclusive with lemmatize)
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            use_spacy: Use spaCy for tokenization/lemmatization
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.replace_urls = replace_urls
        self.remove_emails = remove_emails
        self.replace_emails = replace_emails
        self.remove_numbers = remove_numbers
        self.replace_numbers = replace_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.use_spacy = use_spacy

        # Initialize resources
        self._stopwords = None
        self._stemmer = None
        self._lemmatizer = None

        if self.remove_stopwords or self.stem or (self.lemmatize and not use_spacy):
            _init_nltk()

    @property
    def stopwords(self):
        """Lazy load stopwords."""
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words('english'))
            except:
                self._stopwords = set()
        return self._stopwords

    @property
    def stemmer(self):
        """Lazy load stemmer."""
        if self._stemmer is None and self.stem:
            try:
                from nltk.stem import PorterStemmer
                self._stemmer = PorterStemmer()
            except:
                pass
        return self._stemmer

    @property
    def lemmatizer(self):
        """Lazy load lemmatizer."""
        if self._lemmatizer is None and self.lemmatize and not self.use_spacy:
            try:
                from nltk.stem import WordNetLemmatizer
                self._lemmatizer = WordNetLemmatizer()
            except:
                pass
        return self._lemmatizer

    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        if not text:
            return ""

        # Decode HTML entities
        text = unescape(text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove CSS/style content
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove script content
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

        return text

    def process_urls(self, text: str) -> str:
        """Handle URLs in text."""
        if self.remove_urls:
            return self.URL_PATTERN.sub(' ', text)
        elif self.replace_urls:
            return self.URL_PATTERN.sub(' <URL> ', text)
        return text

    def process_emails(self, text: str) -> str:
        """Handle email addresses in text."""
        if self.remove_emails:
            return self.EMAIL_PATTERN.sub(' ', text)
        elif self.replace_emails:
            return self.EMAIL_PATTERN.sub(' <EMAIL> ', text)
        return text

    def process_numbers(self, text: str) -> str:
        """Handle numbers in text."""
        if self.remove_numbers:
            return re.sub(r'\d+', ' ', text)
        elif self.replace_numbers:
            # Replace phone numbers first
            text = self.PHONE_PATTERN.sub(' <PHONE> ', text)
            # Replace money amounts
            text = self.MONEY_PATTERN.sub(' <MONEY> ', text)
            # Replace remaining numbers
            text = re.sub(r'\b\d+\b', ' <NUM> ', text)
        return text

    def remove_punctuation_fn(self, text: str) -> str:
        """Remove punctuation from text."""
        # Keep special tokens
        text = re.sub(r'<(URL|EMAIL|PHONE|MONEY|NUM)>', r'###\1###', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'###(URL|EMAIL|PHONE|MONEY|NUM)###', r'<\1>', text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.use_spacy:
            nlp = _get_spacy_nlp()
            if nlp:
                doc = nlp(text)
                return [token.text for token in doc]

        # Simple whitespace tokenization
        return text.split()

    def process_tokens(self, tokens: List[str]) -> List[str]:
        """Process individual tokens (stopwords, lemmatization, stemming)."""
        processed = []

        for token in tokens:
            # Skip special tokens
            if token.startswith('<') and token.endswith('>'):
                processed.append(token)
                continue

            # Length filter
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue

            # Stopword removal
            if self.remove_stopwords and token.lower() in self.stopwords:
                continue

            # Lemmatization or stemming
            if self.lemmatize:
                if self.use_spacy:
                    nlp = _get_spacy_nlp()
                    if nlp:
                        doc = nlp(token)
                        token = doc[0].lemma_ if doc else token
                elif self.lemmatizer:
                    token = self.lemmatizer.lemmatize(token.lower())
            elif self.stem and self.stemmer:
                token = self.stemmer.stem(token)

            processed.append(token)

        return processed

    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # HTML cleaning
        if self.remove_html:
            text = self.clean_html(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Process URLs, emails, numbers
        text = self.process_urls(text)
        text = self.process_emails(text)
        text = self.process_numbers(text)

        # Remove punctuation
        if self.remove_punctuation:
            text = self.remove_punctuation_fn(text)

        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = self.normalize_whitespace(text)

        # Token-level processing
        if self.remove_stopwords or self.lemmatize or self.stem:
            tokens = self.tokenize(text)
            tokens = self.process_tokens(tokens)
            text = ' '.join(tokens)

        return text

    def preprocess_batch(
        self,
        texts: Union[List[str], pd.Series],
        show_progress: bool = True
    ) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List or Series of texts to preprocess
            show_progress: Show progress bar

        Returns:
            List of preprocessed texts
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        if show_progress:
            from tqdm import tqdm
            return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
        else:
            return [self.preprocess(text) for text in texts]

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'body',
        output_column: str = 'text_clean',
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name for output column
            inplace: Modify DataFrame in place

        Returns:
            DataFrame with preprocessed text column
        """
        if not inplace:
            df = df.copy()

        df[output_column] = self.preprocess_batch(df[text_column])
        return df

    def extract_features_count(self, text: str) -> dict:
        """
        Extract count-based features from raw text (before preprocessing).

        Useful for phishing/smishing detection feature engineering.

        Args:
            text: Raw input text

        Returns:
            Dictionary of feature counts
        """
        if not text or not isinstance(text, str):
            return {}

        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'url_count': len(self.URL_PATTERN.findall(text)),
            'email_count': len(self.EMAIL_PATTERN.findall(text)),
            'phone_count': len(self.PHONE_PATTERN.findall(text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
            'has_short_url': any(domain in text.lower() for domain in self.SHORT_URL_DOMAINS),
            'urgency_word_count': sum(1 for word in self.URGENCY_WORDS if word in text.lower()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'money_mentions': len(self.MONEY_PATTERN.findall(text)),
        }

        return features


class EmailPreprocessor(TextPreprocessor):
    """Specialized preprocessor for email content."""

    def __init__(self, **kwargs):
        # Email-specific defaults
        kwargs.setdefault('remove_html', True)
        kwargs.setdefault('replace_urls', True)
        kwargs.setdefault('replace_emails', True)
        super().__init__(**kwargs)

    def preprocess_email(self, subject: str, body: str, combine: bool = True) -> Union[str, dict]:
        """
        Preprocess email subject and body.

        Args:
            subject: Email subject line
            body: Email body content
            combine: Whether to combine subject and body

        Returns:
            Preprocessed text or dict with separate fields
        """
        clean_subject = self.preprocess(subject) if subject else ""
        clean_body = self.preprocess(body) if body else ""

        if combine:
            return f"{clean_subject} {clean_body}".strip()
        else:
            return {'subject': clean_subject, 'body': clean_body}


class SMSPreprocessor(TextPreprocessor):
    """Specialized preprocessor for SMS content."""

    def __init__(self, **kwargs):
        # SMS-specific defaults (preserve more content due to short messages)
        kwargs.setdefault('remove_html', False)
        kwargs.setdefault('replace_urls', True)
        kwargs.setdefault('remove_stopwords', False)  # Keep stopwords for short messages
        kwargs.setdefault('lemmatize', False)
        super().__init__(**kwargs)


def get_preprocessor(
    data_type: str = 'general',
    for_classical_ml: bool = True,
    for_deep_learning: bool = False,
    for_transformers: bool = False
) -> TextPreprocessor:
    """
    Factory function to get appropriate preprocessor configuration.

    Args:
        data_type: Type of data ('email', 'sms', 'general')
        for_classical_ml: Optimized for classical ML models (TF-IDF, etc.)
        for_deep_learning: Optimized for deep learning (LSTM, CNN)
        for_transformers: Optimized for transformer models (minimal preprocessing)

    Returns:
        Configured TextPreprocessor instance
    """
    if for_transformers:
        # Minimal preprocessing for transformers
        return TextPreprocessor(
            lowercase=False,
            remove_html=True,
            replace_urls=True,
            replace_emails=True,
            remove_numbers=False,
            remove_punctuation=False,
            remove_stopwords=False,
            lemmatize=False
        )

    if for_deep_learning:
        # Moderate preprocessing for deep learning
        return TextPreprocessor(
            lowercase=True,
            remove_html=True,
            replace_urls=True,
            replace_emails=True,
            replace_numbers=True,
            remove_punctuation=False,
            remove_stopwords=False,
            lemmatize=False
        )

    # Classical ML preprocessing
    if data_type == 'email':
        return EmailPreprocessor(
            lowercase=True,
            remove_stopwords=True,
            lemmatize=True,
            remove_punctuation=True
        )
    elif data_type == 'sms':
        return SMSPreprocessor(
            lowercase=True,
            remove_stopwords=False,  # Keep for short messages
            lemmatize=False
        )
    else:
        return TextPreprocessor(
            lowercase=True,
            remove_stopwords=True,
            lemmatize=True,
            remove_punctuation=True
        )


def get_ml_preprocessor() -> TextPreprocessor:
    """
    Factory function that returns a preprocessor configured consistently
    with the training pipeline.

    IMPORTANT: Use this function for ALL prediction/inference tasks to ensure
    the same preprocessing is applied as during training.

    The configuration matches exactly what is used in TrainingPipeline._preprocess():
    - lowercase: True
    - remove_html: True
    - replace_urls: True
    - replace_emails: True
    - replace_numbers: True
    - remove_extra_whitespace: True

    Returns:
        TextPreprocessor configured for ML inference
    """
    return TextPreprocessor(
        lowercase=True,
        remove_html=True,
        replace_urls=True,
        replace_emails=True,
        replace_numbers=True,
        remove_extra_whitespace=True
    )


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(
        lowercase=True,
        replace_urls=True,
        replace_emails=True,
        remove_extra_whitespace=True
    )

    # Test with sample texts
    test_texts = [
        "URGENT! Your account at http://bank.com has been LOCKED! Contact support@bank.com NOW!",
        "Hey, check out this link: https://bit.ly/fake-link - You won $1000!!!",
        "Meeting tomorrow at 3pm. Let me know if that works for you.",
    ]

    print("Original texts and their preprocessed versions:\n")
    for text in test_texts:
        clean = preprocessor.preprocess(text)
        features = preprocessor.extract_features_count(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {clean}")
        print(f"Features: {features}")
        print()
