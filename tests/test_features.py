"""
Tests for the feature extraction modules.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.text_features import TextFeatureExtractor
from src.features.sms_features import SMSFeatureExtractor
from src.features.email_features import EmailFeatureExtractor


class TestTextFeatureExtractor:
    """Test cases for TextFeatureExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.texts = [
            "urgent verify your account now",
            "meeting tomorrow at 3pm",
            "you won a free prize claim now",
            "project update attached",
            "bank account suspended verify"
        ]

    def test_tfidf_fit_transform(self):
        """Test TF-IDF feature extraction."""
        extractor = TextFeatureExtractor(
            method='tfidf',
            max_features=100,
            ngram_range=(1, 1)
        )

        features = extractor.fit_transform(self.texts)

        assert features is not None
        assert features.shape[0] == len(self.texts)
        assert features.shape[1] <= 100

    def test_transform_after_fit(self):
        """Test transform on new data after fitting."""
        extractor = TextFeatureExtractor(max_features=100)
        extractor.fit(self.texts)

        new_texts = ["verify your account", "meeting reminder"]
        features = extractor.transform(new_texts)

        assert features.shape[0] == 2
        assert features.shape[1] <= 100

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        extractor = TextFeatureExtractor()

        with pytest.raises(ValueError):
            extractor.transform(["test text"])

    def test_with_char_ngrams(self):
        """Test feature extraction with character n-grams."""
        extractor = TextFeatureExtractor(
            max_features=100,
            use_char_ngrams=True,
            char_max_features=50
        )

        features = extractor.fit_transform(self.texts)

        assert features is not None
        assert features.shape[0] == len(self.texts)

    def test_get_feature_names(self):
        """Test getting feature names."""
        extractor = TextFeatureExtractor(max_features=50)
        extractor.fit(self.texts)

        names = extractor.get_feature_names()
        assert len(names) > 0
        assert isinstance(names[0], str)


class TestSMSFeatureExtractor:
    """Test cases for SMSFeatureExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = SMSFeatureExtractor()

    def test_url_features(self):
        """Test URL feature extraction."""
        text = "Click here: http://bit.ly/suspicious-link"
        features = self.extractor.extract_url_features(text)

        assert features['url_count'] >= 1
        assert features['has_url'] == 1
        assert features['has_shortened_url'] == 1

    def test_phone_features(self):
        """Test phone number feature extraction."""
        text = "Call 1-800-123-4567 now!"
        features = self.extractor.extract_phone_features(text)

        assert features['phone_count'] >= 1
        assert features['has_phone'] == 1

    def test_keyword_features(self):
        """Test keyword feature extraction."""
        text = "URGENT: You won a FREE prize! Act NOW!"
        features = self.extractor.extract_keyword_features(text)

        assert features['urgency_count'] > 0
        assert features['promo_count'] > 0
        assert features['has_urgency'] == 1
        assert features['has_promo'] == 1

    def test_character_features(self):
        """Test character-level feature extraction."""
        text = "HELLO!!! Test message 123 😀"
        features = self.extractor.extract_character_features(text)

        assert features['exclamation_count'] == 3
        assert features['uppercase_ratio'] > 0

    def test_all_features(self):
        """Test extraction of all features."""
        text = "URGENT: Click bit.ly/prize to claim $1000! Call 1-800-SCAM"
        features = self.extractor.extract_all_features(text)

        assert isinstance(features, dict)
        assert len(features) > 10  # Should have many features
        assert 'url_count' in features
        assert 'urgency_count' in features

    def test_batch_extraction(self):
        """Test batch feature extraction."""
        texts = [
            "Click here for free prize!",
            "Meeting at 3pm tomorrow",
            "Your bank account is locked"
        ]

        df = self.extractor.extract_features_batch(texts)

        assert len(df) == 3
        assert 'url_count' in df.columns

    def test_empty_text(self):
        """Test handling of empty text."""
        features = self.extractor.extract_all_features("")

        assert features['char_count'] == 0
        assert features['url_count'] == 0


class TestEmailFeatureExtractor:
    """Test cases for EmailFeatureExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EmailFeatureExtractor()

    def test_url_features(self):
        """Test URL feature extraction from email."""
        text = "Click http://suspicious-bank.xyz to verify your account"
        features = self.extractor.extract_url_features(text)

        assert features['url_count'] >= 1
        assert features['has_urls'] == 1
        assert features['has_suspicious_tld'] == 1

    def test_header_features(self):
        """Test email header feature extraction."""
        headers = {
            'From': 'security@paypa1.xyz',
            'Reply-To': 'different@domain.com'
        }
        features = self.extractor.extract_header_features(headers)

        assert features['reply_to_mismatch'] == 1
        assert features['sender_has_suspicious_tld'] == 1

    def test_content_features(self):
        """Test content feature extraction."""
        subject = "URGENT: Your account has been suspended!"
        body = "Click here immediately to verify your account. Act now!"

        features = self.extractor.extract_content_features(subject, body)

        assert features['urgency_keyword_count'] > 0
        assert features['subject_has_urgency'] == 1
        assert features['exclamation_count'] >= 1

    def test_all_features(self):
        """Test extraction of all email features."""
        email_data = {
            'subject': 'URGENT: Verify your account',
            'body': 'Click http://fake-bank.xyz to verify. Act now!',
            'from': 'security@fake-bank.xyz',
            'headers': {'From': 'security@fake-bank.xyz'}
        }

        features = self.extractor.extract_all_features(email_data)

        assert isinstance(features, dict)
        assert len(features) > 10
        assert 'url_count' in features
        assert 'urgency_keyword_count' in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
