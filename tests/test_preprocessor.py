"""
Tests for the text preprocessing module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import TextPreprocessor, get_preprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            replace_urls=True,
            replace_emails=True,
            remove_extra_whitespace=True
        )

    def test_lowercase(self):
        """Test lowercase conversion."""
        text = "HELLO WORLD"
        result = self.preprocessor.preprocess(text)
        assert result == "hello world"

    def test_url_replacement(self):
        """Test URL replacement with token."""
        text = "Visit https://example.com for more info"
        result = self.preprocessor.preprocess(text)
        assert "<url>" in result.lower()
        assert "https://example.com" not in result

    def test_email_replacement(self):
        """Test email replacement with token."""
        text = "Contact us at support@example.com"
        result = self.preprocessor.preprocess(text)
        assert "<email>" in result.lower()
        assert "support@example.com" not in result

    def test_whitespace_normalization(self):
        """Test multiple whitespace normalization."""
        text = "Hello    World   Test"
        result = self.preprocessor.preprocess(text)
        assert "    " not in result
        assert "   " not in result

    def test_empty_text(self):
        """Test handling of empty text."""
        assert self.preprocessor.preprocess("") == ""
        assert self.preprocessor.preprocess(None) == ""

    def test_phone_replacement(self):
        """Test phone number replacement."""
        preprocessor = TextPreprocessor(replace_numbers=True)
        text = "Call us at 1-800-123-4567"
        result = preprocessor.preprocess(text)
        assert "<phone>" in result.lower() or "<num>" in result.lower()

    def test_feature_extraction(self):
        """Test feature extraction from text."""
        text = "URGENT! Click http://suspicious.com to win $1000!"
        features = self.preprocessor.extract_features_count(text)

        assert features['url_count'] >= 1
        assert features['urgency_word_count'] >= 1
        assert features['exclamation_count'] >= 1
        assert features['char_count'] == len(text)


class TestGetPreprocessor:
    """Test cases for get_preprocessor factory function."""

    def test_get_preprocessor_sms(self):
        """Test getting SMS-specific preprocessor."""
        preprocessor = get_preprocessor(data_type='sms', for_classical_ml=True)
        assert preprocessor is not None
        assert preprocessor.lowercase is True

    def test_get_preprocessor_email(self):
        """Test getting email-specific preprocessor."""
        preprocessor = get_preprocessor(data_type='email', for_classical_ml=True)
        assert preprocessor is not None
        assert preprocessor.remove_html is True

    def test_get_preprocessor_transformers(self):
        """Test getting transformer-optimized preprocessor."""
        preprocessor = get_preprocessor(for_transformers=True)
        assert preprocessor is not None
        # Transformers need minimal preprocessing
        assert preprocessor.lowercase is False
        assert preprocessor.remove_stopwords is False


class TestPreprocessorBatch:
    """Test cases for batch preprocessing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor(lowercase=True)

    def test_batch_processing(self):
        """Test preprocessing multiple texts."""
        texts = ["HELLO", "WORLD", "TEST"]
        results = self.preprocessor.preprocess_batch(texts, show_progress=False)

        assert len(results) == 3
        assert results[0] == "hello"
        assert results[1] == "world"
        assert results[2] == "test"

    def test_batch_with_none(self):
        """Test batch processing with None values."""
        texts = ["HELLO", None, "TEST"]
        results = self.preprocessor.preprocess_batch(texts, show_progress=False)

        assert len(results) == 3
        assert results[0] == "hello"
        assert results[1] == ""
        assert results[2] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
