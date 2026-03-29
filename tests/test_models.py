"""
Tests for the machine learning models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classical import ClassicalModels


class TestClassicalModels:
    """Test cases for ClassicalModels class."""

    def setup_method(self):
        """Set up test fixtures with sample data."""
        np.random.seed(42)
        self.X_train = np.random.rand(100, 50)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 50)
        self.y_test = np.random.randint(0, 2, 20)

    def test_get_model_naive_bayes(self):
        """Test getting Naive Bayes model."""
        models = ClassicalModels()
        model = models.get_model('naive_bayes')
        assert model is not None

    def test_get_model_logistic_regression(self):
        """Test getting Logistic Regression model."""
        models = ClassicalModels()
        model = models.get_model('logistic_regression')
        assert model is not None

    def test_get_model_random_forest(self):
        """Test getting Random Forest model."""
        models = ClassicalModels()
        model = models.get_model('random_forest')
        assert model is not None

    def test_get_model_unknown_raises(self):
        """Test that unknown model name raises error."""
        models = ClassicalModels()
        with pytest.raises(ValueError):
            models.get_model('unknown_model')

    def test_train_naive_bayes(self):
        """Test training Naive Bayes model."""
        models = ClassicalModels()
        # Naive Bayes requires non-negative features
        X_pos = np.abs(self.X_train)
        model = models.train('naive_bayes', X_pos, self.y_train)

        assert model is not None
        assert 'naive_bayes' in models.models

    def test_train_logistic_regression(self):
        """Test training Logistic Regression model."""
        models = ClassicalModels()
        model = models.train('logistic_regression', self.X_train, self.y_train)

        assert model is not None
        assert 'logistic_regression' in models.models

    def test_train_random_forest(self):
        """Test training Random Forest model."""
        models = ClassicalModels()
        model = models.train('random_forest', self.X_train, self.y_train)

        assert model is not None
        assert 'random_forest' in models.models

    def test_predict(self):
        """Test making predictions."""
        models = ClassicalModels()
        models.train('logistic_regression', self.X_train, self.y_train)

        predictions = models.predict('logistic_regression', self.X_test)

        assert predictions is not None
        assert len(predictions) == len(self.X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self):
        """Test getting prediction probabilities."""
        models = ClassicalModels()
        models.train('logistic_regression', self.X_train, self.y_train)

        probas = models.predict_proba('logistic_regression', self.X_test)

        assert probas is not None
        assert probas.shape[0] == len(self.X_test)
        assert probas.shape[1] == 2
        assert all(0 <= p <= 1 for row in probas for p in row)

    def test_predict_untrained_raises(self):
        """Test that predicting with untrained model raises error."""
        models = ClassicalModels()

        with pytest.raises(ValueError):
            models.predict('logistic_regression', self.X_test)

    def test_train_all(self):
        """Test training multiple models."""
        models = ClassicalModels()
        # Use absolute values for Naive Bayes compatibility
        X_pos = np.abs(self.X_train)

        trained = models.train_all(X_pos, self.y_train,
                                   models=['naive_bayes', 'logistic_regression'])

        assert len(trained) >= 2
        assert 'naive_bayes' in trained
        assert 'logistic_regression' in trained

    def test_cross_validate(self):
        """Test cross-validation."""
        models = ClassicalModels()
        models.train('logistic_regression', self.X_train, self.y_train)

        scores = models.cross_validate(
            'logistic_regression',
            self.X_train,
            self.y_train,
            cv=3
        )

        assert 'accuracy' in scores
        assert 'f1' in scores
        assert len(scores['accuracy']) == 3

    def test_feature_importance(self):
        """Test getting feature importance."""
        models = ClassicalModels()
        models.train('random_forest', self.X_train, self.y_train)

        importance_df = models.get_feature_importance('random_forest', top_n=10)

        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


class TestModelEnsemble:
    """Test cases for model ensemble functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X_train = np.random.rand(100, 50)
        self.y_train = np.random.randint(0, 2, 100)

    def test_create_ensemble(self):
        """Test creating voting ensemble."""
        models = ClassicalModels()
        models.train('logistic_regression', self.X_train, self.y_train)
        models.train('random_forest', self.X_train, self.y_train)

        ensemble = models.create_ensemble(
            ['logistic_regression', 'random_forest'],
            voting='soft'
        )

        assert ensemble is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
