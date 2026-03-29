"""
Tests for the evaluation module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.7, 0.1, 0.4, 0.85, 0.25])

    def test_evaluate_basic_metrics(self):
        """Test calculation of basic metrics."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            y_true=self.y_true,
            y_pred=self.y_pred,
            model_name="test_model"
        )

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1

    def test_evaluate_with_proba(self):
        """Test evaluation with probability scores."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_proba=self.y_proba,
            model_name="test_model"
        )

        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        evaluator = ModelEvaluator()
        cm = evaluator.get_confusion_matrix(self.y_true, self.y_pred)

        assert cm is not None
        assert cm.shape == (2, 2)
        assert cm.sum() == len(self.y_true)

    def test_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        evaluator = ModelEvaluator()
        cm = evaluator.get_confusion_matrix(self.y_true, self.y_pred, normalize='true')

        assert cm is not None
        # Each row should sum to approximately 1
        assert np.allclose(cm.sum(axis=1), 1.0)

    def test_classification_report(self):
        """Test classification report generation."""
        evaluator = ModelEvaluator()
        report = evaluator.get_classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True
        )

        assert isinstance(report, dict)
        assert '0' in report or 'Legitimate' in report
        assert '1' in report or 'Phishing/Smishing' in report

    def test_roc_curve(self):
        """Test ROC curve calculation."""
        evaluator = ModelEvaluator()
        fpr, tpr, thresholds = evaluator.get_roc_curve(self.y_true, self.y_proba)

        assert len(fpr) > 0
        assert len(tpr) > 0
        assert len(thresholds) > 0
        assert fpr[0] == 0
        assert tpr[-1] == 1

    def test_precision_recall_curve(self):
        """Test Precision-Recall curve calculation."""
        evaluator = ModelEvaluator()
        precision, recall, thresholds = evaluator.get_precision_recall_curve(
            self.y_true, self.y_proba
        )

        assert len(precision) > 0
        assert len(recall) > 0
        assert len(thresholds) > 0

    def test_optimal_threshold(self):
        """Test optimal threshold finding."""
        evaluator = ModelEvaluator()
        threshold, score = evaluator.find_optimal_threshold(
            self.y_true, self.y_proba, metric='f1'
        )

        assert 0 < threshold < 1
        assert 0 <= score <= 1

    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator()

        # Evaluate two models
        evaluator.evaluate(self.y_true, self.y_pred, model_name="model1")
        evaluator.evaluate(self.y_true, self.y_pred, model_name="model2")

        comparison = evaluator.compare_models()

        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'f1' in comparison.columns

    def test_get_best_model(self):
        """Test getting best model."""
        evaluator = ModelEvaluator()

        # Evaluate models with different performance
        y_pred_good = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # Better
        y_pred_bad = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])   # Worse

        evaluator.evaluate(self.y_true, y_pred_good, model_name="good_model")
        evaluator.evaluate(self.y_true, y_pred_bad, model_name="bad_model")

        best_model, best_score = evaluator.get_best_model(metric='f1')

        assert best_model == "good_model"

    def test_results_storage(self):
        """Test that results are stored correctly."""
        evaluator = ModelEvaluator()
        evaluator.evaluate(
            self.y_true,
            self.y_pred,
            self.y_proba,
            model_name="test_model"
        )

        assert "test_model" in evaluator.results
        assert 'metrics' in evaluator.results["test_model"]
        assert 'confusion_matrix' in evaluator.results["test_model"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
