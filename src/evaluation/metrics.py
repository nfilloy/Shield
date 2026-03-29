"""
Evaluation metrics and utilities for phishing/smishing detection models.

Provides comprehensive evaluation including:
- Standard metrics (Accuracy, Precision, Recall, F1)
- ROC-AUC and PR-AUC curves
- Confusion matrix analysis
- Cross-validation utilities
- Model comparison tools
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite for phishing/smishing detection models.

    Provides methods for calculating metrics, generating reports,
    and comparing multiple models.
    """

    def __init__(self, output_dir: str = "reports/results"):
        """
        Initialize the ModelEvaluator.

        Args:
            output_dir: Directory for saving evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Dict] = {}
        self.comparison_df: Optional[pd.DataFrame] = None

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model",
        average: str = 'binary',
        pos_label: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model predictions and calculate all metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for ROC-AUC)
            model_name: Name for storing results
            average: Averaging method for multi-class
            pos_label: Positive class label

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)

        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix derived metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)

            # Specificity (True Negative Rate)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

            # False Positive Rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

            # False Negative Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # AUC metrics (require probabilities)
        if y_proba is not None:
            # Handle 2D probability arrays
            if len(y_proba.shape) == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba

            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = np.nan

            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
            except ValueError as e:
                logger.warning(f"Could not calculate PR-AUC: {e}")
                metrics['pr_auc'] = np.nan

        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'confusion_matrix': cm
        }

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Get confusion matrix with optional normalization.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization method ('true', 'pred', 'all', or None)

        Returns:
            Confusion matrix array
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: List[str] = None,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Get detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for each class
            output_dict: Return as dictionary

        Returns:
            Classification report (string or dict)
        """
        if target_names is None:
            target_names = ['Legitimate', 'Phishing/Smishing']

        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )

    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]

        return roc_curve(y_true, y_proba)

    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Precision-Recall curve.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]

        return precision_recall_curve(y_true, y_proba)

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1',
        min_recall: float = 0.0
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            metric: Metric to optimize ('f1', 'accuracy', 'youden')
            min_recall: Minimum required recall

        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            recall = recall_score(y_true, y_pred, zero_division=0)
            if recall < min_recall:
                continue

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'youden':
                # Youden's J statistic = Sensitivity + Specificity - 1
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        model_name: str = "model"
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation and return detailed metrics.

        Args:
            model: Sklearn-compatible model
            X: Features
            y: Labels
            cv: Number of folds
            model_name: Name for storing results

        Returns:
            Dictionary of metric arrays (one value per fold)
        """
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Get predictions for each fold
        y_pred = cross_val_predict(model, X, y, cv=cv_strategy)

        try:
            y_proba = cross_val_predict(model, X, y, cv=cv_strategy, method='predict_proba')
        except:
            y_proba = None

        # Calculate metrics per fold
        fold_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        for train_idx, val_idx in cv_strategy.split(X, y):
            y_val = y[val_idx]
            y_val_pred = y_pred[val_idx]

            fold_metrics['accuracy'].append(accuracy_score(y_val, y_val_pred))
            fold_metrics['precision'].append(precision_score(y_val, y_val_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_val, y_val_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_val, y_val_pred, zero_division=0))

            if y_proba is not None:
                y_val_proba = y_proba[val_idx]
                if len(y_val_proba.shape) == 2:
                    y_val_proba = y_val_proba[:, 1]
                try:
                    fold_metrics['roc_auc'].append(roc_auc_score(y_val, y_val_proba))
                except:
                    fold_metrics['roc_auc'].append(np.nan)

        # Convert to numpy arrays
        fold_metrics = {k: np.array(v) for k, v in fold_metrics.items()}

        # Store results
        self.results[f"{model_name}_cv"] = {
            'fold_metrics': fold_metrics,
            'mean_metrics': {k: v.mean() for k, v in fold_metrics.items()},
            'std_metrics': {k: v.std() for k, v in fold_metrics.items()}
        }

        return fold_metrics

    def compare_models(
        self,
        models_results: Dict[str, Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create comparison table of multiple models.

        Args:
            models_results: Dictionary of {model_name: metrics_dict}
                           If None, uses stored results

        Returns:
            Comparison DataFrame
        """
        if models_results is None:
            models_results = {
                name: data['metrics']
                for name, data in self.results.items()
                if 'metrics' in data
            }

        if not models_results:
            logger.warning("No results to compare")
            return pd.DataFrame()

        comparison = []
        for model_name, metrics in models_results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison.append(row)

        self.comparison_df = pd.DataFrame(comparison)

        # Sort by F1 score
        if 'f1' in self.comparison_df.columns:
            self.comparison_df = self.comparison_df.sort_values('f1', ascending=False)

        return self.comparison_df

    def get_best_model(
        self,
        metric: str = 'f1'
    ) -> Tuple[str, float]:
        """
        Get the best performing model by metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, metric_value)
        """
        if self.comparison_df is None or self.comparison_df.empty:
            self.compare_models()

        if self.comparison_df is None or metric not in self.comparison_df.columns:
            raise ValueError(f"Metric {metric} not found in results")

        best_idx = self.comparison_df[metric].idxmax()
        best_model = self.comparison_df.loc[best_idx, 'model']
        best_value = self.comparison_df.loc[best_idx, metric]

        return best_model, best_value

    def analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        texts: List[str],
        model_name: str = "model"
    ) -> pd.DataFrame:
        """
        Analyze misclassified samples.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Original texts
            model_name: Model name for reference

        Returns:
            DataFrame with error analysis
        """
        errors = []

        for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
            if true != pred:
                error_type = 'False Positive' if pred == 1 else 'False Negative'
                errors.append({
                    'index': i,
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'true_label': 'Phishing/Smishing' if true == 1 else 'Legitimate',
                    'pred_label': 'Phishing/Smishing' if pred == 1 else 'Legitimate',
                    'error_type': error_type
                })

        error_df = pd.DataFrame(errors)

        if not error_df.empty:
            logger.info(f"\nError Analysis for {model_name}:")
            logger.info(f"  Total errors: {len(error_df)}")
            logger.info(f"  False Positives: {(error_df['error_type'] == 'False Positive').sum()}")
            logger.info(f"  False Negatives: {(error_df['error_type'] == 'False Negative').sum()}")

        return error_df

    def save_results(
        self,
        model_name: str,
        filename: Optional[str] = None
    ):
        """
        Save evaluation results to file.

        Args:
            model_name: Name of the model
            filename: Output filename
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")

        if filename is None:
            filename = f"{model_name}_results.json"

        filepath = self.output_dir / filename

        # Prepare serializable results
        results = self.results[model_name].copy()
        results.pop('y_true', None)
        results.pop('y_pred', None)
        results.pop('y_proba', None)

        if 'confusion_matrix' in results:
            results['confusion_matrix'] = results['confusion_matrix'].tolist()

        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved results to {filepath}")

    def save_comparison(self, filename: str = "model_comparison.csv"):
        """Save model comparison to CSV."""
        if self.comparison_df is None:
            self.compare_models()

        if self.comparison_df is not None:
            filepath = self.output_dir / filename
            self.comparison_df.to_csv(filepath, index=False)
            logger.info(f"Saved comparison to {filepath}")

    def print_summary(self, model_name: str):
        """Print summary of evaluation results."""
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return

        results = self.results[model_name]
        metrics = results.get('metrics', {})

        print(f"\n{'='*50}")
        print(f"Evaluation Summary: {model_name}")
        print('='*50)

        print("\nClassification Metrics:")
        print(f"  Accuracy:    {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision:   {metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:      {metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score:    {metrics.get('f1', 'N/A'):.4f}")
        print(f"  MCC:         {metrics.get('mcc', 'N/A'):.4f}")

        if 'roc_auc' in metrics:
            print(f"\nAUC Metrics:")
            print(f"  ROC-AUC:     {metrics.get('roc_auc', 'N/A'):.4f}")
            print(f"  PR-AUC:      {metrics.get('pr_auc', 'N/A'):.4f}")

        if 'confusion_matrix' in results:
            print(f"\nConfusion Matrix:")
            cm = results['confusion_matrix']
            print(f"  TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
            print(f"  FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")

        print('='*50)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulated predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10)
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0] * 10)
    y_proba = np.random.rand(100)
    y_proba[y_true == 1] += 0.3  # Make probabilities somewhat correlated with true labels
    y_proba = np.clip(y_proba, 0, 1)

    # Create evaluator
    evaluator = ModelEvaluator()

    # Evaluate model
    metrics = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name="test_model"
    )

    # Print summary
    evaluator.print_summary("test_model")

    # Find optimal threshold
    threshold, score = evaluator.find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"\nOptimal threshold: {threshold:.2f} (F1: {score:.4f})")
