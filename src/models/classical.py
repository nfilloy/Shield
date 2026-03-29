"""
Classical Machine Learning models for phishing/smishing detection.

Implements baseline and traditional ML models:
- Naive Bayes (MultinomialNB, ComplementNB)
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- XGBoost
- Voting Ensemble
"""

import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")


class ClassicalModels:
    """
    Collection of classical ML models for phishing/smishing detection.

    Provides a unified interface for training, evaluating, and comparing
    multiple traditional machine learning models.
    """

    # Default hyperparameters for each model
    DEFAULT_PARAMS = {
        'naive_bayes': {
            'alpha': 1.0,
            'fit_prior': True
        },
        'complement_nb': {
            'alpha': 1.0,
            'fit_prior': True,
            'norm': False
        },
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'random_state': 42
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        },
        'linear_svm': {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 10000,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
    }

    # Hyperparameter grid for tuning
    PARAM_GRIDS = {
        'naive_bayes': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'linear_svm': {
            'C': [0.01, 0.1, 1.0, 10.0]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    }

    def __init__(self, model_dir: str = "models"):
        """
        Initialize ClassicalModels.

        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.cv_scores: Dict[str, Dict] = {}

    def get_model(
        self,
        model_name: str,
        params: Optional[Dict] = None,
        calibrated: bool = False
    ) -> Any:
        """
        Get a model instance by name.

        Args:
            model_name: Name of the model
            params: Custom parameters (uses defaults if None)
            calibrated: Wrap model with probability calibration

        Returns:
            Model instance
        """
        if params is None:
            params = self.DEFAULT_PARAMS.get(model_name, {})

        if model_name == 'naive_bayes':
            model = MultinomialNB(**params)
        elif model_name == 'complement_nb':
            model = ComplementNB(**params)
        elif model_name == 'logistic_regression':
            model = LogisticRegression(**params)
        elif model_name == 'svm':
            # SVM es sensible a la escala de features, usar Pipeline con StandardScaler
            base_svm = SVC(**params)
            model = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),  # with_mean=False para sparse matrices
                ('svm', base_svm)
            ])
        elif model_name == 'linear_svm':
            base_model = LinearSVC(**params)
            # LinearSVC doesn't have predict_proba, so calibrate
            model = CalibratedClassifierCV(base_model, cv=3)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(**params)
        elif model_name == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed")
            model = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if calibrated and model_name not in ['linear_svm', 'svm']:
            model = CalibratedClassifierCV(model, cv=3)

        return model

    def train(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict] = None
    ) -> Any:
        """
        Train a model.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            params: Custom parameters

        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")

        model = self.get_model(model_name, params)
        model.fit(X_train, y_train)

        self.models[model_name] = model
        return model

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models.

        Args:
            X_train: Training features
            y_train: Training labels
            models: List of model names (trains all if None)

        Returns:
            Dictionary of trained models
        """
        if models is None:
            models = ['naive_bayes', 'logistic_regression', 'linear_svm',
                      'random_forest']
            if XGBOOST_AVAILABLE:
                models.append('xgboost')

        for model_name in models:
            try:
                self.train(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        return self.models

    def tune_hyperparameters(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'f1'
    ) -> Tuple[Any, Dict]:
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid (uses default if None)
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info(f"Tuning hyperparameters for {model_name}...")

        if param_grid is None:
            param_grid = self.PARAM_GRIDS.get(model_name, {})

        base_model = self.get_model(model_name)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params[model_name] = grid_search.best_params_
        self.models[model_name] = grid_search.best_estimator_

        logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation for a model.

        Args:
            model_name: Name of the model
            X: Features
            y: Labels
            cv: Number of folds
            scoring: List of scoring metrics

        Returns:
            Dictionary of scores for each metric
        """
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        model = self.models.get(model_name) or self.get_model(model_name)

        logger.info(f"Cross-validating {model_name}...")

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = {}

        for metric in scoring:
            try:
                metric_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=metric)
                scores[metric] = metric_scores
                logger.info(f"  {metric}: {metric_scores.mean():.4f} (+/- {metric_scores.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"  {metric}: Failed - {e}")
                scores[metric] = np.array([np.nan] * cv)

        self.cv_scores[model_name] = scores
        return scores

    def cross_validate_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: Optional[List[str]] = None,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Cross-validate multiple models and return comparison DataFrame.

        Args:
            X: Features
            y: Labels
            models: List of model names
            cv: Number of folds

        Returns:
            DataFrame with cross-validation results
        """
        if models is None:
            models = list(self.models.keys())

        results = []
        for model_name in models:
            if model_name not in self.models:
                self.train(model_name, X, y)

            scores = self.cross_validate(model_name, X, y, cv=cv)

            result = {'model': model_name}
            for metric, values in scores.items():
                result[f'{metric}_mean'] = values.mean()
                result[f'{metric}_std'] = values.std()

            results.append(result)

        return pd.DataFrame(results)

    def predict(
        self,
        model_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with a trained model.

        Args:
            model_name: Name of the model
            X: Features

        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        return self.models[model_name].predict(X)

    def predict_proba(
        self,
        model_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            model_name: Name of the model
            X: Features

        Returns:
            Probability predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise AttributeError(f"Model {model_name} does not support predict_proba")

    def create_ensemble(
        self,
        model_names: List[str],
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> VotingClassifier:
        """
        Create an ensemble of trained models.

        Args:
            model_names: List of model names to include
            voting: 'hard' or 'soft' voting
            weights: Weights for each model

        Returns:
            VotingClassifier ensemble
        """
        estimators = []
        for name in model_names:
            if name not in self.models:
                raise ValueError(f"Model {name} not trained")
            estimators.append((name, self.models[name]))

        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )

        return ensemble

    def get_feature_importance(
        self,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.

        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            raise AttributeError(f"Model {model_name} does not have feature importances")

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        return df.nlargest(top_n, 'importance')

    def save_model(self, model_name: str, filename: Optional[str] = None):
        """
        Save a trained model to disk.

        Args:
            model_name: Name of the model
            filename: Custom filename
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        if filename is None:
            filename = f"{model_name}.pkl"

        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)

        logger.info(f"Saved {model_name} to {filepath}")

    def load_model(self, model_name: str, filename: Optional[str] = None):
        """
        Load a model from disk.

        Args:
            model_name: Name to assign to loaded model
            filename: Filename to load from
        """
        if filename is None:
            filename = f"{model_name}.pkl"

        filepath = self.model_dir / filename
        with open(filepath, 'rb') as f:
            self.models[model_name] = pickle.load(f)

        logger.info(f"Loaded {model_name} from {filepath}")

    def save_all_models(self):
        """Save all trained models."""
        for model_name in self.models:
            self.save_model(model_name)

    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.

        Returns:
            DataFrame with model information
        """
        summaries = []
        for name, model in self.models.items():
            summary = {
                'model': name,
                'type': type(model).__name__,
                'params': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
            }

            if name in self.cv_scores:
                for metric, scores in self.cv_scores[name].items():
                    summary[f'{metric}_cv'] = f"{scores.mean():.4f}"

            summaries.append(summary)

        return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train models
    models = ClassicalModels()

    # Train all models
    models.train_all(X_train, y_train)

    # Cross-validate
    cv_results = models.cross_validate_all(X_train, y_train)
    print("\nCross-validation results:")
    print(cv_results.to_string())

    # Feature importance (for Random Forest)
    if 'random_forest' in models.models:
        importance = models.get_feature_importance('random_forest')
        print("\nTop feature importances (Random Forest):")
        print(importance.head(10).to_string())
