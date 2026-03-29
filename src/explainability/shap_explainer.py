"""
Explicador basado en SHAP (SHapley Additive exPlanations).

SHAP proporciona explicaciones teóricamente fundamentadas basadas en
valores de Shapley de teoría de juegos.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from .base import BaseExplainer

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    Explicador basado en SHAP para modelos de detección de phishing/smishing.

    Soporta diferentes tipos de explicadores según el modelo:
    - TreeExplainer: Para modelos basados en árboles (XGBoost, Random Forest) - Rápido
    - LinearExplainer: Para modelos lineales (Logistic Regression) - Muy rápido
    - KernelExplainer: Universal, pero más lento
    """

    def __init__(
        self,
        model,
        vectorizer,
        char_vectorizer=None,
        feature_extractor=None,
        preprocessor=None,
        feature_names: Optional[List[str]] = None,
        url_feature_names: Optional[List[str]] = None,
        model_type: str = 'auto',
        background_data: Optional[np.ndarray] = None
    ):
        """
        Inicializa el explicador SHAP.

        Args:
            model: Modelo entrenado
            vectorizer: Vectorizador TF-IDF (word)
            char_vectorizer: Vectorizador TF-IDF (char) opcional
            feature_extractor: Extractor de features URL
            preprocessor: Preprocesador de texto
            feature_names: Nombres de features TF-IDF
            url_feature_names: Nombres de features URL
            model_type: 'tree', 'linear', 'kernel', o 'auto' para detectar
            background_data: Datos de fondo para KernelExplainer
        """
        super().__init__(
            model=model,
            vectorizer=vectorizer,
            char_vectorizer=char_vectorizer,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            feature_names=feature_names,
            url_feature_names=url_feature_names
        )

        self.model_type = model_type
        self.background_data = background_data
        self.shap_explainer = None
        self._shap_available = False

        self._init_shap_explainer()

    def _detect_model_type(self) -> str:
        """Detecta automáticamente el tipo de modelo."""
        model_class = type(self.model).__name__.lower()

        # Si es CalibratedClassifierCV, extraer el tipo del estimador base
        if 'calibrated' in model_class:
            base_estimator = getattr(self.model, 'estimator', None) or getattr(self.model, 'base_estimator', None)
            if base_estimator is not None:
                base_class = type(base_estimator).__name__.lower()
                # LinearSVC es lineal (tiene coef_)
                if 'linearsvc' in base_class or 'linearsvm' in base_class:
                    return 'linear'
                elif 'svc' in base_class or 'svm' in base_class:
                    return 'kernel'

        if any(name in model_class for name in ['xgb', 'lightgbm', 'catboost', 'forest', 'tree', 'gradient']):
            return 'tree'
        elif any(name in model_class for name in ['logistic', 'linear', 'ridge', 'lasso']):
            return 'linear'
        elif 'svm' in model_class or 'svc' in model_class:
            return 'kernel'
        elif 'naive' in model_class or 'bayes' in model_class:
            # Naive Bayes no tiene buen soporte SHAP nativo, usar fallback
            return 'fallback'
        else:
            return 'kernel'

    def _init_shap_explainer(self):
        """Inicializa el explicador SHAP apropiado."""
        try:
            import shap
            self._shap_available = True
        except ImportError:
            logger.warning("SHAP no está instalado. Instalar con: pip install shap")
            self._shap_available = False
            return

        if self.model_type == 'auto':
            self.model_type = self._detect_model_type()

        logger.info(f"Inicializando SHAP con tipo de modelo: {self.model_type}")

        try:
            if self.model_type == 'tree':
                # TreeExplainer - muy rápido para modelos de árboles
                self.shap_explainer = shap.TreeExplainer(self.model)

            elif self.model_type == 'linear':
                # LinearExplainer - para modelos lineales
                if self.background_data is not None:
                    self.shap_explainer = shap.LinearExplainer(
                        self.model,
                        self.background_data
                    )
                else:
                    # Sin datos de fondo, intentar crear explainer básico
                    self.shap_explainer = shap.LinearExplainer(
                        self.model,
                        np.zeros((1, len(self.tfidf_feature_names) + len(self.url_feature_names)))
                    )

            elif self.model_type == 'kernel':
                # KernelExplainer - universal pero lento
                if self.background_data is not None:
                    # Usar muestra pequeña de background
                    if len(self.background_data) > 100:
                        background_sample = shap.sample(self.background_data, 100)
                    else:
                        background_sample = self.background_data

                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        background_sample
                    )
                else:
                    logger.warning(
                        "KernelExplainer requiere background_data. "
                        "Las explicaciones pueden no estar disponibles."
                    )
                    self.shap_explainer = None

            elif self.model_type == 'fallback':
                # Para modelos sin buen soporte SHAP (ej. Naive Bayes)
                # Usar explicación basada en coeficientes/importancias
                logger.info("Usando explicación fallback para este tipo de modelo")
                self.shap_explainer = None

        except Exception as e:
            logger.error(f"Error inicializando SHAP explainer: {e}")
            self.shap_explainer = None

    def explain(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Genera explicación SHAP para el texto.

        Args:
            text: Texto a explicar

        Returns:
            Dict con:
                - shap_values: Array de valores SHAP
                - base_value: Valor base del modelo
                - feature_importance: Dict {feature_name: shap_value}
                - prediction: Predicción del modelo (0 o 1)
                - probability: Probabilidad de amenaza
                - top_positive: Top features que aumentan riesgo
                - top_negative: Top features que reducen riesgo
        """
        if not self._shap_available or self.shap_explainer is None:
            return self._fallback_explanation(text)

        try:
            # Preparar input
            X, url_features = self._prepare_input(text)

            # Calcular valores SHAP
            shap_values = self.shap_explainer.shap_values(X)

            # Manejar diferentes formatos de salida SHAP
            if isinstance(shap_values, list):
                # Para clasificación binaria, usar clase positiva (índice 1)
                shap_values_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_class = shap_values

            # Asegurar que es array 1D
            if shap_values_class.ndim > 1:
                shap_values_class = shap_values_class[0]

            # Obtener valor base
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
                else:
                    base_value = float(base_value)
            else:
                base_value = 0.5

            # Crear mapa de feature -> valor SHAP
            feature_importance = {}
            n_tfidf = len(self.tfidf_feature_names)

            for i, shap_val in enumerate(shap_values_class):
                if abs(shap_val) > 1e-6:  # Solo features con impacto significativo
                    feature_name = self._get_feature_name(i)
                    feature_importance[feature_name] = float(shap_val)

            # Separar en positivos y negativos
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            top_positive = [
                (name, val) for name, val in sorted_features if val > 0
            ][:10]

            top_negative = [
                (name, val) for name, val in sorted_features if val < 0
            ][:10]

            # Predicción
            proba = self.model.predict_proba(X)[0]
            prediction = int(proba[1] > 0.5)

            return {
                'shap_values': shap_values_class.tolist(),
                'base_value': float(base_value),
                'feature_importance': feature_importance,
                'prediction': prediction,
                'probability': float(proba[1]),
                'prediction_contribution': float(np.sum(shap_values_class)),
                'top_positive': top_positive,
                'top_negative': top_negative,
                'method': 'shap',
                'model_type': self.model_type
            }

        except Exception as e:
            logger.error(f"Error en explicación SHAP: {e}")
            return self._fallback_explanation(text)

    def _fallback_explanation(self, text: str) -> Dict[str, Any]:
        """
        Explicación de fallback cuando SHAP no está disponible.

        Usa importancia de features del modelo si está disponible.
        """
        X, url_features = self._prepare_input(text)
        proba = self.model.predict_proba(X)[0]
        prediction = int(proba[1] > 0.5)

        feature_importance = {}

        # Intentar obtener feature_importances_ del modelo
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, imp in enumerate(importances):
                if imp > 0.001:
                    feature_name = self._get_feature_name(i)
                    feature_importance[feature_name] = float(imp)

        elif hasattr(self.model, 'coef_'):
            # Para modelos lineales
            coefs = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            for i, coef in enumerate(coefs):
                if abs(coef) > 0.01:
                    feature_name = self._get_feature_name(i)
                    feature_importance[feature_name] = float(coef)

        elif hasattr(self.model, 'calibrated_classifiers_'):
            # CalibratedClassifierCV con estimador base que tiene coef_
            try:
                base = self.model.calibrated_classifiers_[0].estimator
                if hasattr(base, 'coef_'):
                    coefs = base.coef_[0] if base.coef_.ndim > 1 else base.coef_
                    for i, coef in enumerate(coefs):
                        if abs(coef) > 0.01:
                            feature_name = self._get_feature_name(i)
                            feature_importance[feature_name] = float(coef)
            except (AttributeError, IndexError):
                pass

        return {
            'shap_values': [],
            'base_value': 0.5,
            'feature_importance': feature_importance,
            'prediction': prediction,
            'probability': float(proba[1]),
            'prediction_contribution': 0.0,
            'top_positive': [],
            'top_negative': [],
            'method': 'fallback',
            'model_type': self.model_type
        }

    def get_top_features(self, text: str, n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Obtiene las N características más importantes.

        Args:
            text: Texto a analizar
            n: Número de características

        Returns:
            Lista de (nombre, valor_shap, dirección)
        """
        result = self.explain(text)

        features = []

        # Añadir positivos
        for name, value in result.get('top_positive', [])[:n // 2]:
            features.append((name, value, 'positive'))

        # Añadir negativos
        for name, value in result.get('top_negative', [])[:n // 2]:
            features.append((name, abs(value), 'negative'))

        # Ordenar por valor absoluto
        features.sort(key=lambda x: abs(x[1]), reverse=True)

        return features[:n]

    def get_word_features_only(self, text: str, n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Obtiene solo features de palabras (excluye URL features).

        Args:
            text: Texto a analizar
            n: Número de características

        Returns:
            Lista de (palabra, valor_shap, dirección)
        """
        all_features = self.get_top_features(text, n=n * 2)
        return self._filter_word_features(all_features)[:n]
