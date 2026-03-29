"""
Explicador basado en LIME (Local Interpretable Model-agnostic Explanations).

LIME genera explicaciones interpretables perturbando el texto
y observando cómo cambian las predicciones.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import re

from .base import BaseExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    Explicador basado en LIME para modelos de detección de phishing/smishing.

    LIME es model-agnostic (funciona con cualquier modelo) y genera
    explicaciones locales interpretables basadas en perturbaciones del texto.
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
        class_names: Optional[List[str]] = None
    ):
        """
        Inicializa el explicador LIME.

        Args:
            model: Modelo entrenado
            vectorizer: Vectorizador TF-IDF (word)
            char_vectorizer: Vectorizador TF-IDF (char) opcional
            feature_extractor: Extractor de features URL
            preprocessor: Preprocesador de texto
            feature_names: Nombres de features TF-IDF
            url_feature_names: Nombres de features URL
            class_names: Nombres de las clases ['SAFE', 'THREAT']
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

        self.class_names = class_names or ['SAFE', 'THREAT']
        self.lime_explainer = None
        self._lime_available = False

        self._init_lime_explainer()

    def _init_lime_explainer(self):
        """Inicializa el explicador LIME."""
        try:
            from lime.lime_text import LimeTextExplainer
            self._lime_available = True

            self.lime_explainer = LimeTextExplainer(
                class_names=self.class_names,
                split_expression=r'\W+',  # Dividir por caracteres no-palabra
                bow=True,  # Bag of words
                random_state=42
            )

            logger.info("LIME explainer inicializado correctamente")

        except ImportError:
            logger.warning("LIME no está instalado. Instalar con: pip install lime")
            self._lime_available = False

    def _predict_proba_for_lime(self, texts: List[str]) -> np.ndarray:
        """
        Función de predicción que LIME usa internamente.

        Esta función se llama muchas veces (~500) por cada explicación,
        así que debe ser eficiente.

        Args:
            texts: Lista de textos (perturbaciones)

        Returns:
            Array de shape (n_samples, n_classes) con probabilidades
        """
        results = []

        for text in texts:
            try:
                # Preprocesar
                if self.preprocessor is not None:
                    clean_text = self.preprocessor.preprocess(text)
                else:
                    clean_text = text

                # Vectorizar TF-IDF (word + char) usando método del padre
                X_tfidf = self._vectorize_text(clean_text)

                # Para perturbaciones de LIME, las URL features no cambian
                # (LIME perturba palabras, no URLs)
                # Usamos features de URL del texto original (guardadas)
                if hasattr(self, '_current_url_features') and self.url_feature_names:
                    X_url = np.array([[
                        self._current_url_features.get(name, 0)
                        for name in self.url_feature_names
                    ]])
                    X = np.hstack([X_tfidf, X_url])
                else:
                    X = X_tfidf

                # Predecir
                proba = self.model.predict_proba(X)[0]
                results.append(proba)

            except Exception as e:
                # En caso de error, retornar probabilidad neutral
                logger.debug(f"Error en predicción LIME: {e}")
                results.append(np.array([0.5, 0.5]))

        return np.array(results)

    def explain(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera explicación LIME para el texto.

        Args:
            text: Texto a explicar
            num_features: Número de palabras a incluir en la explicación
            num_samples: Número de perturbaciones (más = más preciso pero lento)

        Returns:
            Dict con:
                - word_weights: Lista de (palabra, peso, dirección)
                - prediction: Predicción del modelo (0 o 1)
                - probabilities: [prob_safe, prob_threat]
                - explanation_html: HTML de la explicación LIME
                - intercept: Intercept del modelo local
        """
        if not self._lime_available or self.lime_explainer is None:
            return self._fallback_explanation(text)

        try:
            # Guardar URL features del texto original para uso en perturbaciones
            if self.feature_extractor is not None:
                try:
                    self._current_url_features = self.feature_extractor.extract_all_features(text)
                except TypeError:
                    self._current_url_features = self.feature_extractor.extract_all_features({
                        'body': text, 'subject': '', 'headers': {}
                    })
            else:
                self._current_url_features = {}

            # Generar explicación LIME
            explanation = self.lime_explainer.explain_instance(
                text,
                self._predict_proba_for_lime,
                num_features=num_features,
                num_samples=num_samples
            )

            # Extraer pesos de palabras
            # LIME retorna lista de (palabra, peso) para la clase predicha
            word_weights_raw = explanation.as_list()

            # Convertir a formato estándar
            word_weights = []
            for word, weight in word_weights_raw:
                direction = 'positive' if weight > 0 else 'negative'
                word_weights.append((word, weight, direction))

            # Ordenar por valor absoluto
            word_weights.sort(key=lambda x: abs(x[1]), reverse=True)

            # Obtener predicción y probabilidades
            proba = self._predict_proba_for_lime([text])[0]
            prediction = int(proba[1] > 0.5)

            # Generar HTML de explicación
            try:
                explanation_html = explanation.as_html()
            except Exception:
                explanation_html = ""

            return {
                'word_weights': word_weights,
                'prediction': prediction,
                'probabilities': proba.tolist(),
                'explanation_html': explanation_html,
                'intercept': float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0.0,
                'score': float(explanation.score) if hasattr(explanation, 'score') else 0.0,
                'method': 'lime',
                'num_samples': num_samples
            }

        except Exception as e:
            logger.error(f"Error en explicación LIME: {e}")
            return self._fallback_explanation(text)

        finally:
            # Limpiar URL features guardadas
            self._current_url_features = {}

    def _fallback_explanation(self, text: str) -> Dict[str, Any]:
        """
        Explicación de fallback cuando LIME no está disponible.

        Usa análisis básico de palabras sospechosas.
        """
        X, url_features = self._prepare_input(text)
        proba = self.model.predict_proba(X)[0]
        prediction = int(proba[1] > 0.5)

        # Palabras sospechosas conocidas
        suspicious_words = {
            'urgent', 'verify', 'suspended', 'click', 'immediately',
            'account', 'winner', 'prize', 'free', 'claim', 'expires',
            'confirm', 'update', 'security', 'alert', 'warning',
            'limited', 'offer', 'act now', 'don\'t miss'
        }

        safe_words = {
            'thanks', 'thank', 'hello', 'hi', 'meeting', 'lunch',
            'dinner', 'tomorrow', 'today', 'love', 'friend', 'family'
        }

        # Analizar texto
        words_lower = set(re.findall(r'\w+', text.lower()))
        word_weights = []

        for word in words_lower:
            if word in suspicious_words:
                word_weights.append((word, 0.3, 'positive'))
            elif word in safe_words:
                word_weights.append((word, 0.2, 'negative'))

        return {
            'word_weights': word_weights,
            'prediction': prediction,
            'probabilities': proba.tolist(),
            'explanation_html': '',
            'intercept': 0.0,
            'score': 0.0,
            'method': 'fallback',
            'num_samples': 0
        }

    def get_top_features(self, text: str, n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Obtiene las N características más importantes.

        Args:
            text: Texto a analizar
            n: Número de características

        Returns:
            Lista de (palabra, peso, dirección)
        """
        result = self.explain(text, num_features=n)
        return result.get('word_weights', [])[:n]

    def get_html_explanation(self, text: str, num_features: int = 10) -> str:
        """
        Genera HTML con la explicación visual de LIME.

        Args:
            text: Texto a explicar
            num_features: Número de palabras a mostrar

        Returns:
            HTML con la explicación
        """
        result = self.explain(text, num_features=num_features)

        if result.get('explanation_html'):
            return result['explanation_html']

        # Generar HTML básico si no hay explicación LIME
        word_weights = result.get('word_weights', [])

        html_parts = ['<div style="font-family: monospace; padding: 1rem;">']
        html_parts.append('<h4>Palabras Importantes:</h4>')
        html_parts.append('<ul>')

        for word, weight, direction in word_weights[:num_features]:
            color = '#ff3366' if direction == 'positive' else '#39FF14'
            sign = '+' if direction == 'positive' else ''
            html_parts.append(
                f'<li style="color: {color};">{word}: {sign}{weight:.3f}</li>'
            )

        html_parts.append('</ul>')
        html_parts.append('</div>')

        return ''.join(html_parts)

    def explain_batch(
        self,
        texts: List[str],
        num_features: int = 10,
        num_samples: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Genera explicaciones para múltiples textos.

        Args:
            texts: Lista de textos
            num_features: Número de palabras por explicación
            num_samples: Muestras por explicación (reducido para batch)

        Returns:
            Lista de explicaciones
        """
        return [
            self.explain(text, num_features=num_features, num_samples=num_samples)
            for text in texts
        ]
