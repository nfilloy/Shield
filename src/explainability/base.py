"""
Clase base para explicadores de modelos ML.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from scipy.sparse import hstack, issparse


class BaseExplainer(ABC):
    """
    Clase base abstracta para explicadores de modelos.

    Proporciona la interfaz común para SHAP, LIME y otros métodos de explicabilidad.
    Soporta vectorizadores con word y char n-grams.
    """

    def __init__(
        self,
        model,
        vectorizer,
        char_vectorizer=None,
        feature_extractor=None,
        preprocessor=None,
        feature_names: Optional[List[str]] = None,
        url_feature_names: Optional[List[str]] = None
    ):
        """
        Inicializa el explicador base.

        Args:
            model: Modelo entrenado (sklearn/xgboost)
            vectorizer: Vectorizador TF-IDF para palabras (word_vectorizer)
            char_vectorizer: Vectorizador TF-IDF para caracteres (opcional)
            feature_extractor: Extractor de features URL (SMSFeatureExtractor o EmailFeatureExtractor)
            preprocessor: Preprocesador de texto
            feature_names: Lista de nombres de features TF-IDF (word + char)
            url_feature_names: Lista de nombres de features URL
        """
        self.model = model
        self.vectorizer = vectorizer
        self.char_vectorizer = char_vectorizer
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor

        # Construir nombres de features TF-IDF
        if feature_names is not None:
            self.tfidf_feature_names = feature_names
        else:
            self.tfidf_feature_names = self._build_tfidf_feature_names()

        self.url_feature_names = url_feature_names or []

        # Crear vocabulario inverso para mapeo rápido
        self._build_vocabulary_index()

    def _build_tfidf_feature_names(self) -> List[str]:
        """Construye lista de nombres de features TF-IDF (word + char)."""
        names = []

        # Word features
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            names.extend([f"word_{n}" for n in self.vectorizer.get_feature_names_out()])
        elif hasattr(self.vectorizer, 'get_feature_names'):
            names.extend([f"word_{n}" for n in self.vectorizer.get_feature_names()])

        # Char features
        if self.char_vectorizer is not None:
            if hasattr(self.char_vectorizer, 'get_feature_names_out'):
                names.extend([f"char_{n}" for n in self.char_vectorizer.get_feature_names_out()])
            elif hasattr(self.char_vectorizer, 'get_feature_names'):
                names.extend([f"char_{n}" for n in self.char_vectorizer.get_feature_names()])

        return names

    def _build_vocabulary_index(self):
        """Construye índice inverso del vocabulario."""
        self.vocab_index = {}
        if hasattr(self.vectorizer, 'vocabulary_'):
            self.vocab_index = {idx: word for word, idx in self.vectorizer.vocabulary_.items()}

    def _get_feature_name(self, index: int) -> str:
        """
        Obtiene el nombre de una feature dado su índice.

        Args:
            index: Índice de la feature

        Returns:
            Nombre de la feature o 'feature_N' si no se encuentra
        """
        n_tfidf = len(self.tfidf_feature_names)

        if index < n_tfidf:
            # Es una feature TF-IDF
            if index < len(self.tfidf_feature_names):
                return self.tfidf_feature_names[index]
            return f"tfidf_{index}"
        else:
            # Es una feature URL
            url_index = index - n_tfidf
            if url_index < len(self.url_feature_names):
                return self.url_feature_names[url_index]
            return f"url_feature_{url_index}"

    def _vectorize_text(self, text: str) -> np.ndarray:
        """
        Vectoriza un texto usando word y char vectorizers.

        Args:
            text: Texto preprocesado

        Returns:
            Array de features TF-IDF
        """
        # Word features
        X_word = self.vectorizer.transform([text])

        # Char features
        if self.char_vectorizer is not None:
            X_char = self.char_vectorizer.transform([text])
            X_tfidf = hstack([X_word, X_char])
        else:
            X_tfidf = X_word

        # Convertir a array denso
        if issparse(X_tfidf):
            X_tfidf = X_tfidf.toarray()

        return X_tfidf

    def _prepare_input(self, text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepara el texto para predicción.

        Args:
            text: Texto original (sin preprocesar)

        Returns:
            Tuple de (vector de features completo, dict de features URL)
        """
        # Preprocesar texto
        if self.preprocessor is not None:
            clean_text = self.preprocessor.preprocess(text)
        else:
            clean_text = text

        # Vectorizar TF-IDF (word + char)
        X_tfidf = self._vectorize_text(clean_text)

        # Extraer features URL del texto original
        url_features = {}
        if self.feature_extractor is not None:
            # SMSFeatureExtractor espera string, EmailFeatureExtractor espera dict
            if hasattr(self.feature_extractor, 'extract_all_features'):
                try:
                    # Intentar como SMS (string)
                    url_features = self.feature_extractor.extract_all_features(text)
                except TypeError:
                    # Intentar como Email (dict)
                    url_features = self.feature_extractor.extract_all_features({
                        'body': text, 'subject': '', 'headers': {}
                    })

        # Combinar features si hay URL features
        if self.url_feature_names:
            X_url = np.array([[url_features.get(name, 0) for name in self.url_feature_names]])
            X = np.hstack([X_tfidf, X_url])
        else:
            X = X_tfidf

        return X, url_features

    def predict_proba(self, text: str) -> np.ndarray:
        """
        Obtiene probabilidades de predicción para un texto.

        Args:
            text: Texto a clasificar

        Returns:
            Array de probabilidades [prob_safe, prob_threat]
        """
        X, _ = self._prepare_input(text)
        return self.model.predict_proba(X)[0]

    @abstractmethod
    def explain(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Genera explicación para un texto dado.

        Args:
            text: Texto a explicar
            **kwargs: Argumentos adicionales específicos del explicador

        Returns:
            Dict con la explicación (estructura específica del explicador)
        """
        raise NotImplementedError

    @abstractmethod
    def get_top_features(self, text: str, n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Obtiene las N características más importantes.

        Args:
            text: Texto a analizar
            n: Número de características a retornar

        Returns:
            Lista de tuplas: [(nombre_feature, importancia, 'positive'|'negative'), ...]
            'positive' significa que aumenta la probabilidad de amenaza
            'negative' significa que reduce la probabilidad de amenaza
        """
        raise NotImplementedError

    def _classify_direction(self, value: float) -> str:
        """
        Clasifica la dirección del impacto de una feature.

        Args:
            value: Valor de importancia (positivo o negativo)

        Returns:
            'positive' si aumenta riesgo, 'negative' si lo reduce
        """
        return 'positive' if value > 0 else 'negative'

    def _filter_word_features(
        self,
        features: List[Tuple[str, float, str]],
        exclude_url_features: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Filtra features para quedarse solo con palabras.

        Args:
            features: Lista de features con importancia
            exclude_url_features: Si True, excluye features de URL

        Returns:
            Lista filtrada de features
        """
        if not exclude_url_features:
            return features

        url_feature_set = set(self.url_feature_names)
        return [
            (name, value, direction)
            for name, value, direction in features
            if name not in url_feature_set and not name.startswith('char_')
        ]

    def _extract_word_from_feature_name(self, feature_name: str) -> str:
        """
        Extrae la palabra de un nombre de feature.

        Args:
            feature_name: Nombre de la feature (ej: 'word_urgent', 'char_ng')

        Returns:
            La palabra o el nombre original si no tiene prefijo
        """
        if feature_name.startswith('word_'):
            return feature_name[5:]
        if feature_name.startswith('char_'):
            return feature_name[5:]
        return feature_name
