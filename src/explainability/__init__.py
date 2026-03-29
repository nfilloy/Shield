"""
Módulo de Explicabilidad (XAI) para modelos de detección de phishing/smishing.

Proporciona explicaciones interpretables de las predicciones usando SHAP y LIME.
"""

from .base import BaseExplainer
from .text_highlighter import TextHighlighter, highlight_text, create_word_chips
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer

__all__ = [
    'BaseExplainer',
    'TextHighlighter',
    'highlight_text',
    'create_word_chips',
    'SHAPExplainer',
    'LIMEExplainer',
]
