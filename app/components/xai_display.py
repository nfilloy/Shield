"""
Componentes de visualización XAI (Explicabilidad) para Streamlit.

Proporciona visualizaciones interactivas de las explicaciones SHAP y LIME.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Modelos que NO tienen soporte SHAP nativo
MODELS_WITHOUT_SHAP = {'naive_bayes', 'email_naive_bayes'}


def render_word_importance_chart(
    word_weights: List[Tuple[str, float, str]],
    max_words: int = 10,
    title: str = "Importancia de Palabras"
) -> None:
    """
    Renderiza gráfico de barras horizontales con importancia de palabras.

    Args:
        word_weights: Lista de (palabra, peso, dirección)
        max_words: Máximo número de palabras a mostrar
        title: Título del gráfico
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly no instalado. Instalar con: pip install plotly")
        _render_simple_list(word_weights, max_words)
        return

    if not word_weights:
        st.info("No hay palabras significativas para mostrar.")
        return

    # Separar positivos (aumentan riesgo) y negativos (reducen riesgo)
    positive = [(w, p) for w, p, d in word_weights if d == 'positive'][:max_words // 2]
    negative = [(w, abs(p)) for w, p, d in word_weights if d == 'negative'][:max_words // 2]

    # Crear figura
    fig = go.Figure()

    # Barras rojas (aumentan riesgo)
    if positive:
        fig.add_trace(go.Bar(
            y=[w for w, _ in positive],
            x=[p for _, p in positive],
            orientation='h',
            marker_color='#ff3366',
            name='Aumenta riesgo',
            hovertemplate='<b>%{y}</b><br>Peso: +%{x:.3f}<extra></extra>'
        ))

    # Barras verdes (reducen riesgo)
    if negative:
        fig.add_trace(go.Bar(
            y=[w for w, _ in negative],
            x=[-p for _, p in negative],  # Negativo para mostrar a la izquierda
            orientation='h',
            marker_color='#39FF14',
            name='Reduce riesgo',
            hovertemplate='<b>%{y}</b><br>Peso: %{x:.3f}<extra></extra>'
        ))

    # Configurar layout con estilo cyber-noir
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family='IBM Plex Mono', size=14, color='#00fff2'),
            x=0.5
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,15,0.8)',
        font=dict(family='IBM Plex Mono', color='#8892a0', size=11),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            gridcolor='rgba(26,26,36,0.8)',
            zerolinecolor='#1a1a24',
            title='Contribución al resultado'
        ),
        yaxis=dict(
            gridcolor='rgba(26,26,36,0.8)'
        ),
        barmode='relative'
    )

    st.plotly_chart(fig, width='stretch')


def _render_simple_list(
    word_weights: List[Tuple[str, float, str]],
    max_words: int = 10
) -> None:
    """Renderiza lista simple cuando Plotly no está disponible."""
    positive = [(w, p) for w, p, d in word_weights if d == 'positive'][:max_words // 2]
    negative = [(w, p) for w, p, d in word_weights if d == 'negative'][:max_words // 2]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Aumentan riesgo:**")
        for word, weight in positive:
            st.markdown(f"- :red[{word}] (+{weight:.3f})")

    with col2:
        st.markdown("**Reducen riesgo:**")
        for word, weight in negative:
            st.markdown(f"- :green[{word}] ({weight:.3f})")


def render_highlighted_text(
    text: str,
    word_weights: List[Tuple[str, float, str]]
) -> str:
    """
    Genera HTML con texto resaltado según importancia de palabras.

    Args:
        text: Texto original
        word_weights: Lista de (palabra, peso, dirección)

    Returns:
        HTML con palabras coloreadas
    """
    import re

    # Crear mapa de palabra normalizada -> (peso, dirección)
    weight_map = {}
    for word, weight, direction in word_weights:
        normalized = re.sub(r'[^\w]', '', word.lower())
        if normalized:
            weight_map[normalized] = (abs(weight), direction)

    # Procesar texto
    html_parts = []
    tokens = re.findall(r'\S+|\s+', text)

    for token in tokens:
        if token.isspace():
            html_parts.append(token)
            continue

        normalized = re.sub(r'[^\w]', '', token.lower())

        if normalized in weight_map:
            weight, direction = weight_map[normalized]
            intensity = min(weight * 3, 0.8)  # Normalizar intensidad

            if direction == 'positive':
                color = f'rgba(255,51,102,{intensity})'
            else:
                color = f'rgba(57,255,20,{intensity})'

            html_parts.append(f'<span style="background-color:{color};padding:2px 4px;border-radius:3px;">{token}</span>')
        else:
            html_parts.append(token)

    return ''.join(html_parts)


def render_word_chips(word_weights: List[Tuple[str, float, str]], max_words: int = 8) -> None:
    """
    Renderiza chips/badges de palabras importantes.

    Args:
        word_weights: Lista de (palabra, peso, dirección)
        max_words: Máximo de palabras a mostrar
    """
    positive = [(w, p) for w, p, d in word_weights if d == 'positive'][:max_words // 2]
    negative = [(w, p) for w, p, d in word_weights if d == 'negative'][:max_words // 2]

    chips = []

    # Chips de peligro (rojo)
    for word, weight in positive:
        chips.append(f'<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 12px;background:rgba(255,51,102,0.15);border:1px solid #ff3366;border-radius:4px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#ff3366;">&#9650; {word} <span style="opacity:0.6;font-size:0.65rem;">+{weight:.2f}</span></span>')

    # Chips de seguridad (verde)
    for word, weight in negative:
        chips.append(f'<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 12px;background:rgba(57,255,20,0.15);border:1px solid #39FF14;border-radius:4px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#39FF14;">&#9660; {word} <span style="opacity:0.6;font-size:0.65rem;">{weight:.2f}</span></span>')

    html = f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:1rem 0;">{"".join(chips)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_xai_education_section() -> None:
    """
    Renderiza una sección educativa que explica las métricas XAI
    en términos sencillos para usuarios no técnicos.
    """
    with st.expander("💡 ¿Qué significan estas métricas?", expanded=False):
        st.markdown("""
        ### Guía de Interpretación XAI

        Las técnicas de **Explicabilidad de IA (XAI)** nos ayudan a entender
        *por qué* el modelo toma ciertas decisiones. Aquí explicamos cada métrica:

        ---

        #### 📊 LIME (Local Interpretable Model-agnostic Explanations)

        | Métrica | ¿Qué significa? |
        |---------|-----------------|
        | **Muestras** | Número de variaciones del texto que LIME genera para entender qué palabras son importantes. Más muestras = análisis más preciso pero más lento. |
        | **Score R²** | Qué tan bien el modelo explicativo simple replica el comportamiento del modelo complejo. Va de 0 a 1: valores cercanos a 1 indican que la explicación es confiable. |
        | **Palabras positivas (rojas)** | Palabras que **aumentan** la probabilidad de que el mensaje sea fraudulento/phishing. |
        | **Palabras negativas (verdes)** | Palabras que **reducen** la probabilidad de fraude, indicando legitimidad. |

        ---

        #### 🎯 SHAP (SHapley Additive exPlanations)

        | Métrica | ¿Qué significa? |
        |---------|-----------------|
        | **Base Value** | La probabilidad inicial del modelo antes de considerar ninguna palabra específica. Es como el "punto de partida" neutral. |
        | **Contribución** | El cambio total en la predicción causado por todas las palabras encontradas. Valores positivos empujan hacia "fraude", negativos hacia "legítimo". |
        | **Método** | El tipo de explicador usado (TreeExplainer para bosques/XGBoost, LinearExplainer para regresión logística). |
        | **Valores SHAP** | Cada palabra tiene un valor que indica cuánto contribuye a la predicción final. La suma de todos los valores SHAP + Base Value = Predicción final. |

        ---

        #### 🔴🟢 Interpretación de Colores

        - **Rojo / Flechas ▲**: La palabra aumenta el riesgo de que sea phishing/smishing
        - **Verde / Flechas ▼**: La palabra reduce el riesgo (indica mensaje legítimo)
        - **Intensidad del color**: Cuanto más intenso, mayor es el impacto de esa palabra

        ---

        #### 💡 Consejos de Uso

        1. **Fíjate en las palabras rojas**: Son las que el modelo considera sospechosas
        2. **Un R² bajo (< 0.5)** sugiere que la explicación puede no ser muy fiable
        3. **Revisa el contexto**: Una palabra puede ser sospechosa en un contexto y no en otro
        4. **Compara LIME vs SHAP**: Si ambos destacan las mismas palabras, la explicación es más robusta
        """)


def render_xai_section(
    text: str,
    model_name: str,
    detector,
    text_type: str = 'sms'
) -> None:
    """
    Componente principal de explicabilidad XAI.

    Args:
        text: Texto analizado
        model_name: Nombre del modelo usado
        detector: Instancia de PhishingDetector
        text_type: 'sms' o 'email'
    """
    with st.expander("🔍 ¿Por qué esta clasificación?", expanded=True):

        # Botón de ayuda integrado
        render_xai_education_section()

        # Verificar si el modelo soporta SHAP
        model_supports_shap = model_name not in MODELS_WITHOUT_SHAP

        # Obtener explicadores
        lime_explainer, shap_explainer = _get_explainers(detector, model_name, text_type)

        if model_supports_shap:
            # Mostrar ambos tabs
            tab_lime, tab_shap = st.tabs(["LIME", "SHAP"])

            with tab_lime:
                if lime_explainer is not None:
                    _render_lime_explanation(text, lime_explainer)
                else:
                    st.warning("LIME no disponible. Instalar con: `pip install lime`")

            with tab_shap:
                if shap_explainer is not None:
                    _render_shap_explanation(text, shap_explainer)
                else:
                    st.warning("SHAP no disponible. Instalar con: `pip install shap`")
        else:
            # Solo LIME para modelos sin soporte SHAP
            st.info("ℹ️ SHAP no disponible para Naive Bayes (usa probabilidades bayesianas, incompatible con SHAP)")

            if lime_explainer is not None:
                _render_lime_explanation(text, lime_explainer)
            else:
                st.warning("LIME no disponible. Instalar con: `pip install lime`")


@st.cache_resource
def _get_explainers(_detector, model_name: str, text_type: str):
    """
    Obtiene o crea los explicadores para un modelo.

    Usa caché para no recrearlos en cada llamada.
    """
    try:
        from src.explainability.lime_explainer import LIMEExplainer
        from src.explainability.shap_explainer import SHAPExplainer
        from src.data.preprocessor import get_ml_preprocessor
    except ImportError as e:
        logger.error(f"Error importando explicadores: {e}")
        return None, None

    # Obtener modelo y vectorizador
    model = _detector.models.get(model_name)
    if model is None:
        return None, None

    # Seleccionar vectorizador y extractor según tipo
    if text_type == 'email' or model_name.startswith('email_'):
        vectorizer = _detector.vectorizers.get('email_tfidf')
        char_vectorizer = _detector.vectorizers.get('email_char')
        feature_extractor = _detector.email_feature_extractor
        url_config = _detector.url_configs.get('email', {})
    else:
        vectorizer = _detector.vectorizers.get('tfidf')
        char_vectorizer = _detector.vectorizers.get('char')
        feature_extractor = _detector.sms_feature_extractor
        url_config = _detector.url_configs.get('sms', {})

    if vectorizer is None:
        return None, None

    url_feature_names = url_config.get('feature_names', [])
    preprocessor = get_ml_preprocessor()

    # Crear explicadores
    lime_exp = None
    shap_exp = None

    try:
        lime_exp = LIMEExplainer(
            model=model,
            vectorizer=vectorizer,
            char_vectorizer=char_vectorizer,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )
    except Exception as e:
        logger.warning(f"No se pudo crear LIMEExplainer: {e}")

    try:
        shap_exp = SHAPExplainer(
            model=model,
            vectorizer=vectorizer,
            char_vectorizer=char_vectorizer,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='auto'
        )
    except Exception as e:
        logger.warning(f"No se pudo crear SHAPExplainer: {e}")

    return lime_exp, shap_exp


def _render_lime_explanation(text: str, explainer) -> None:
    """Renderiza la explicación LIME."""
    with st.spinner("Analizando con LIME..."):
        try:
            result = explainer.explain(text, num_features=10, num_samples=300)

            word_weights = result.get('word_weights', [])

            if word_weights:
                # Mostrar chips de palabras
                st.markdown("##### Palabras clave identificadas")
                render_word_chips(word_weights)

                # Mostrar gráfico
                render_word_importance_chart(word_weights, title="Contribución de palabras (LIME)")

                # Mostrar texto resaltado
                st.markdown("##### Texto analizado")
                highlighted = render_highlighted_text(text, word_weights)
                st.markdown(f'<div style="background:#0a0a0f;padding:1rem;border:1px solid #1a1a24;font-family:monospace;font-size:0.9rem;line-height:1.8;color:#e2e8f0;border-radius:4px;">{highlighted}</div>', unsafe_allow_html=True)

                # Métricas adicionales
                st.markdown("---")
                col1, col2 = st.columns(2)
                col1.metric("Muestras", result.get('num_samples', 'N/A'))
                col2.metric("Score R²", f"{result.get('score', 0):.3f}")

            else:
                st.info("No se encontraron palabras significativas en el análisis.")

        except Exception as e:
            logger.error(f"Error en explicación LIME: {e}")
            st.error(f"Error generando explicación: {str(e)}")


def _is_user_readable_feature(name: str) -> bool:
    """Determina si una feature es legible para el usuario (no técnica)."""
    # Features de URL descriptivas que SÍ queremos mostrar (con nombres amigables)
    url_descriptive_features = (
        'url_count', 'shortened_url_count', 'suspicious_tld_count',
        'ip_url_count', 'unique_domain_count', 'has_url', 'has_urls',
        'has_shortened_url', 'has_suspicious_tld', 'has_ip_url'
    )
    if name in url_descriptive_features:
        return True

    # Prefijos técnicos a excluir
    technical_prefixes = (
        'char_',      # n-gramas de caracteres
        'url_',       # features de URL restantes
        'total_',     # conteos totales
    )
    # Caracteres técnicos en el nombre
    technical_chars = ('<', '>', '_num', '_ratio')

    if name.startswith(technical_prefixes):
        return False
    if any(tc in name for tc in technical_chars):
        return False
    # Solo palabras alfabéticas son legibles
    return name.isalpha() or ' ' in name


def _render_shap_metrics(base_value: float, contribution: float, method: str) -> None:
    """Renderiza métricas SHAP con indicadores visuales de interpretación."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Base Value con indicador
        direction = "legítimo" if base_value < 0 else "fraude"
        color = "#39FF14" if base_value < 0 else "#ff3366"
        st.markdown(f"""
        <div style="text-align:center;">
            <p style="color:#8892a0;font-size:0.85rem;margin-bottom:4px;">Base Value</p>
            <p style="font-size:1.5rem;font-weight:bold;color:{color};margin:0;">{base_value:.3f}</p>
            <p style="color:#8892a0;font-size:0.75rem;margin-top:4px;">Tendencia inicial → {direction}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Contribución con indicador
        if contribution < 0:
            icon, label, color = "✓", "Reduce riesgo", "#39FF14"
        else:
            icon, label, color = "⚠", "Aumenta riesgo", "#ff3366"
        st.markdown(f"""
        <div style="text-align:center;">
            <p style="color:#8892a0;font-size:0.85rem;margin-bottom:4px;">Contribución</p>
            <p style="font-size:1.5rem;font-weight:bold;color:{color};margin:0;">{icon} {contribution:.3f}</p>
            <p style="color:#8892a0;font-size:0.75rem;margin-top:4px;">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="text-align:center;">
            <p style="color:#8892a0;font-size:0.85rem;margin-bottom:4px;">Método</p>
            <p style="font-size:1.5rem;font-weight:bold;color:#00fff2;margin:0;">{method}</p>
            <p style="color:#8892a0;font-size:0.75rem;margin-top:4px;">Explicador SHAP</p>
        </div>
        """, unsafe_allow_html=True)


def _render_shap_explanation(text: str, explainer) -> None:
    """Renderiza la explicación SHAP."""
    with st.spinner("Analizando con SHAP..."):
        try:
            result = explainer.explain(text)

            # Convertir feature_importance a formato word_weights
            feature_importance = result.get('feature_importance', {})

            if feature_importance:
                # Filtrar solo features legibles para el usuario
                readable_features = {
                    name: value for name, value in feature_importance.items()
                    if _is_user_readable_feature(name)
                }

                # Si no hay features legibles, mostrar mensaje con métricas
                if not readable_features:
                    st.info("Las features más importantes son técnicas (n-gramas de caracteres). El modelo detecta patrones sutiles en la estructura del texto.")
                    _render_shap_metrics(
                        result.get('base_value', 0),
                        result.get('prediction_contribution', 0),
                        result.get('model_type', 'N/A')
                    )
                    return

                word_weights = [
                    (name, value, 'positive' if value > 0 else 'negative')
                    for name, value in sorted(
                        readable_features.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:10]
                ]

                # Mostrar chips
                st.markdown("##### Palabras clave identificadas")
                render_word_chips(word_weights)

                # Mostrar gráfico
                render_word_importance_chart(word_weights, title="Contribución de palabras (SHAP)")

                # Mostrar texto resaltado
                word_only = [(w, v, d) for w, v, d in word_weights if w.isalpha()]
                if word_only:
                    st.markdown("##### Texto analizado")
                    highlighted = render_highlighted_text(text, word_only)
                    st.markdown(f'<div style="background:#0a0a0f;padding:1rem;border:1px solid #1a1a24;font-family:monospace;font-size:0.9rem;line-height:1.8;color:#e2e8f0;border-radius:4px;">{highlighted}</div>', unsafe_allow_html=True)

                # Métricas SHAP con indicadores visuales
                _render_shap_metrics(
                    result.get('base_value', 0),
                    result.get('prediction_contribution', 0),
                    result.get('model_type', 'N/A')
                )

            else:
                st.info("No se encontraron features significativas en el análisis SHAP.")

        except Exception as e:
            logger.error(f"Error en explicación SHAP: {e}")
            st.error(f"Error generando explicación: {str(e)}")
