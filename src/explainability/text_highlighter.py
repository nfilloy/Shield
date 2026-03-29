"""
Utilidades para resaltar texto según importancia de palabras.

Genera HTML con colores que indican qué palabras contribuyen
a clasificar el mensaje como amenaza (rojo) o seguro (verde).
"""

import re
from typing import List, Tuple, Dict, Optional


class TextHighlighter:
    """
    Genera HTML con palabras resaltadas según su importancia en la clasificación.
    """

    # Colores del tema cyber-noir
    COLORS = {
        'danger_high': 'rgba(255, 51, 102, 0.8)',      # Rojo intenso
        'danger_medium': 'rgba(255, 51, 102, 0.5)',    # Rojo medio
        'danger_low': 'rgba(255, 51, 102, 0.25)',      # Rojo suave
        'safe_high': 'rgba(57, 255, 20, 0.8)',         # Verde intenso
        'safe_medium': 'rgba(57, 255, 20, 0.5)',       # Verde medio
        'safe_low': 'rgba(57, 255, 20, 0.25)',         # Verde suave
        'neutral': 'transparent',                       # Sin color
    }

    def __init__(self, intensity_scale: float = 2.0):
        """
        Args:
            intensity_scale: Factor para escalar la intensidad del color
        """
        self.intensity_scale = intensity_scale

    def _normalize_word(self, word: str) -> str:
        """Normaliza una palabra para comparación."""
        return re.sub(r'[^\w]', '', word.lower())

    def _get_color_for_weight(self, weight: float, direction: str) -> str:
        """
        Determina el color basado en el peso e dirección.

        Args:
            weight: Valor absoluto del peso
            direction: 'positive' (aumenta riesgo) o 'negative' (reduce riesgo)

        Returns:
            Color CSS rgba
        """
        # Normalizar intensidad
        intensity = min(abs(weight) * self.intensity_scale, 1.0)

        if direction == 'positive':
            # Escala de rojos (aumenta riesgo)
            return f'rgba(255, 51, 102, {intensity:.2f})'
        else:
            # Escala de verdes (reduce riesgo)
            return f'rgba(57, 255, 20, {intensity:.2f})'

    def highlight(
        self,
        text: str,
        word_weights: List[Tuple[str, float, str]],
        show_weights: bool = False
    ) -> str:
        """
        Genera HTML con palabras resaltadas.

        Args:
            text: Texto original
            word_weights: Lista de (palabra, peso, dirección)
            show_weights: Si mostrar los valores de peso

        Returns:
            HTML con spans coloreados
        """
        # Crear mapa de palabra normalizada -> (peso, dirección)
        weight_map: Dict[str, Tuple[float, str]] = {}
        for word, weight, direction in word_weights:
            normalized = self._normalize_word(word)
            if normalized:
                weight_map[normalized] = (abs(weight), direction)

        # Procesar cada palabra del texto
        html_parts = []
        # Dividir preservando espacios y puntuación
        tokens = re.findall(r'\S+|\s+', text)

        for token in tokens:
            if token.isspace():
                html_parts.append(token)
                continue

            normalized = self._normalize_word(token)

            if normalized in weight_map:
                weight, direction = weight_map[normalized]
                color = self._get_color_for_weight(weight, direction)

                # Crear span con estilo
                if show_weights:
                    title = f'title="Peso: {weight:.3f} ({direction})"'
                else:
                    title = ''

                html_parts.append(
                    f'<span style="background-color: {color}; '
                    f'padding: 2px 4px; border-radius: 3px; '
                    f'transition: all 0.2s;" {title}>{token}</span>'
                )
            else:
                html_parts.append(token)

        return ''.join(html_parts)

    def highlight_with_legend(
        self,
        text: str,
        word_weights: List[Tuple[str, float, str]]
    ) -> str:
        """
        Genera HTML con texto resaltado y leyenda explicativa.

        Args:
            text: Texto original
            word_weights: Lista de (palabra, peso, dirección)

        Returns:
            HTML completo con leyenda
        """
        highlighted_text = self.highlight(text, word_weights, show_weights=True)

        legend_html = """
        <div style="display: flex; gap: 1.5rem; margin-bottom: 1rem;
                    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 16px; height: 16px; background: rgba(255, 51, 102, 0.6);
                            border-radius: 3px;"></span>
                <span style="color: #8892a0;">Aumenta riesgo</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 16px; height: 16px; background: rgba(57, 255, 20, 0.6);
                            border-radius: 3px;"></span>
                <span style="color: #8892a0;">Reduce riesgo</span>
            </div>
        </div>
        """

        text_container = f"""
        <div style="background: #0a0a0f; padding: 1.25rem; border: 1px solid #1a1a24;
                    font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem;
                    line-height: 1.8; color: #e2e8f0; border-radius: 4px;">
            {highlighted_text}
        </div>
        """

        return legend_html + text_container


def highlight_text(
    text: str,
    word_weights: List[Tuple[str, float, str]],
    show_legend: bool = True,
    intensity_scale: float = 2.0
) -> str:
    """
    Función de conveniencia para resaltar texto.

    Args:
        text: Texto original
        word_weights: Lista de (palabra, peso, dirección)
        show_legend: Si incluir leyenda explicativa
        intensity_scale: Factor de escala para intensidad de colores

    Returns:
        HTML con texto resaltado
    """
    highlighter = TextHighlighter(intensity_scale=intensity_scale)

    if show_legend:
        return highlighter.highlight_with_legend(text, word_weights)
    else:
        return highlighter.highlight(text, word_weights)


def create_word_chips(
    word_weights: List[Tuple[str, float, str]],
    max_words: int = 10
) -> str:
    """
    Crea chips/badges HTML para las palabras más importantes.

    Args:
        word_weights: Lista de (palabra, peso, dirección)
        max_words: Máximo número de palabras a mostrar

    Returns:
        HTML con chips de palabras
    """
    # Separar positivas y negativas
    positive = [(w, p) for w, p, d in word_weights if d == 'positive'][:max_words // 2]
    negative = [(w, p) for w, p, d in word_weights if d == 'negative'][:max_words // 2]

    html_parts = ['<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">']

    # Chips de peligro (rojas)
    for word, weight in positive:
        html_parts.append(f'''
            <span style="display: inline-flex; align-items: center; gap: 0.25rem;
                        padding: 4px 10px; background: rgba(255, 51, 102, 0.15);
                        border: 1px solid #ff3366; border-radius: 4px;
                        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
                        color: #ff3366;">
                <span style="font-size: 0.6rem;">&#9650;</span>
                {word}
                <span style="opacity: 0.7;">+{weight:.2f}</span>
            </span>
        ''')

    # Chips de seguridad (verdes)
    for word, weight in negative:
        html_parts.append(f'''
            <span style="display: inline-flex; align-items: center; gap: 0.25rem;
                        padding: 4px 10px; background: rgba(57, 255, 20, 0.15);
                        border: 1px solid #39FF14; border-radius: 4px;
                        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
                        color: #39FF14;">
                <span style="font-size: 0.6rem;">&#9660;</span>
                {word}
                <span style="opacity: 0.7;">{weight:.2f}</span>
            </span>
        ''')

    html_parts.append('</div>')

    return ''.join(html_parts)
