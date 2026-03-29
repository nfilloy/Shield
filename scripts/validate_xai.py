"""
Script para validar manualmente las explicaciones XAI.

Ejecutar: python scripts/validate_xai.py

Muestra explicaciones LIME y SHAP para varios textos de ejemplo,
permitiendo verificar visualmente que las explicaciones tienen sentido.
"""

import sys
from pathlib import Path

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from typing import List, Tuple


def load_components():
    """Cargar modelo, vectorizador y componentes necesarios."""
    models_dir = Path(__file__).parent.parent / "models"

    print("Cargando componentes...")

    # Cargar modelo
    model_path = models_dir / "xgboost.pkl"
    if not model_path.exists():
        print(f"ERROR: No se encontro {model_path}")
        print("Ejecuta primero el entrenamiento: python scripts/train_sms_models.py")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  - Modelo: xgboost.pkl")

    # Cargar vectorizadores (word y char)
    vec_path = models_dir / "tfidf_vectorizer.pkl"
    with open(vec_path, 'rb') as f:
        vec_data = pickle.load(f)
        if isinstance(vec_data, dict):
            vectorizer = vec_data['word_vectorizer']
            char_vectorizer = vec_data.get('char_vectorizer')
        else:
            vectorizer = vec_data
            char_vectorizer = None
    print(f"  - Vectorizador word: OK")
    print(f"  - Vectorizador char: {'OK' if char_vectorizer else 'No disponible'}")

    # Cargar config de URL features
    url_feature_names = []
    config_path = models_dir / "url_features_config.pkl"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            url_feature_names = config.get('feature_names', [])
        print(f"  - URL features: {len(url_feature_names)} features")

    # Cargar preprocesador y extractor
    from src.data.preprocessor import get_ml_preprocessor
    from src.features.sms_features import SMSFeatureExtractor

    preprocessor = get_ml_preprocessor()
    feature_extractor = SMSFeatureExtractor()

    print("  - Preprocesador y extractor: OK")
    print()

    return model, vectorizer, char_vectorizer, preprocessor, feature_extractor, url_feature_names


def create_explainers(model, vectorizer, char_vectorizer, preprocessor, feature_extractor, url_feature_names):
    """Crear explicadores LIME y SHAP."""
    from src.explainability.lime_explainer import LIMEExplainer
    from src.explainability.shap_explainer import SHAPExplainer

    lime_exp = LIMEExplainer(
        model=model,
        vectorizer=vectorizer,
        char_vectorizer=char_vectorizer,
        feature_extractor=feature_extractor,
        preprocessor=preprocessor,
        url_feature_names=url_feature_names
    )

    shap_exp = SHAPExplainer(
        model=model,
        vectorizer=vectorizer,
        char_vectorizer=char_vectorizer,
        feature_extractor=feature_extractor,
        preprocessor=preprocessor,
        url_feature_names=url_feature_names,
        model_type='tree'
    )

    return lime_exp, shap_exp


def print_separator(char='=', length=70):
    """Imprimir separador visual."""
    print(char * length)


def print_header(title: str):
    """Imprimir encabezado."""
    print_separator()
    print(f"  {title}")
    print_separator()


def print_word_weights(word_weights: List[Tuple[str, float, str]], max_words: int = 8):
    """Imprimir pesos de palabras de forma visual."""
    if not word_weights:
        print("  (No hay palabras significativas)")
        return

    for word, weight, direction in word_weights[:max_words]:
        bar_length = int(abs(weight) * 30)
        bar = "#" * min(bar_length, 20)

        if direction == 'positive':
            symbol = "[!]"
            sign = "+"
        else:
            symbol = "[OK]"
            sign = ""

        print(f"  {symbol} {word:20} {sign}{weight:+.4f}  {bar}")


def analyze_text(text: str, lime_exp, shap_exp, title: str = ""):
    """Analizar un texto con LIME y SHAP."""
    print_header(title if title else "ANALISIS")
    print()
    print(f"Texto: {text[:80]}{'...' if len(text) > 80 else ''}")
    print()

    # Analisis LIME
    print("--- LIME ---")
    try:
        lime_result = lime_exp.explain(text, num_features=8, num_samples=200)
        verdict = "PHISHING/SMISHING" if lime_result['prediction'] == 1 else "LEGITIMO"
        prob = lime_result['probabilities'][1] * 100

        print(f"  Veredicto: {verdict}")
        print(f"  Probabilidad amenaza: {prob:.1f}%")
        print()
        print("  Palabras influyentes:")
        print_word_weights(lime_result['word_weights'])
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # Analisis SHAP
    print("--- SHAP ---")
    try:
        shap_result = shap_exp.explain(text)
        verdict = "PHISHING/SMISHING" if shap_result['prediction'] == 1 else "LEGITIMO"
        prob = shap_result['probability'] * 100

        print(f"  Veredicto: {verdict}")
        print(f"  Probabilidad amenaza: {prob:.1f}%")
        print(f"  Base value: {shap_result['base_value']:.4f}")
        print()

        # Convertir feature_importance a formato de word_weights
        importance = shap_result['feature_importance']
        word_weights = [
            (name, value, 'positive' if value > 0 else 'negative')
            for name, value in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        ]
        print("  Features influyentes:")
        print_word_weights(word_weights)
    except Exception as e:
        print(f"  Error: {e}")

    print()


def main():
    """Funcion principal."""
    print()
    print_header("VALIDACION DE EXPLICABILIDAD XAI")
    print()

    # Cargar componentes
    model, vectorizer, char_vectorizer, preprocessor, feature_extractor, url_feature_names = load_components()

    # Crear explicadores
    print("Inicializando explicadores...")
    lime_exp, shap_exp = create_explainers(
        model, vectorizer, char_vectorizer, preprocessor, feature_extractor, url_feature_names
    )
    print("  - LIME: OK")
    print("  - SHAP: OK")
    print()

    # Casos de prueba
    test_cases = [
        (
            "Phishing obvio",
            "URGENT: Your bank account has been suspended! Click bit.ly/verify NOW to restore access immediately!"
        ),
        (
            "Smishing con premio",
            "Congratulations! You've won 1000 euros. Claim your prize now at winner.xyz/claim before it expires!"
        ),
        (
            "Mensaje legitimo",
            "Hi Mom! I'll be home for dinner around 7pm. Do you need me to bring anything? Love you!"
        ),
        (
            "Codigo de verificacion",
            "Your verification code is 847293. This code expires in 10 minutes. Do not share it with anyone."
        ),
        (
            "Caso ambiguo",
            "Your package is waiting for delivery. Please confirm your address at ups-tracking.com/confirm"
        ),
    ]

    # Analizar cada caso
    for title, text in test_cases:
        analyze_text(text, lime_exp, shap_exp, title)
        print()

    print_separator()
    print("  VALIDACION COMPLETADA")
    print_separator()
    print()
    print("Revisa los resultados:")
    print("  - Los mensajes de phishing deberian tener alta probabilidad de amenaza")
    print("  - Las palabras como 'urgent', 'click', 'verify' deberian aumentar el riesgo")
    print("  - Los mensajes legitimos deberian tener baja probabilidad")
    print()


def interactive_mode():
    """Modo interactivo para probar textos personalizados."""
    print()
    print_header("MODO INTERACTIVO")
    print()
    print("Escribe un mensaje para analizar (o 'salir' para terminar)")
    print()

    # Cargar componentes
    model, vectorizer, char_vectorizer, preprocessor, feature_extractor, url_feature_names = load_components()
    lime_exp, shap_exp = create_explainers(
        model, vectorizer, char_vectorizer, preprocessor, feature_extractor, url_feature_names
    )

    while True:
        print_separator('-', 50)
        text = input("\nMensaje: ").strip()

        if text.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\nHasta luego!")
            break

        if not text:
            print("Por favor, introduce un mensaje.")
            continue

        print()
        analyze_text(text, lime_exp, shap_exp, "ANALISIS PERSONALIZADO")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validar explicaciones XAI")
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Modo interactivo para probar textos personalizados'
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        main()
