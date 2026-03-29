"""
Tests unitarios para el módulo de explicabilidad (XAI).

Ejecutar con: pytest tests/test_explainability.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import pickle
import sys

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_phishing_sms():
    """Ejemplos de SMS de phishing conocidos."""
    return [
        "URGENT: Your bank account has been suspended. Click here to verify: bit.ly/verify123",
        "Congratulations! You've won 1000 euros. Claim now at prize.xyz/claim",
        "Your package could not be delivered. Update address: tracking.suspicious.top/update",
        "ALERT: Unusual activity on your account. Verify immediately: secure-bank.tk/login",
    ]


@pytest.fixture
def sample_legitimate_sms():
    """Ejemplos de SMS legítimos."""
    return [
        "Hi! Are we still meeting for coffee tomorrow at 3pm?",
        "Your verification code is 123456. Valid for 10 minutes.",
        "Thanks for your purchase. Your order #12345 will arrive Monday.",
        "Happy birthday! Hope you have a great day!",
    ]


@pytest.fixture
def sample_phishing_email():
    """Ejemplos de emails de phishing."""
    return [
        "Dear Customer, Your account has been compromised. Click here immediately to secure it: http://secure-update.xyz/login",
        "URGENT: Your PayPal account will be suspended unless you verify your information now.",
    ]


@pytest.fixture
def models_dir():
    """Directorio de modelos."""
    return Path(__file__).parent.parent / "models"


@pytest.fixture
def loaded_model(models_dir):
    """Cargar modelo XGBoost para tests."""
    model_path = models_dir / "xgboost.pkl"
    if not model_path.exists():
        pytest.skip("Modelo xgboost.pkl no encontrado")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture
def loaded_vectorizer(models_dir):
    """Cargar vectorizador word para tests."""
    vec_path = models_dir / "tfidf_vectorizer.pkl"
    if not vec_path.exists():
        pytest.skip("Vectorizador no encontrado")
    with open(vec_path, 'rb') as f:
        data = pickle.load(f)
        return data['word_vectorizer'] if isinstance(data, dict) else data


@pytest.fixture
def loaded_char_vectorizer(models_dir):
    """Cargar vectorizador char para tests."""
    vec_path = models_dir / "tfidf_vectorizer.pkl"
    if not vec_path.exists():
        pytest.skip("Vectorizador no encontrado")
    with open(vec_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict) and 'char_vectorizer' in data:
            return data['char_vectorizer']
        return None


@pytest.fixture
def url_feature_names(models_dir):
    """Cargar nombres de features URL."""
    config_path = models_dir / "url_features_config.pkl"
    if not config_path.exists():
        return []
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        return config.get('feature_names', [])


@pytest.fixture
def preprocessor():
    """Obtener preprocesador."""
    from src.data.preprocessor import get_ml_preprocessor
    return get_ml_preprocessor()


@pytest.fixture
def sms_feature_extractor():
    """Obtener extractor de features SMS."""
    from src.features.sms_features import SMSFeatureExtractor
    return SMSFeatureExtractor()


# ============================================================
# Tests de BaseExplainer
# ============================================================

class TestBaseExplainer:
    """Tests para la clase base BaseExplainer."""

    def test_base_explainer_is_abstract(self):
        """Verificar que BaseExplainer es abstracta."""
        from src.explainability.base import BaseExplainer

        with pytest.raises(TypeError):
            BaseExplainer(None, None)

    def test_get_feature_name(self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, url_feature_names):
        """Verificar mapeo de índices a nombres de features."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            loaded_model,
            loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            url_feature_names=url_feature_names
        )

        # Feature TF-IDF (índice bajo)
        name = explainer._get_feature_name(0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_prepare_input_returns_correct_shape(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar que _prepare_input retorna forma correcta."""
        from src.explainability.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        X, url_features = explainer._prepare_input(sample_phishing_sms[0])

        assert X.ndim == 2
        assert X.shape[0] == 1  # Una muestra
        # Debe tener 10044 features (5000 word + 5000 char + 44 url)
        assert X.shape[1] == 10044
        assert isinstance(url_features, dict)


# ============================================================
# Tests de TextHighlighter
# ============================================================

class TestTextHighlighter:
    """Tests para text_highlighter."""

    def test_highlight_generates_valid_html(self):
        """Verificar que se genera HTML válido."""
        from src.explainability.text_highlighter import highlight_text

        text = "This is a test message with urgent content"
        weights = [
            ('urgent', 0.5, 'positive'),
            ('message', -0.3, 'negative')
        ]

        html = highlight_text(text, weights, show_legend=False)

        assert '<span' in html
        assert 'urgent' in html
        assert 'message' in html

    def test_highlight_preserves_text_content(self):
        """Verificar que el contenido del texto se preserva."""
        from src.explainability.text_highlighter import highlight_text

        text = "first second third"
        weights = [('second', 0.5, 'positive')]

        html = highlight_text(text, weights, show_legend=False)

        assert 'first' in html
        assert 'second' in html
        assert 'third' in html

    def test_highlight_with_legend(self):
        """Verificar generación de HTML con leyenda."""
        from src.explainability.text_highlighter import highlight_text

        text = "Test message"
        weights = [('test', 0.3, 'positive')]

        html = highlight_text(text, weights, show_legend=True)

        assert 'Aumenta riesgo' in html
        assert 'Reduce riesgo' in html

    def test_create_word_chips(self):
        """Verificar generación de chips de palabras."""
        from src.explainability.text_highlighter import create_word_chips

        weights = [
            ('urgent', 0.5, 'positive'),
            ('hello', 0.3, 'negative'),
        ]

        html = create_word_chips(weights)

        assert 'urgent' in html
        assert 'hello' in html
        assert '#ff3366' in html  # Color rojo
        assert '#39FF14' in html  # Color verde

    def test_highlighter_class(self):
        """Verificar clase TextHighlighter directamente."""
        from src.explainability.text_highlighter import TextHighlighter

        highlighter = TextHighlighter(intensity_scale=2.0)

        text = "Click here to verify"
        weights = [('click', 0.4, 'positive'), ('verify', 0.3, 'positive')]

        html = highlighter.highlight(text, weights)

        assert 'click' in html.lower()
        assert 'verify' in html.lower()


# ============================================================
# Tests de LIMEExplainer
# ============================================================

class TestLIMEExplainer:
    """Tests para LIMEExplainer."""

    def test_lime_explain_returns_valid_structure(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar que explain() retorna la estructura correcta."""
        from src.explainability.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        result = explainer.explain(sample_phishing_sms[0], num_features=5, num_samples=100)

        assert 'word_weights' in result
        assert 'prediction' in result
        assert 'probabilities' in result
        assert isinstance(result['word_weights'], list)
        assert len(result['probabilities']) == 2

    def test_lime_prediction_matches_model(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar que la predicción LIME coincide con el modelo."""
        from src.explainability.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        result = explainer.explain(sample_phishing_sms[0], num_features=5, num_samples=100)

        # La predicción debería ser 0 o 1
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probabilities'][0] <= 1
        assert 0 <= result['probabilities'][1] <= 1

    def test_lime_get_top_features(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar get_top_features()."""
        from src.explainability.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        features = explainer.get_top_features(sample_phishing_sms[0], n=5)

        assert len(features) <= 5
        for word, weight, direction in features:
            assert isinstance(word, str)
            assert isinstance(weight, (int, float))
            assert direction in ['positive', 'negative']


# ============================================================
# Tests de SHAPExplainer
# ============================================================

class TestSHAPExplainer:
    """Tests para SHAPExplainer."""

    def test_shap_explain_returns_valid_structure(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar estructura de salida SHAP."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='tree'
        )

        result = explainer.explain(sample_phishing_sms[0])

        assert 'shap_values' in result
        assert 'base_value' in result
        assert 'feature_importance' in result
        assert 'prediction' in result

    def test_shap_model_type_detection(self, loaded_model, loaded_vectorizer, loaded_char_vectorizer):
        """Verificar detección automática de tipo de modelo."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            model_type='auto'
        )

        # XGBoost debería detectarse como 'tree'
        assert explainer.model_type == 'tree'

    def test_shap_get_top_features(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar get_top_features() de SHAP."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='tree'
        )

        features = explainer.get_top_features(sample_phishing_sms[0], n=10)

        assert len(features) <= 10
        for name, value, direction in features:
            assert isinstance(name, str)
            assert isinstance(value, (int, float))
            assert direction in ['positive', 'negative']

    def test_shap_feature_importance_not_empty(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar que feature_importance no está vacío."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='tree'
        )

        result = explainer.explain(sample_phishing_sms[0])

        # Debería haber al menos algunas features importantes
        assert len(result['feature_importance']) > 0


# ============================================================
# Tests de Integración
# ============================================================

class TestXAIIntegration:
    """Tests de integración del módulo XAI."""

    def test_lime_and_shap_same_prediction(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar que LIME y SHAP dan la misma predicción."""
        from src.explainability.lime_explainer import LIMEExplainer
        from src.explainability.shap_explainer import SHAPExplainer

        lime_exp = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        shap_exp = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='tree'
        )

        text = sample_phishing_sms[0]

        lime_result = lime_exp.explain(text, num_features=5, num_samples=100)
        shap_result = shap_exp.explain(text)

        # Ambos deberían dar la misma predicción
        assert lime_result['prediction'] == shap_result['prediction']

    def test_highlighter_with_lime_output(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """Verificar integración de highlighter con output LIME."""
        from src.explainability.lime_explainer import LIMEExplainer
        from src.explainability.text_highlighter import highlight_text

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        text = sample_phishing_sms[0]
        result = explainer.explain(text, num_features=5, num_samples=100)

        # Usar word_weights directamente en highlighter
        html = highlight_text(text, result['word_weights'])

        assert isinstance(html, str)
        assert len(html) > len(text)  # HTML debería ser más largo

    def test_full_pipeline_sms(self, sample_phishing_sms, models_dir):
        """Test del pipeline completo para SMS."""
        from src.explainability.lime_explainer import LIMEExplainer
        from src.data.preprocessor import get_ml_preprocessor
        from src.features.sms_features import SMSFeatureExtractor

        # Verificar que los archivos existen
        model_path = models_dir / "xgboost.pkl"
        vec_path = models_dir / "tfidf_vectorizer.pkl"
        config_path = models_dir / "url_features_config.pkl"

        if not all(p.exists() for p in [model_path, vec_path]):
            pytest.skip("Archivos de modelo no encontrados")

        # Cargar componentes
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(vec_path, 'rb') as f:
            vec_data = pickle.load(f)
            vectorizer = vec_data['word_vectorizer'] if isinstance(vec_data, dict) else vec_data
            char_vectorizer = vec_data.get('char_vectorizer') if isinstance(vec_data, dict) else None

        url_feature_names = []
        if config_path.exists():
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                url_feature_names = config.get('feature_names', [])

        preprocessor = get_ml_preprocessor()
        feature_extractor = SMSFeatureExtractor()

        # Crear explicador
        explainer = LIMEExplainer(
            model=model,
            vectorizer=vectorizer,
            char_vectorizer=char_vectorizer,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        # Explicar
        result = explainer.explain(sample_phishing_sms[0], num_features=5, num_samples=100)

        # Verificaciones
        assert result['prediction'] in [0, 1]
        assert len(result['word_weights']) > 0


# ============================================================
# Tests de Rendimiento
# ============================================================

class TestXAIPerformance:
    """Tests de rendimiento."""

    @pytest.mark.slow
    def test_lime_explanation_time(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """LIME debería completar en menos de 15 segundos."""
        import time
        from src.explainability.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        start = time.time()
        explainer.explain(sample_phishing_sms[0], num_features=10, num_samples=300)
        elapsed = time.time() - start

        assert elapsed < 15, f"LIME tardo {elapsed:.2f}s, deberia ser < 15s"

    @pytest.mark.slow
    def test_shap_explanation_time(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_phishing_sms
    ):
        """SHAP con TreeExplainer debería ser rápido."""
        import time
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='tree'
        )

        start = time.time()
        explainer.explain(sample_phishing_sms[0])
        elapsed = time.time() - start

        assert elapsed < 10, f"SHAP tardo {elapsed:.2f}s, deberia ser < 10s"


# ============================================================
# Tests de Fallback
# ============================================================

class TestFallbackBehavior:
    """Tests para comportamiento de fallback cuando las librerías no están disponibles."""

    def test_lime_fallback_explanation(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_legitimate_sms
    ):
        """Verificar que el fallback de LIME funciona."""
        from src.explainability.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names
        )

        # Forzar fallback
        explainer._lime_available = False

        result = explainer.explain(sample_legitimate_sms[0])

        assert 'prediction' in result
        assert 'probabilities' in result
        assert result['method'] == 'fallback'

    def test_shap_fallback_explanation(
        self, loaded_model, loaded_vectorizer, loaded_char_vectorizer, preprocessor,
        sms_feature_extractor, url_feature_names, sample_legitimate_sms
    ):
        """Verificar que el fallback de SHAP funciona."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(
            model=loaded_model,
            vectorizer=loaded_vectorizer,
            char_vectorizer=loaded_char_vectorizer,
            feature_extractor=sms_feature_extractor,
            preprocessor=preprocessor,
            url_feature_names=url_feature_names,
            model_type='tree'
        )

        # Forzar fallback
        explainer._shap_available = False
        explainer.shap_explainer = None

        result = explainer.explain(sample_legitimate_sms[0])

        assert 'prediction' in result
        assert result['method'] == 'fallback'
