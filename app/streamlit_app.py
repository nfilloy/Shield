"""
Streamlit Web Application for Phishing & Smishing Detection

Cyber-noir terminal aesthetic with distinctive visual effects.
Optimized for performance with caching.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from typing import Dict, List, Any
from scipy.sparse import hstack as sparse_hstack

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import get_ml_preprocessor
from src.features.sms_features import SMSFeatureExtractor
from src.features.email_features import EmailFeatureExtractor

# XAI Components
try:
    from app.components.xai_display import render_xai_section
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False

# Auth Components
try:
    from app.components.auth import (
        init_auth_state,
        is_authenticated,
        is_guest,
        get_current_user,
        render_logout_button,
        render_user_menu,
        get_auth_css,
        render_auth_page
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Database - Analysis saving
try:
    from src.database import save_analysis, init_db
    # Initialize database on startup
    init_db()
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SHIELD // Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Model descriptions (actualizado tras re-entrenamiento)
MODEL_INFO = {
    'naive_bayes': {
        'name': 'BAYES.neural',
        'description': 'Probabilistic classifier // Bayesian inference',
        'accuracy': '98.65%',
        'f1': '95.05%'
    },
    'logistic_regression': {
        'name': 'LOGIT.core',
        'description': 'Linear model // Maximum likelihood',
        'accuracy': '98.48%',
        'f1': '94.31%'
    },
    'random_forest': {
        'name': 'FOREST.ensemble',
        'description': 'Decision trees // Bagging method',
        'accuracy': '98.57%',
        'f1': '94.37%'
    },
    'linear_svm': {
        'name': 'VECTOR.linear',
        'description': 'Linear SVM // High-dim optimized',
        'accuracy': '98.92%',
        'f1': '95.83%'
    },
    'xgboost': {
        'name': 'XGBOOST.prime',
        'description': 'Gradient boosting // Regularized',
        'accuracy': '98.83%',
        'f1': '95.50%'
    },
    'email_naive_bayes': {
        'name': 'BAYES.mail',
        'description': 'Email-tuned Bayesian classifier',
        'accuracy': '60.83%',
        'f1': '63.61%'
    },
    'email_logistic_regression': {
        'name': 'LOGIT.mail',
        'description': 'Email-optimized regression',
        'accuracy': '93.43%',
        'f1': '91.44%'
    },
    'email_random_forest': {
        'name': 'FOREST.mail',
        'description': 'Email pattern ensemble',
        'accuracy': '97.49%',
        'f1': '96.62%'
    },
    'email_xgboost': {
        'name': 'XGBOOST.mail',
        'description': 'Email gradient booster',
        'accuracy': '97.43%',
        'f1': '96.60%'
    }
}


@st.cache_data
def get_css():
    """Return cached CSS styles - Cyber-noir terminal aesthetic."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Anybody:wght@400;600;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap&font-display=swap');

        :root {
            --void: #050508;
            --surface: #0a0a0f;
            --surface-2: #12121a;
            --surface-3: #1a1a24;
            --neon-green: #39FF14;
            --neon-cyan: #00fff2;
            --neon-red: #ff3366;
            --neon-amber: #ffaa00;
            --text-dim: #4a5568;
            --text-mid: #8892a0;
            --text-bright: #e2e8f0;
        }

        /* Base App */
        .stApp {
            background: var(--void);
        }

        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 2px,
                    rgba(0, 255, 242, 0.01) 2px,
                    rgba(0, 255, 242, 0.01) 4px
                );
            pointer-events: none;
            z-index: 1000;
        }

        #MainMenu, footer, header {visibility: hidden;}

        /* ══════════════════════════════════════════
           HIDE STREAMLIT DEV TOOLBAR
           ══════════════════════════════════════════ */
        [data-testid="stToolbar"] {
            visibility: hidden !important;
            height: 0 !important;
            position: fixed !important;
            top: -100px !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
        }

        [data-testid="stStatusWidget"] {
            visibility: hidden !important;
        }

        .stDeployButton {
            display: none !important;
        }

        .main .block-container {
            padding: 1rem 2rem;
            max-width: 1400px;
        }

        /* ══════════════════════════════════════════
           HERO SECTION - Terminal Boot Sequence
           ══════════════════════════════════════════ */
        .hero-terminal {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem;
            position: relative;
        }

        .hero-terminal::before {
            content: '[ SYSTEM ONLINE ]';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: var(--neon-green);
            letter-spacing: 4px;
            opacity: 0.7;
        }

        .hero-title {
            font-family: 'Anybody', sans-serif;
            font-size: 4.5rem;
            font-weight: 800;
            letter-spacing: -3px;
            margin: 0;
            line-height: 1;
            background: linear-gradient(
                180deg,
                var(--neon-cyan) 0%,
                var(--neon-green) 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 80px rgba(0, 255, 242, 0.5);
            animation: flicker 4s infinite;
        }

        @keyframes flicker {
            0%, 100% { opacity: 1; }
            92% { opacity: 1; }
            93% { opacity: 0.8; }
            94% { opacity: 1; }
            96% { opacity: 0.9; }
            97% { opacity: 1; }
        }

        .hero-subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-dim);
            letter-spacing: 6px;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        .hero-subtitle span {
            color: var(--neon-green);
        }

        /* ══════════════════════════════════════════
           TYPE SELECTOR - Mode Switch
           ══════════════════════════════════════════ */
        .mode-switch {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .stButton > button {
            background: transparent !important;
            border: 1px solid var(--surface-3) !important;
            border-radius: 0 !important;
            color: var(--text-mid) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 500 !important;
            font-size: 0.8rem !important;
            padding: 0.8rem 2rem !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            transition: all 0.15s ease !important;
            position: relative !important;
        }

        .stButton > button:hover {
            border-color: var(--neon-cyan) !important;
            color: var(--neon-cyan) !important;
            box-shadow: 0 0 20px rgba(0, 255, 242, 0.2) !important;
            transform: none !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--neon-green), var(--neon-cyan)) !important;
            border-color: var(--neon-green) !important;
            color: var(--void) !important;
            font-weight: 600 !important;
            box-shadow:
                0 0 20px rgba(57, 255, 20, 0.3),
                inset 0 0 20px rgba(255, 255, 255, 0.1) !important;
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 0 30px rgba(57, 255, 20, 0.5) !important;
        }

        /* Compare All Button - Secondary */
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 2px solid var(--neon-amber) !important;
            color: var(--neon-amber) !important;
            font-weight: 600 !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 170, 0, 0.15) !important;
            box-shadow: 0 0 30px rgba(255, 170, 0, 0.4) !important;
            color: var(--neon-amber) !important;
        }

        /* ══════════════════════════════════════════
           TEXT INPUT - Terminal Input
           ══════════════════════════════════════════ */
        .stTextArea textarea {
            background: var(--surface) !important;
            border: 1px solid var(--surface-3) !important;
            border-radius: 0 !important;
            color: var(--text-bright) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.9rem !important;
            padding: 1.2rem !important;
            line-height: 1.6 !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--neon-cyan) !important;
            box-shadow:
                0 0 0 1px var(--neon-cyan),
                0 0 30px rgba(0, 255, 242, 0.1) !important;
        }

        .stTextArea textarea::placeholder {
            color: var(--text-dim) !important;
        }

        /* ══════════════════════════════════════════
           RESULT CARDS - Threat Analysis Output
           ══════════════════════════════════════════ */
        .result-panel {
            position: relative;
            padding: 2.5rem;
            margin: 1.5rem 0;
            background: var(--surface);
            overflow: hidden;
        }

        .result-panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }

        .result-safe {
            border: 1px solid rgba(57, 255, 20, 0.3);
        }

        .result-safe::before {
            background: var(--neon-green);
            box-shadow: 0 0 20px var(--neon-green);
        }

        .result-danger {
            border: 1px solid rgba(255, 51, 102, 0.3);
            animation: threat-pulse 2s ease-in-out infinite;
        }

        .result-danger::before {
            background: var(--neon-red);
            box-shadow: 0 0 20px var(--neon-red);
        }

        @keyframes threat-pulse {
            0%, 100% {
                box-shadow: 0 0 20px rgba(255, 51, 102, 0.2);
            }
            50% {
                box-shadow: 0 0 40px rgba(255, 51, 102, 0.4);
            }
        }

        .result-status {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .result-safe .result-status { color: var(--neon-green); }
        .result-danger .result-status { color: var(--neon-red); }

        .result-verdict {
            font-family: 'Anybody', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin: 0.5rem 0;
        }

        .result-safe .result-verdict { color: var(--neon-green); }
        .result-danger .result-verdict { color: var(--neon-red); }

        .result-confidence {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 3.5rem;
            font-weight: 600;
            margin: 1rem 0;
        }

        .result-safe .result-confidence {
            color: var(--neon-green);
            text-shadow: 0 0 30px rgba(57, 255, 20, 0.5);
        }
        .result-danger .result-confidence {
            color: var(--neon-red);
            text-shadow: 0 0 30px rgba(255, 51, 102, 0.5);
        }

        .result-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-dim);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        /* ══════════════════════════════════════════
           COMPARISON TABLE - Multi-Model Analysis
           ══════════════════════════════════════════ */
        .comparison-container {
            background: var(--surface);
            border: 1px solid var(--surface-3);
            margin: 0.5rem 0 1.5rem;
            overflow: hidden;
        }

        .comparison-header {
            padding: 1.25rem;
            border-bottom: 1px solid var(--surface-3);
            text-align: center;
        }

        .consensus-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 1.5rem;
        }

        .consensus-icon {
            font-size: 1.25rem;
        }

        .consensus-text {
            font-family: 'Anybody', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .consensus-sub {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-dim);
            letter-spacing: 1px;
        }

        .consensus-safe .consensus-icon,
        .consensus-safe .consensus-text { color: var(--neon-green); }
        .consensus-danger .consensus-icon,
        .consensus-danger .consensus-text { color: var(--neon-red); }
        .consensus-mixed .consensus-icon,
        .consensus-mixed .consensus-text { color: var(--neon-amber); }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'IBM Plex Mono', monospace;
        }

        .comparison-table thead tr {
            background: var(--surface-2);
        }

        .comparison-table th {
            padding: 0.75rem 1rem;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 2px;
            color: var(--text-dim);
            text-align: left;
            border-bottom: 1px solid var(--surface-3);
        }

        .th-rank { width: 50px; text-align: center; }
        .th-model { width: 180px; }
        .th-verdict { width: 100px; text-align: center; }
        .th-prob { width: 100px; text-align: right; }
        .th-bar { }

        .comparison-table td {
            padding: 1rem;
            border-bottom: 1px solid var(--surface-3);
        }

        .comparison-table tbody tr {
            transition: background 0.15s ease;
        }

        .comparison-table tbody tr:hover {
            background: var(--surface-2);
        }

        .comparison-table tbody tr:last-child td {
            border-bottom: none;
        }

        .cell-rank {
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-dim);
        }

        .cell-model {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-bright);
        }

        .cell-verdict {
            text-align: center;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 2px;
        }

        .cell-verdict.safe { color: var(--neon-green); }
        .cell-verdict.danger { color: var(--neon-red); }

        .cell-prob {
            text-align: right;
            font-size: 1rem;
            font-weight: 600;
            font-family: 'Anybody', sans-serif;
            color: var(--neon-cyan);
        }

        .cell-bar {
            padding-right: 1.5rem;
        }

        .prob-bar-bg {
            height: 8px;
            background: var(--surface-3);
            border-radius: 0;
            overflow: hidden;
        }

        .prob-bar-fill {
            height: 100%;
            transition: width 0.5s ease;
        }

        .prob-bar-fill.safe {
            background: linear-gradient(90deg, var(--neon-green), rgba(57, 255, 20, 0.5));
        }

        .prob-bar-fill.danger {
            background: linear-gradient(90deg, var(--neon-red), rgba(255, 51, 102, 0.5));
        }

        .table-row-danger .cell-prob { color: var(--neon-red); }
        .table-row-safe .cell-prob { color: var(--neon-green); }

        /* ══════════════════════════════════════════
           METRICS GRID - Feature Indicators
           ══════════════════════════════════════════ */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin: 1rem 0;
        }

        .metric-box {
            background: var(--surface);
            border: 1px solid var(--surface-3);
            padding: 1rem;
            text-align: center;
        }

        .metric-box:hover {
            border-color: var(--neon-cyan);
        }

        .metric-value {
            font-family: 'Anybody', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--neon-cyan);
        }

        .metric-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            color: var(--text-dim);
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 0.25rem;
        }

        /* ══════════════════════════════════════════
           MODEL CARDS - Algorithm Info
           ══════════════════════════════════════════ */
        .model-info-card {
            background: var(--surface);
            border: 1px solid var(--surface-3);
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            transition: all 0.2s ease;
        }

        .model-info-card:hover {
            border-color: var(--neon-cyan);
        }

        .model-info-name {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--neon-cyan);
            margin-bottom: 0.5rem;
        }

        .model-info-desc {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-dim);
            margin-bottom: 1rem;
        }

        .model-info-stats {
            display: flex;
            gap: 2rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--surface-3);
        }

        .model-stat {
            text-align: center;
        }

        .model-stat-value {
            font-family: 'Anybody', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--neon-green);
        }

        .model-stat-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            color: var(--text-dim);
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        /* ══════════════════════════════════════════
           TABS - Navigation
           ══════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            gap: 0;
            border-bottom: 1px solid var(--surface-3);
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            color: var(--text-dim);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            padding: 1rem 1.5rem;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-bright);
        }

        .stTabs [aria-selected="true"] {
            background: transparent !important;
            border-bottom-color: var(--neon-cyan) !important;
            color: var(--neon-cyan) !important;
        }

        /* ══════════════════════════════════════════
           SECTION HEADERS
           ══════════════════════════════════════════ */
        .section-header {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--neon-cyan);
            letter-spacing: 3px;
            text-transform: uppercase;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--surface-3);
        }

        .section-header::before {
            content: '// ';
            color: var(--text-dim);
        }

        /* ══════════════════════════════════════════
           SELECT BOX
           ══════════════════════════════════════════ */
        .stSelectbox > div > div {
            background: var(--surface) !important;
            border: 1px solid var(--surface-3) !important;
            border-radius: 0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .stSelectbox > div > div:hover {
            border-color: var(--neon-cyan) !important;
        }

        /* ══════════════════════════════════════════
           FOOTER
           ══════════════════════════════════════════ */
        .terminal-footer {
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 1px solid var(--surface-3);
        }

        .footer-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-dim);
            letter-spacing: 2px;
        }

        .footer-brand {
            color: var(--neon-green);
        }

        /* ══════════════════════════════════════════
           SCROLLBAR
           ══════════════════════════════════════════ */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: var(--void); }
        ::-webkit-scrollbar-thumb { background: var(--surface-3); }
        ::-webkit-scrollbar-thumb:hover { background: var(--neon-cyan); }

        /* ══════════════════════════════════════════
           UTILITIES
           ══════════════════════════════════════════ */
        .text-center { text-align: center; }
        .mt-1 { margin-top: 0.5rem; }
        .mt-2 { margin-top: 1rem; }
        .mb-1 { margin-bottom: 0.5rem; }
        .mb-2 { margin-bottom: 1rem; }

        /* ══════════════════════════════════════════
           NAVIGATION LINKS
           ══════════════════════════════════════════ */
        [data-testid="stPageLink"] a {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem !important;
            color: var(--text-mid) !important;
            text-decoration: none !important;
            padding: 0.5rem 1rem !important;
            border: 1px solid var(--surface-3) !important;
            border-radius: 0 !important;
            transition: all 0.2s ease !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }

        [data-testid="stPageLink"] a:hover {
            color: var(--neon-cyan) !important;
            border-color: var(--neon-cyan) !important;
            background: rgba(0, 255, 242, 0.05) !important;
        }

        [data-testid="stPageLink"] a[aria-current="page"] {
            color: var(--neon-green) !important;
            border-color: var(--neon-green) !important;
        }

        /* ══════════════════════════════════════════
           NAVIGATION ACTIVE ITEM
           ══════════════════════════════════════════ */
        .nav-active {
            display: flex;
            justify-content: center;
            align-items: center;
            background: transparent;
            border: 1px solid var(--neon-cyan);
            color: var(--neon-cyan);
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
            font-size: 0.8rem;
            padding: 0.8rem 2rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            width: 100%;
            box-sizing: border-box;
        }
    </style>
    """


@st.cache_resource
def load_detector():
    """Load and cache the PhishingDetector."""
    return PhishingDetector()


class PhishingDetector:
    """Main detector class with optimized model loading."""

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.url_configs = {}
        self.preprocessor = get_ml_preprocessor()
        self.sms_feature_extractor = SMSFeatureExtractor()
        self.email_feature_extractor = EmailFeatureExtractor()
        self._load_models()

    def _load_models(self):
        """Load trained models."""
        models_dir = Path(__file__).parent.parent / "models"

        # SMS models
        sms_models = {
            'random_forest': 'random_forest.pkl',
            'logistic_regression': 'logistic_regression.pkl',
            'naive_bayes': 'naive_bayes.pkl',
            'xgboost': 'xgboost.pkl',
            'linear_svm': 'linear_svm.pkl'
        }

        for name, filename in sms_models.items():
            filepath = models_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        self.models[name] = pickle.load(f)
                except Exception:
                    pass

        # Email models
        email_models = {
            'email_logistic_regression': 'email_logistic_regression.pkl',
            'email_naive_bayes': 'email_naive_bayes.pkl',
            'email_random_forest': 'email_random_forest.pkl',
            'email_xgboost': 'email_xgboost.pkl'
        }

        for name, filename in email_models.items():
            filepath = models_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        self.models[name] = pickle.load(f)
                except Exception:
                    pass

        # Load vectorizers
        self._load_vectorizer(models_dir / 'tfidf_vectorizer.pkl', 'tfidf', 'char', 'svd')
        self._load_vectorizer(models_dir / 'email_tfidf_vectorizer.pkl', 'email_tfidf', 'email_char')

        # Load URL configs
        for config_name, filename in [('sms', 'url_features_config.pkl'), ('email', 'email_url_features_config.pkl')]:
            config_path = models_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'rb') as f:
                        self.url_configs[config_name] = pickle.load(f)
                except Exception:
                    pass

    def _load_vectorizer(self, path, word_key, char_key=None, svd_key=None):
        """Helper to load vectorizer files."""
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'word_vectorizer' in data:
                        self.vectorizers[word_key] = data['word_vectorizer']
                        if char_key and 'char_vectorizer' in data:
                            self.vectorizers[char_key] = data.get('char_vectorizer')
                        if svd_key and 'svd' in data:
                            self.vectorizers[svd_key] = data.get('svd')
                    else:
                        self.vectorizers[word_key] = data
            except Exception:
                pass

    def predict(self, text: str, model_name: str, text_type: str = 'sms') -> Dict[str, Any]:
        """Make prediction on input text."""
        clean_text = self.preprocessor.preprocess(text)

        # Extract features from RAW text
        if text_type == 'sms':
            features = self.sms_feature_extractor.extract_all_features(text)
            vec_key, char_key, svd_key, url_config_key = 'tfidf', 'char', 'svd', 'sms'
        else:
            features = self.email_feature_extractor.extract_all_features({
                'body': text, 'subject': '', 'headers': {}
            })
            vec_key, char_key, url_config_key = 'email_tfidf', 'email_char', 'email'
            svd_key = None

        # Adjust model name for email
        if text_type == 'email' and not model_name.startswith('email_'):
            email_model_name = f'email_{model_name}'
            if email_model_name in self.models:
                model_name = email_model_name

        if model_name not in self.models or vec_key not in self.vectorizers:
            return self._rule_based_detection(text, features, text_type)

        # TF-IDF features
        X_word = self.vectorizers[vec_key].transform([clean_text])

        if self.vectorizers.get(char_key) is not None:
            X_char = self.vectorizers[char_key].transform([clean_text])
            X_tfidf = sparse_hstack([X_word, X_char])
        else:
            X_tfidf = X_word

        if svd_key and self.vectorizers.get(svd_key) is not None:
            X_tfidf = self.vectorizers[svd_key].transform(X_tfidf)

        if hasattr(X_tfidf, "toarray"):
            X_tfidf = X_tfidf.toarray()

        # Combine with URL features
        if url_config_key in self.url_configs:
            url_config = self.url_configs[url_config_key]
            feature_names = url_config['feature_names']
            X_url = np.array([[features.get(name, 0) for name in feature_names]])
            X = np.hstack([X_tfidf, X_url])
        else:
            X = X_tfidf

        model = self.models[model_name]

        # Validar dimensionalidad antes de predecir
        expected_features = getattr(model, 'n_features_in_', None)
        if expected_features is not None and X.shape[1] != expected_features:
            # Mismatch de features - usar detección basada en reglas como fallback
            return self._rule_based_detection(text, features, text_type)

        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        return {
            'prediction': int(prediction),
            'probability': float(proba[1]) * 100,
            'confidence': max(proba) * 100,
            'features': features,
            'clean_text': clean_text,
            'model_used': model_name
        }

    def predict_all_models(self, text: str, text_type: str = 'sms') -> List[Dict[str, Any]]:
        """Run prediction with ALL available models for comparison."""
        if text_type == 'sms':
            model_list = ['naive_bayes', 'logistic_regression', 'random_forest', 'linear_svm', 'xgboost']
        else:
            model_list = ['email_naive_bayes', 'email_logistic_regression', 'email_random_forest', 'email_xgboost']

        available_models = [m for m in model_list if m in self.models]
        results = []

        for model_name in available_models:
            try:
                result = self.predict(text, model_name, text_type)
                result['model_key'] = model_name
                results.append(result)
            except Exception:
                pass

        return results

    def _rule_based_detection(self, text: str, features: dict, text_type: str) -> Dict[str, Any]:
        """Fallback rule-based detection."""
        score = 0
        max_score = 10

        if features.get('has_url', 0):
            score += 1
        if features.get('has_shortened_url', 0):
            score += 2
        if features.get('has_suspicious_tld', 0):
            score += 2

        urgency = features.get('urgency_count', 0) or features.get('urgency_keyword_count', 0)
        score += min(urgency, 2)

        suspicious_keywords = ['free', 'winner', 'prize', 'urgent', 'verify', 'suspended', 'click']
        text_lower = text.lower()
        score += min(sum(1 for kw in suspicious_keywords if kw in text_lower), 3)

        probability = (score / max_score) * 100

        return {
            'prediction': 1 if probability > 50 else 0,
            'probability': probability,
            'confidence': abs(probability - 50) + 50,
            'features': features,
            'clean_text': self.preprocessor.preprocess(text),
            'model_used': 'rule_based'
        }


def _style_verdict(val):
    """Style function for verdict column."""
    if val == 'THREAT':
        return 'color: #ff3366; font-weight: bold'
    return 'color: #39FF14; font-weight: bold'


def _style_prob(val):
    """Style function for probability column."""
    try:
        num = float(val.replace('%', ''))
        if num > 50:
            return 'color: #ff3366; font-weight: bold'
        return 'color: #39FF14; font-weight: bold'
    except Exception:
        return ''


def render_comparison_results(results: List[Dict], text_type: str):
    """Render comparison table for all model results."""
    if not results:
        st.warning("No hay resultados para mostrar.")
        return

    # Calculate consensus
    predictions = [r['prediction'] for r in results]
    fraud_count = sum(predictions)
    total = len(predictions)
    safe_count = total - fraud_count

    # Consensus badge
    if fraud_count == total:
        consensus_class = "consensus-danger"
        consensus_icon = "!"
        consensus_text = "THREAT CONFIRMED"
        consensus_sub = f"{total}/{total} models agree"
    elif safe_count == total:
        consensus_class = "consensus-safe"
        consensus_icon = "OK"
        consensus_text = "ALL CLEAR"
        consensus_sub = f"{total}/{total} models agree"
    else:
        consensus_class = "consensus-mixed"
        consensus_icon = "?"
        consensus_text = "MIXED RESULTS"
        consensus_sub = f"{fraud_count} threat / {safe_count} clear"

    # Build table rows using Streamlit columns for reliable rendering
    sorted_results = sorted(results, key=lambda x: x['probability'], reverse=True)

    # Display consensus header
    if consensus_class == "consensus-danger":
        st.error(f"**{consensus_icon} {consensus_text}** - {consensus_sub}")
    elif consensus_class == "consensus-safe":
        st.success(f"**{consensus_icon} {consensus_text}** - {consensus_sub}")
    else:
        st.warning(f"**{consensus_icon} {consensus_text}** - {consensus_sub}")

    # Create table using st.dataframe with custom styling
    table_data = []
    for i, result in enumerate(sorted_results):
        model_key = result.get('model_key', result['model_used'])
        model_info = MODEL_INFO.get(model_key, {'name': model_key})
        is_fraud = result['prediction'] == 1
        fraud_prob = result['probability']

        table_data.append({
            '#': i + 1,
            'MODEL': model_info.get('name', model_key),
            'VERDICT': 'THREAT' if is_fraud else 'CLEAR',
            'FRAUD %': f"{fraud_prob:.1f}%",
            'PROB': fraud_prob
        })

    df = pd.DataFrame(table_data)

    # Style the dataframe (use map for Pandas 2.x compatibility)
    styled_df = df[['#', 'MODEL', 'VERDICT', 'FRAUD %']].style.map(
        _style_verdict, subset=['VERDICT']
    ).map(
        _style_prob, subset=['FRAUD %']
    )

    st.dataframe(
        styled_df,
        width='stretch',
        hide_index=True,
        column_config={
            '#': st.column_config.NumberColumn('#', width='small'),
            'MODEL': st.column_config.TextColumn('MODEL', width='medium'),
            'VERDICT': st.column_config.TextColumn('VERDICT', width='small'),
            'FRAUD %': st.column_config.TextColumn('FRAUD %', width='small'),
        }
    )

    # Add progress bars below the table
    st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
    for result in sorted_results:
        model_key = result.get('model_key', result['model_used'])
        model_info = MODEL_INFO.get(model_key, {'name': model_key})
        fraud_prob = result['probability']
        is_fraud = result['prediction'] == 1

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{model_info.get('name', model_key)}**")
        with col2:
            if is_fraud:
                st.progress(fraud_prob / 100, text=f"{fraud_prob:.1f}% THREAT")
            else:
                st.progress((100 - fraud_prob) / 100, text=f"{100 - fraud_prob:.1f}% SAFE")


def render_model_card(model_key: str):
    """Render a model info card."""
    info = MODEL_INFO.get(model_key, {})
    if not info:
        return ""

    return f"""
    <div class="model-info-card">
        <div class="model-info-name">{info.get('name', model_key)}</div>
        <div class="model-info-desc">{info.get('description', '')}</div>
        <div class="model-info-stats">
            <div class="model-stat">
                <div class="model-stat-value">{info.get('accuracy', 'N/A')}</div>
                <div class="model-stat-label">Accuracy</div>
            </div>
            <div class="model-stat">
                <div class="model-stat-value">{info.get('f1', 'N/A')}</div>
                <div class="model-stat-label">F1 Score</div>
            </div>
        </div>
    </div>
    """


def save_analysis_to_db(result: Dict[str, Any], text_input: str, text_type: str) -> bool:
    """
    Save analysis result to database if user is authenticated.

    Args:
        result: Prediction result dictionary
        text_input: Original input text
        text_type: 'sms' or 'email'

    Returns:
        True if saved, False otherwise
    """
    if not DB_AVAILABLE or not AUTH_AVAILABLE:
        return False

    if not is_authenticated():
        return False

    user = get_current_user()
    if not user:
        return False

    try:
        analysis_id = save_analysis(
            text_input=text_input,
            text_type=text_type,
            model_used=result.get('model_used', 'unknown'),
            prediction=result.get('prediction', 0),
            probability=result.get('probability', 0.0),
            features=result.get('features'),
            user_id=user.get('id')
        )
        return analysis_id is not None
    except Exception:
        return False


def render_metrics(features: dict, text_type: str):
    """Render feature metrics."""
    metrics = [
        ('url_count', 'URLs'),
        ('urgency_count' if text_type == 'sms' else 'urgency_keyword_count', 'Urgency'),
        ('shortened_url_count', 'Short URLs'),
        ('has_suspicious_tld', 'Bad TLD')
    ]

    boxes = []
    for key, label in metrics:
        val = features.get(key, 0)
        display = str(int(val)) if isinstance(val, (int, float)) else str(val)
        boxes.append(f'<div class="metric-box"><div class="metric-value">{display}</div><div class="metric-label">{label}</div></div>')

    return f'<div class="metrics-row">{"".join(boxes)}</div>'


def main():
    """Main Streamlit application."""

    # CRITICAL: Inject splash IMMEDIATELY via JavaScript
    # st.components.v1.html() executes JS and can access parent window
    components.html("""
    <script>
        (function() {
            const parent = window.parent.document;

            // Check if splash already exists
            if (parent.getElementById('shield-splash')) return;

            // Create splash overlay
            const splash = parent.createElement('div');
            splash.id = 'shield-splash';
            splash.innerHTML = `
                <style>
                    #shield-splash {
                        position: fixed !important;
                        top: 0 !important;
                        left: 0 !important;
                        width: 100vw !important;
                        height: 100vh !important;
                        background: #050508 !important;
                        display: flex !important;
                        flex-direction: column !important;
                        align-items: center !important;
                        justify-content: center !important;
                        z-index: 999999 !important;
                        animation: shieldFadeOut 0.5s ease-out 2s forwards !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    }
                    #shield-splash .logo {
                        font-size: 4.5rem;
                        font-weight: 800;
                        letter-spacing: -3px;
                        background: linear-gradient(180deg, #00fff2 0%, #39FF14 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        animation: shieldGlow 1.5s ease-in-out infinite;
                    }
                    #shield-splash .status {
                        font-size: 0.7rem;
                        color: #4a5568;
                        letter-spacing: 4px;
                        text-transform: uppercase;
                        margin-top: 1.5rem;
                        font-family: 'Consolas', 'Monaco', monospace;
                    }
                    #shield-splash .loader {
                        margin-top: 1rem;
                        width: 180px;
                        height: 2px;
                        background: #1a1a24;
                        position: relative;
                        overflow: hidden;
                    }
                    #shield-splash .loader::after {
                        content: '';
                        position: absolute;
                        left: -50%;
                        height: 100%;
                        width: 50%;
                        background: linear-gradient(90deg, transparent, #00fff2, #39FF14, transparent);
                        animation: shieldLoading 1.2s ease-in-out infinite;
                    }
                    @keyframes shieldLoading {
                        0% { left: -50%; }
                        100% { left: 100%; }
                    }
                    @keyframes shieldGlow {
                        0%, 100% { filter: drop-shadow(0 0 20px rgba(0, 255, 242, 0.4)); }
                        50% { filter: drop-shadow(0 0 40px rgba(57, 255, 20, 0.7)); }
                    }
                    @keyframes shieldFadeOut {
                        to { opacity: 0; visibility: hidden; pointer-events: none; }
                    }
                </style>
                <div class="logo">SHIELD</div>
                <div class="status">Initializing Neural Systems</div>
                <div class="loader"></div>
            `;

            // Insert at the very beginning of body
            parent.body.insertBefore(splash, parent.body.firstChild);

            // Also hide toolbar immediately
            const hideElements = () => {
                const selectors = [
                    '[data-testid="stToolbar"]',
                    '[data-testid="stDecoration"]',
                    '[data-testid="stStatusWidget"]',
                    '.stDeployButton'
                ];
                selectors.forEach(sel => {
                    parent.querySelectorAll(sel).forEach(el => {
                        el.style.cssText = 'display:none!important;visibility:hidden!important;';
                    });
                });
            };
            hideElements();

            // MutationObserver to catch late-added elements
            const observer = new MutationObserver(hideElements);
            observer.observe(parent.body, { childList: true, subtree: true });

            // Stop observing after 5 seconds
            setTimeout(() => observer.disconnect(), 5000);

            // Remove splash from DOM after animation completes (2.5s = 2s delay + 0.5s fade)
            setTimeout(() => {
                const splashEl = parent.getElementById('shield-splash');
                if (splashEl) splashEl.remove();
            }, 2600);
        })();
    </script>
    """, height=0)

    # Inject main CSS
    st.markdown(get_css(), unsafe_allow_html=True)

    # Authentication - require login before showing app
    if AUTH_AVAILABLE:
        st.markdown(get_auth_css(), unsafe_allow_html=True)
        if not render_auth_page():
            st.stop()

    # Hero Section
    st.markdown("""
    <div class="hero-terminal">
        <h1 class="hero-title">SHIELD</h1>
        <p class="hero-subtitle">Threat Detection <span>//</span> Neural Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

    # User bar with navigation (show if authenticated)
    if AUTH_AVAILABLE and is_authenticated():
        user = get_current_user()

        # Navigation bar
        nav_cols = st.columns([1, 1, 1, 2, 1, 1])

        with nav_cols[0]:
            st.markdown('<div class="nav-active">🔍 SCANNER</div>', unsafe_allow_html=True)

        with nav_cols[1]:
            if st.button("📜 History", use_container_width=True, key="nav_history"):
                st.switch_page("pages/history.py")

        with nav_cols[2]:
            # Admin link - only show for admins
            if user and user.get('role') == 'admin':
                if st.button("👑 Admin", use_container_width=True, key="nav_admin"):
                    st.switch_page("pages/admin_dashboard.py")

        with nav_cols[4]:
            render_user_menu()

        with nav_cols[5]:
            render_logout_button()

    # Load detector (cached)
    detector = load_detector()

    # Initialize session state
    if 'text_type' not in st.session_state:
        st.session_state.text_type = 'sms'

    # Type Selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        type_col1, type_col2 = st.columns(2)
        with type_col1:
            if st.button("SMS // SMISHING", use_container_width=True,
                        type="primary" if st.session_state.text_type == 'sms' else "secondary"):
                st.session_state.text_type = 'sms'
                st.rerun()
        with type_col2:
            if st.button("EMAIL // PHISHING", use_container_width=True,
                        type="primary" if st.session_state.text_type == 'email' else "secondary"):
                st.session_state.text_type = 'email'
                st.rerun()

    text_type = st.session_state.text_type

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["ANALYZE", "MODELS", "BATCH"])

    with tab1:
        # Model selection - include "Compare All" option
        if text_type == 'sms':
            available_models = [m for m in ['linear_svm', 'logistic_regression', 'xgboost', 'random_forest', 'naive_bayes']
                              if m in detector.models]
        else:
            available_models = [m for m in ['email_logistic_regression', 'email_random_forest', 'email_xgboost', 'email_naive_bayes']
                              if m in detector.models]

        # Add "Compare All" as first option
        model_options = ['compare_all'] + available_models

        def format_model_name(x):
            if x == 'compare_all':
                return '⧉ COMPARE // ALL MODELS'
            name = MODEL_INFO.get(x, {}).get('name', x)
            # Add recommended tag to Naive Bayes models
            if 'naive_bayes' in x:
                return f"{name} (Recomendado)"
            return name

        col1, col2 = st.columns([3, 1])
        with col2:
            if available_models:
                selected_model = st.selectbox(
                    "Model",
                    model_options,
                    format_func=format_model_name
                )
            else:
                selected_model = 'rule_based'
                st.warning("No models loaded")

        # Text Input
        placeholder = "> Enter suspicious SMS message..." if text_type == 'sms' else "> Paste email content for analysis..."
        input_text = st.text_area("Input", height=140, placeholder=placeholder, label_visibility="collapsed")

        # Single Analyze Button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze = st.button("◈ ANALYZE", use_container_width=True, type="primary", key="analyze_btn")

        # Analysis Results
        if analyze and input_text:
            # Compare All Models Mode
            if selected_model == 'compare_all':
                with st.spinner("Running all models..."):
                    results = detector.predict_all_models(input_text, text_type)

                # Save best result to database (highest confidence)
                if results:
                    best_result = max(results, key=lambda x: x.get('probability', 0) if x.get('prediction', 0) == 1 else 100 - x.get('probability', 0))
                    if save_analysis_to_db(best_result, input_text, text_type):
                        st.toast("Analysis saved to history", icon="💾")

                render_comparison_results(results, text_type)

            # Single Model Mode
            else:
                with st.spinner("Processing..."):
                    result = detector.predict(input_text, selected_model, text_type)

                # Save analysis to database (if user is logged in)
                if save_analysis_to_db(result, input_text, text_type):
                    st.toast("Analysis saved to history", icon="💾")

                if result['prediction'] == 1:
                    threat_type = "SMISHING DETECTED" if text_type == 'sms' else "PHISHING DETECTED"
                    st.markdown(f"""
                    <div class="result-panel result-danger text-center">
                        <div class="result-status">⚠ Threat Analysis Complete</div>
                        <div class="result-verdict">{threat_type}</div>
                        <div class="result-confidence">{result['probability']:.1f}%</div>
                        <div class="result-label">Threat Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-panel result-safe text-center">
                        <div class="result-status">✓ Scan Complete</div>
                        <div class="result-verdict">MESSAGE CLEAR</div>
                        <div class="result-confidence">{100 - result['probability']:.1f}%</div>
                        <div class="result-label">Safety Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Feature metrics
                st.markdown('<div class="section-header">Detected Indicators</div>', unsafe_allow_html=True)
                st.markdown(render_metrics(result['features'], text_type), unsafe_allow_html=True)

                # XAI Explanation Section
                if XAI_AVAILABLE:
                    st.markdown('<div class="section-header">Explainability Analysis</div>', unsafe_allow_html=True)
                    render_xai_section(
                        text=input_text,
                        model_name=selected_model,
                        detector=detector,
                        text_type=text_type
                    )

                with st.expander("View Raw Data"):
                    st.json(result['features'])

        elif analyze and not input_text:
            st.warning("Input required for analysis.")

    with tab2:
        st.markdown('<div class="section-header">SMS Detection Models</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, model_key in enumerate(['linear_svm', 'logistic_regression', 'xgboost', 'random_forest', 'naive_bayes']):
            if model_key in detector.models:
                with cols[i % 2]:
                    st.markdown(render_model_card(model_key), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Email Detection Models</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, model_key in enumerate(['email_logistic_regression', 'email_random_forest', 'email_xgboost', 'email_naive_bayes']):
            if model_key in detector.models:
                with cols[i % 2]:
                    st.markdown(render_model_card(model_key), unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">Batch Analysis</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV with messages", type=['csv'])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                text_col = None
                for col in ['text', 'message', 'body', 'sms', 'email', 'content']:
                    if col in df.columns:
                        text_col = col
                        break

                if text_col is None:
                    st.error("No text column found.")
                else:
                    st.success(f"{len(df)} messages loaded")
                    st.dataframe(df.head(), width='stretch')

                    if st.button("◈ ANALYZE BATCH", type="primary"):
                        progress = st.progress(0)
                        results = []

                        batch_model = available_models[0] if available_models else 'rule_based'

                        for i, row in df.iterrows():
                            text = str(row[text_col])
                            result = detector.predict(text, batch_model, text_type)
                            results.append({
                                'text': text[:50] + '...' if len(text) > 50 else text,
                                'verdict': 'THREAT' if result['prediction'] == 1 else 'CLEAR',
                                'probability': f"{result['probability']:.1f}%"
                            })
                            progress.progress((i + 1) / len(df))

                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, width='stretch')

                        fraud_count = sum(1 for r in results if 'THREAT' in r['verdict'])
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total", len(results))
                        col2.metric("Threats", fraud_count)
                        col3.metric("Clear", len(results) - fraud_count)

                        csv = results_df.to_csv(index=False)
                        st.download_button("Download Results", csv, "shield_results.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

    # Footer
    st.markdown("""
    <div class="terminal-footer">
        <p class="footer-text">
            <span class="footer-brand">SHIELD</span> // Neural Threat Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
