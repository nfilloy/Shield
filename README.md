# Shield: Phishing and Smishing Detection

## 📋 Overview

**Phishing (emails) and smishing (SMS) detection** system using NLP and Machine Learning.
Includes a **user management system** with authentication, analysis history, and an administration dashboard.
Bachelor's Thesis (Trabajo de Fin de Grado - TFG) project.

---

## 🚀 Quick Commands

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run main web application
streamlit run app/streamlit_app.py

# Run additional pages
streamlit run app/pages/history.py          # User's analysis history
streamlit run app/pages/admin_dashboard.py  # Admin dashboard

# Train SMS models (via wrapper)
python scripts/train_sms_models.py

# Train Email models (via wrapper)
python scripts/train_email_models.py

# Run from command line (unified CLI)
python -m src.main train --type sms
python -m src.main train --type email
python -m src.main predict "Suspicious message here" --type sms
python -m src.main predict "Suspicious email here" --type email
python -m src.main app

# Initialize database
python -c "from src.database import init_db; init_db()"

# Create admin user
python -c "from src.auth import register_user; register_user('admin', 'admin@shield.io', 'AdminPass123', role='admin')"
```

---

## 📁 Project Structure

```
TFG/
├── app/
│   ├── streamlit_app.py        # Main Streamlit web interface
│   ├── components/
│   │   ├── xai_display.py      # XAI visualization components
│   │   └── auth.py             # Authentication UI components
│   └── pages/
│       ├── history.py          # User's analysis history
│       └── admin_dashboard.py  # Administration dashboard
├── archive/                    # Archived modules (not currently in use)
│   ├── deep_learning.py        # DL models (future)
│   └── transformers.py         # Transformer models (future)
├── data/
│   ├── external/               # Downloaded datasets (Mendeley, HuggingFace cache)
│   └── shield.db               # SQLite database (users and analyses)
├── docs/                       # Documentation
│   ├── notas/                  # Notes and drafts
│   ├── sesiones/               # Development session logs
│   └── *.pdf                   # Reference documents
├── models/                     # Trained models (.pkl)
│   ├── logistic_regression.pkl         # SMS models
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── linear_svm.pkl
│   ├── xgboost.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── url_features_config.pkl         # SMS URL features config
│   ├── email_logistic_regression.pkl   # Email models
│   ├── email_naive_bayes.pkl
│   ├── email_random_forest.pkl
│   ├── email_xgboost.pkl
│   ├── email_tfidf_vectorizer.pkl
│   └── email_url_features_config.pkl   # Email URL features config
├── scripts/                    # Execution wrappers
│   ├── train_sms_models.py     # Wrapper for SMS flow
│   └── train_email_models.py   # Wrapper for Email flow
├── src/
│   ├── data/                   # Loading and preprocessing
│   │   ├── sms_loader.py       # SMS Loader (Mendeley + HuggingFace smishing)
│   │   ├── email_loader.py     # Email Loader (HuggingFace phishing)
│   │   └── preprocessor.py     # Text preprocessor
│   ├── database/               # Database and persistence
│   │   ├── models.py           # SQLAlchemy Models (User, Analysis)
│   │   ├── connection.py       # DB connection and sessions
│   │   └── analysis_service.py # Analysis management service
│   ├── auth/                   # User authentication
│   │   └── auth_service.py     # Registration, login, validation
│   ├── training/               # Centralized training logic
│   │   └── pipeline.py         # Unified TrainingPipeline
│   ├── features/               # Feature extraction
│   │   ├── text_features.py    # TF-IDF, n-grams
│   │   ├── sms_features.py     # Specific SMS features
│   │   └── email_features.py   # Specific email features
│   ├── explainability/         # XAI Module (Explainability)
│   │   ├── base.py             # Base class for explainers
│   │   ├── shap_explainer.py   # SHAP explainer
│   │   ├── lime_explainer.py   # LIME explainer
│   │   └── text_highlighter.py # HTML text highlighting
│   ├── models/                 # ML models
│   │   └── classical.py        # Wrapper for Sklearn/XGBoost classes
│   └── main.py                 # CLI entry point
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── config.yaml                 # System configuration
└── README.md                   # Main documentation
```

---

## 🤖 Models and Datasets

### SMS (Smishing) - Combined Datasets

**Combined data sources:**
1. **Mendeley SMS Phishing Dataset** ([f45bkkt8pr](https://data.mendeley.com/datasets/f45bkkt8pr/1))
   - Location: `data/external/SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION/Dataset_5971.csv`
   - 5,971 messages: 638 smishing + 489 spam + 4,844 ham
   - Filtered for SMISHING vs HAM only (excludes 489 generic spam)
   - Columns: LABEL, TEXT, URL, EMAIL, PHONE

2. **Hugging Face ucirvine/sms_spam** (UCI SMS Spam Collection)
   - 5,574 messages labeled as spam/ham
   - Standard reference dataset for SMS spam detection

**Final combined dataset:** 5,960 unique messages (after removing 5,096 duplicates)
- 919 smishing (15.4%)
- 5,041 legitimate (84.6%)

| Model | File |
|--------|---------|
| Linear SVM | `linear_svm.pkl` |
| XGBoost | `xgboost.pkl` |
| Naive Bayes | `naive_bayes.pkl` |
| Random Forest | `random_forest.pkl` |
| Logistic Regression | `logistic_regression.pkl` |

**Features:** TF-IDF + URL/structural features
**Vectorizer:** `tfidf_vectorizer.pkl` + `url_features_config.pkl`

### Email (Phishing) - Email Dataset

**Data source:**
- **Hugging Face zefang-liu/phishing-email-dataset**
  - Local path: `data/external/phishing_email_dataset.csv`
  - 17,515 emails (after removing duplicates)
  - 6,539 phishing (37.3%) + 10,976 safe (62.7%)
  - Labels: "Phishing Email" (1), "Safe Email" (0)
  - Includes phishing emails with real tactics: urgency, impersonation, malicious links

| Model | File |
|--------|---------|
| XGBoost | `email_xgboost.pkl` |
| Random Forest | `email_random_forest.pkl` |
| Logistic Regression | `email_logistic_regression.pkl` |
| Naive Bayes | `email_naive_bayes.pkl` |

**Features:** TF-IDF + URL/structural features
**Vectorizer:** `email_tfidf_vectorizer.pkl` + `email_url_features_config.pkl`

> **Note:** Ensemble models (XGBoost, Random Forest) perform best with the TF-IDF + URL features combination. An internet connection is required to download Hugging Face datasets (the first time).

---

## 🔧 Main Components

### 1. TrainingPipeline (`src/training/pipeline.py`)
Core class that orchestrates the entire training flow.
```python
from src.training.pipeline import TrainingPipeline
pipeline = TrainingPipeline()
results = pipeline.run(data_type='email') # or 'sms'
```

### 2. SMSLoader (`src/data/sms_loader.py`)
Loads real smishing datasets from Mendeley and Hugging Face.
```python
from src.data.sms_loader import SMSLoader
loader = SMSLoader()

# Load combined dataset (Mendeley + HuggingFace)
df = loader.load_combined_smishing()

# Or load individually
df_mendeley = loader.load_mendeley_smishing()
df_hf = loader.load_huggingface_sms()
```

### 3. EmailLoader (`src/data/email_loader.py`)
Loads phishing dataset from Hugging Face.
```python
from src.data.email_loader import EmailLoader
loader = EmailLoader()

# Load phishing dataset
df = loader.load_huggingface_phishing()
```

### 4. ClassicalModels (`src/models/classical.py`)
Unified wrapper for Scikit-Learn and XGBoost models with predefined parameters.

### 5. Streamlit App (`app/streamlit_app.py`)
- Dynamic message type selector (SMS/Email).
- Automatic loading of the corresponding model and vectorizer.
- Integrated XAI explainability section.

### 6. XAI Module (`src/explainability/`)
Explainability system to understand model predictions.

```python
from src.explainability import LIMEExplainer, SHAPExplainer

# LIME - Local interpretable explanations
lime_exp = LIMEExplainer(model, vectorizer, preprocessor=preprocessor)
result = lime_exp.explain(text, num_features=10)
print(result['word_weights'])  # [(word, weight, direction), ...]

# SHAP - Shapley values
shap_exp = SHAPExplainer(model, vectorizer, model_type='tree')
result = shap_exp.explain(text)
print(result['feature_importance'])  # {feature: shap_value, ...}
```

**Components:**
| File | Description |
|---------|-------------|
| `base.py` | Abstract `BaseExplainer` class |
| `lime_explainer.py` | LIME explanations (model-agnostic) |
| `shap_explainer.py` | SHAP explanations (TreeExplainer, LinearExplainer) |
| `text_highlighter.py` | HTML generation with highlighted text |

---

## ⚙️ Configuration (`config.yaml`)

Centralized hyperparameter control:
```yaml
preprocessing:
  lowercase: true
  replace_urls: true

models:
  random_forest:
    n_estimators: 200
```

---

## 🗄️ Database (`src/database/`)

Persistence system with **SQLite + SQLAlchemy 2.0** for user and analysis management.

### Data Models

**User** (`users` table):
| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key, autoincrement |
| username | String(50) | Unique, indexed |
| email | String(255) | Unique, indexed |
| password_hash | String(255) | bcrypt hash |
| role | String(20) | 'user' or 'admin' |
| created_at | DateTime | Creation timestamp |
| last_login | DateTime | Last login (nullable) |

**Analysis** (`analyses` table):
| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key |
| user_id | Integer FK | Reference to User (nullable) |
| text_input | Text | Analyzed text |
| text_type | String(20) | 'sms' or 'email' |
| model_used | String(50) | Model name |
| prediction | Integer | 0=safe, 1=threat |
| probability | Float | Probability 0-100 |
| features_json | Text | Extracted features (JSON) |
| created_at | DateTime | Timestamp, indexed |

### Usage

```python
from src.database import init_db, get_db_session, User, Analysis

# Initialize DB (create tables)
init_db()

# Use context manager for operations
with get_db_session() as session:
    user = User(username="test", email="test@test.io", password_hash="...")
    session.add(user)

# Save analysis
from src.database import save_analysis
save_analysis(
    text_input="Click here to win!",
    text_type="sms",
    model_used="xgboost",
    prediction=1,
    probability=94.5,
    user_id=1
)

# Get history
from src.database import get_user_analyses, get_user_stats
history = get_user_analyses(user_id=1, limit=50)
stats = get_user_stats(user_id=1)
```

### Analysis Service Functions

| Function | Description |
|---------|-------------|
| `save_analysis()` | Save new analysis |
| `get_user_analyses()` | User's paginated history |
| `get_user_stats()` | User statistics |
| `get_global_stats()` | Global statistics (admin) |
| `get_recent_analyses()` | Recent analyses (admin) |
| `delete_analysis()` | Delete analysis |

---

## 🔐 Authentication (`src/auth/`)

Authentication system using **bcrypt** for password hashing.

### Main Functions

```python
from src.auth import register_user, authenticate_user, AuthResult

# Register user
result = register_user(
    username="user",
    email="user@example.com",
    password="Password123",
    role="user"  # or "admin"
)
if result.success:
    user = result.user

# Authenticate (login)
result = authenticate_user("user", "Password123")
# Also accepts email: authenticate_user("user@example.com", "Password123")
if result.success:
    user = result.user
```

### Password Validations

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit

### Available Functions

| Function | Description |
|---------|-------------|
| `register_user()` | Registration with validations |
| `authenticate_user()` | Login via username/email |
| `validate_password()` | Validate requirements |
| `validate_email()` | Validate email format |
| `validate_username()` | Validate username format |
| `get_user_by_id()` | Get user by ID |
| `get_user_by_username()` | Get by username |
| `update_password()` | Change password |
| `is_admin()` | Verify admin role |
| `get_all_users()` | List users (admin) |
| `delete_user()` | Delete user |

---

## 🎨 UI Authentication Components (`app/components/auth.py`)

Streamlit components with a cyber-noir design for login/registration.

### Basic Usage

```python
from app.components.auth import render_auth_page, is_authenticated, get_current_user

# In your Streamlit page:
if not render_auth_page():
    st.stop()  # Shows login if not authenticated

# Authenticated user
user = get_current_user()
st.write(f"Welcome {user['username']}")
```

### UI Functions

| Function | Description |
|---------|-------------|
| `render_auth_page()` | Full login/registration page |
| `render_logout_button()` | Logout button |
| `render_user_menu()` | User badge |
| `is_authenticated()` | Check if session exists |
| `is_admin()` | Verify admin role |
| `get_current_user()` | Current user data |
| `@require_auth` | Decorator to protect functions |
| `@require_admin` | Decorator for admin functions |

### Session State

```python
st.session_state.authenticated  # bool
st.session_state.user           # dict {id, username, email, role}
st.session_state.auth_mode      # 'login' or 'register'
```

---

## 📜 Additional Pages (`app/pages/`)

### History (`history.py`)
Analysis history for the logged-in user.
- Table containing all analyses
- Filters by type (SMS/Email), verdict, date
- Export to CSV
- Personal statistics

```powershell
streamlit run app/pages/history.py
```

### Admin Dashboard (`admin_dashboard.py`)
Administration panel (role='admin' only).
- Global statistics: users, analyses, threats
- Daily activity graph (last 30 days)
- Distribution by type (donut chart)
- Model usage (bar chart)
- Recent activity table

```powershell
streamlit run app/pages/admin_dashboard.py
```

---

## 📝 Development Notes
- **Real Phishing Datasets**: Exclusively uses real phishing/smishing datasets (Mendeley, HuggingFace), not generic spam.
- **Complete Refactoring**: Duplicated code in training scripts was removed by moving the logic to `src/training/pipeline.py`.
- **Flexibility**: The system easily supports adding new data types (like 'whatsapp' or 'social_media') by extending the pipeline and loaders.
- **Centralized Preprocessor**: The `get_ml_preprocessor()` function in `src/data/preprocessor.py` ensures consistency between training and inference.
- **Integrated URL Analysis**: Models now combine TF-IDF with specific URL features extracted from the original text (before preprocessing).
- **HuggingFace Dependency**: The `datasets` library is required to load the datasets (`pip install datasets`).
- **User System**: Authentication with bcrypt, roles (user/admin), analysis history per user.
- **SQLite Persistence**: Lightweight database in `data/shield.db`, automatic initialization on app startup.
- **Automatic Saving**: Analyses are saved automatically if the user is logged in.
- **Cyber-Noir Design**: Consistent UI with a dark theme, neon colors (cyan, green, red, amber).

---

## 🔗 Feature Engineering (URL Analysis)

**IMPORTANT**: Models now combine TF-IDF features with URL analysis from the **original** text (before preprocessing).

### Extracted URL Features

#### For SMS (`SMSFeatureExtractor` in `src/features/sms_features.py`):
| Category | Features |
|-----------|-----------------|
| **URLs** | `url_count`, `has_url`, `url_ratio`, `shortened_url_count`, `has_shortened_url`, `suspicious_tld_count`, `has_suspicious_tld`, `ip_url_count`, `has_ip_url` |
| **Keywords** | `urgency_count`, `promo_count`, `financial_count`, `action_count`, `total_suspicious_keywords` |
| **Characters** | `uppercase_ratio`, `digit_ratio`, `special_char_ratio`, `emoji_count`, `money_mention_count` |
| **Patterns** | `has_verification_code`, `has_tracking_number`, `impersonation_indicator` |

#### For Email (`EmailFeatureExtractor` in `src/features/email_features.py`):
| Category | Features |
|-----------|-----------------|
| **URLs** | `url_count`, `has_urls`, `url_to_text_ratio`, `shortened_url_count`, `suspicious_tld_count`, `ip_url_count`, `unique_domain_count` |
| **Headers** | `sender_domain_length`, `sender_is_legitimate_domain`, `sender_has_suspicious_tld`, `reply_to_mismatch` |
| **Content** | `urgency_keyword_count`, `phishing_keyword_count`, `subject_has_urgency`, `misspelling_indicator` |
| **Structure** | `html_tag_count`, `link_text_mismatch_count`, `email_addresses_in_body` |

### Suspicious Domains Detected
- **URL Shorteners**: bit.ly, tinyurl.com, goo.gl, t.co, ow.ly, is.gd, buff.ly, adf.ly, snip.ly, sniply.io, clck.ru, v.gd, bc.vc, po.st, etc.
- **Suspicious TLDs**: .xyz, .top, .work, .click, .link, .info, .online, .tk, .ga, .ml, .cf, etc.
- **IP-based URLs**: URLs with IP addresses instead of domains.

### Prefix-less URL Detection
The system detects URLs **without** `http://` or `https://` if they end in known TLDs:
- **Shorteners**: .ly, .to, .gl, .gg, .cc
- **Countries**: .com, .org, .net, .co, .uk, .io, .me, .es, .de, .fr, .it, .ru, .cn, .in, .br, .mx, .ar, .cl, .pe, .us, .tv
- **Generic**: .info, .biz, .xyz, .top, .click, .link, .online, .site, .tech, .store, .shop, .app, .dev, .ai, .cloud

Detected examples:
- `snip.ly/servicioseur` ✓
- `bit.ly/oferta` ✓
- `pagina.es/registro` ✓
- `malware.xyz` ✓

### Configuration Files
```
models/
├── url_features_config.pkl       # Config for SMS
└── email_url_features_config.pkl # Config for Email
```

These files store the feature order to ensure consistency between training and inference.

---

## ⚠️ Consistent Preprocessing

**IMPORTANT**: For correct predictions, always use `get_ml_preprocessor()`:

```python
from src.data.preprocessor import get_ml_preprocessor

preprocessor = get_ml_preprocessor()
clean_text = preprocessor.preprocess(raw_text)
```

Preprocessor configuration (identical in training and inference):
- `lowercase=True`
- `remove_html=True`
- `replace_urls=True` (URLs -> `<URL>`)
- `replace_emails=True` (emails -> `<EMAIL>`)
- `replace_numbers=True` (numbers -> `<NUM>`)
- `remove_extra_whitespace=True`
