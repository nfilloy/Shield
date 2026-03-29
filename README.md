# Shield

Phishing and smishing detection system based on NLP and machine learning models.

## Project Contents

- `app/`: Streamlit interface and application pages.
- `src/`: core logic (data, features, models, evaluation, and explainability).
- `scripts/`: training and validation scripts.
- `tests/`: unit tests.
- `data/`: project datasets.
- `models/`: trained models and artifacts.
- `reports/`: results and evaluation outputs.

## Requirements

- Python 3.10 or higher
- pip

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app/streamlit_app.py
```

## Training

```bash
python scripts/train_sms_models.py
python scripts/train_email_models.py
```

## Tests

```bash
pytest
```

## Note

This repository (`Shield`) is the clean and stable version of the project, separate from the testing repository (`TFG`).
