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
