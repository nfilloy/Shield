"""
User Analysis History Page for SHIELD Application.

Displays analysis history with filtering and export capabilities.
Cyber-noir terminal aesthetic.
"""

import streamlit as st
import pandas as pd
import html
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database import get_user_analyses, get_user_stats, delete_analysis
from app.components.auth import (
    render_auth_page,
    is_authenticated,
    get_current_user,
    render_logout_button,
    get_auth_css
)

# Page configuration
st.set_page_config(
    page_title="SHIELD // Analysis History",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def get_history_css():
    """Return CSS styles for history page."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Anybody:wght@400;600;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

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

        .stApp {
            background: var(--void);
        }

        #MainMenu, footer, header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden !important;}

        .main .block-container {
            padding: 1rem 2rem;
            max-width: 1400px;
        }

        /* ══════════════════════════════════════════
           PAGE HEADER
           ══════════════════════════════════════════ */
        .history-header {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 1rem;
        }

        .history-title {
            font-family: 'Anybody', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(180deg, #00fff2 0%, #39FF14 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        .history-subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: #4a5568;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        /* ══════════════════════════════════════════
           STATS CARDS
           ══════════════════════════════════════════ */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .stat-card {
            background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
            border: 1px solid #1a1a24;
            padding: 1.2rem;
            text-align: center;
            position: relative;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--neon-cyan);
        }

        .stat-card.threats::before {
            background: var(--neon-red);
        }

        .stat-card.safe::before {
            background: var(--neon-green);
        }

        .stat-value {
            font-family: 'Anybody', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: #00fff2;
        }

        .stat-card.threats .stat-value {
            color: #ff3366;
        }

        .stat-card.safe .stat-value {
            color: #39FF14;
        }

        .stat-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #4a5568;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 0.3rem;
        }

        /* ══════════════════════════════════════════
           FILTERS
           ══════════════════════════════════════════ */
        .filters-container {
            background: #0a0a0f;
            border: 1px solid #1a1a24;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }

        .filter-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: #8892a0;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        /* ══════════════════════════════════════════
           HISTORY TABLE
           ══════════════════════════════════════════ */
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        .history-table th {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: #4a5568;
            letter-spacing: 2px;
            text-transform: uppercase;
            text-align: left;
            padding: 0.8rem 1rem;
            border-bottom: 1px solid #1a1a24;
            background: #0a0a0f;
        }

        .history-table td {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #e2e8f0;
            padding: 1rem;
            border-bottom: 1px solid #12121a;
            background: #050508;
        }

        .history-table tr:hover td {
            background: #0a0a0f;
        }

        .verdict-threat {
            color: #ff3366;
            font-weight: 600;
        }

        .verdict-safe {
            color: #39FF14;
            font-weight: 600;
        }

        .type-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            font-size: 0.65rem;
            letter-spacing: 1px;
            border: 1px solid;
        }

        .type-sms {
            color: #00fff2;
            border-color: rgba(0, 255, 242, 0.3);
            background: rgba(0, 255, 242, 0.1);
        }

        .type-email {
            color: #ffaa00;
            border-color: rgba(255, 170, 0, 0.3);
            background: rgba(255, 170, 0, 0.1);
        }

        .probability-bar {
            width: 100px;
            height: 6px;
            background: #1a1a24;
            position: relative;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            transition: width 0.3s ease;
        }

        .probability-fill.threat {
            background: linear-gradient(90deg, #ff3366, #ff6699);
        }

        .probability-fill.safe {
            background: linear-gradient(90deg, #39FF14, #66ff66);
        }

        /* ══════════════════════════════════════════
           EMPTY STATE
           ══════════════════════════════════════════ */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: #0a0a0f;
            border: 1px dashed #1a1a24;
            margin: 2rem 0;
        }

        .empty-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .empty-title {
            font-family: 'Anybody', sans-serif;
            font-size: 1.2rem;
            color: #4a5568;
            margin-bottom: 0.5rem;
        }

        .empty-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #4a5568;
        }

        /* ══════════════════════════════════════════
           BUTTONS
           ══════════════════════════════════════════ */
        .stButton > button {
            background: transparent !important;
            border: 1px solid #1a1a24 !important;
            border-radius: 0 !important;
            color: #8892a0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem !important;
            padding: 0.6rem 1.2rem !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
        }

        .stButton > button:hover {
            border-color: #00fff2 !important;
            color: #00fff2 !important;
        }

        .stDownloadButton > button {
            background: linear-gradient(90deg, #39FF14, #00fff2) !important;
            border: none !important;
            color: #050508 !important;
            font-weight: 600 !important;
        }

        /* ══════════════════════════════════════════
           SELECT BOX
           ══════════════════════════════════════════ */
        .stSelectbox > div > div {
            background: #050508 !important;
            border: 1px solid #1a1a24 !important;
            border-radius: 0 !important;
        }

        .stSelectbox label {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.7rem !important;
            color: #8892a0 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
        }

        /* Date input */
        .stDateInput > div > div > input {
            background: #050508 !important;
            border: 1px solid #1a1a24 !important;
            border-radius: 0 !important;
            color: #e2e8f0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .stDateInput label {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.7rem !important;
            color: #8892a0 !important;
        }

        /* ══════════════════════════════════════════
           NAVIGATION LINKS
           ══════════════════════════════════════════ */
        [data-testid="stPageLink"] a {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem !important;
            color: #8892a0 !important;
            text-decoration: none !important;
            padding: 0.5rem 1rem !important;
            border: 1px solid #1a1a24 !important;
            border-radius: 0 !important;
            transition: all 0.2s ease !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }

        [data-testid="stPageLink"] a:hover {
            color: #00fff2 !important;
            border-color: #00fff2 !important;
            background: rgba(0, 255, 242, 0.05) !important;
        }

        [data-testid="stPageLink"] a[aria-current="page"] {
            color: #39FF14 !important;
            border-color: #39FF14 !important;
        }

        /* ══════════════════════════════════════════
           NAVIGATION ACTIVE ITEM
           ══════════════════════════════════════════ */
        .nav-active {
            display: flex;
            justify-content: center;
            align-items: center;
            background: transparent;
            border: 1px solid #00fff2;
            color: #00fff2;
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


def render_stats_cards(stats: dict):
    """Render statistics cards."""
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['total']}</div>
            <div class="stat-label">Total Analyses</div>
        </div>
        <div class="stat-card threats">
            <div class="stat-value">{stats['threats']}</div>
            <div class="stat-label">Threats Detected</div>
        </div>
        <div class="stat-card safe">
            <div class="stat-value">{stats['safe']}</div>
            <div class="stat-label">Safe Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['threat_rate']:.1f}%</div>
            <div class="stat-label">Threat Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_history_table(analyses: list):
    """Render history with modern cards design."""
    if not analyses:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-title">NO RECORDS FOUND</div>
            <div class="empty-text">Your analysis history will appear here</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Inject additional CSS for history cards
    st.markdown("""
    <style>
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .history-card {
            background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
            border: 1px solid #1a1a24;
            padding: 1.2rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .history-card:hover {
            border-color: #2a2a3a;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }
        
        .history-card.threat {
            border-left: 3px solid #ff3366;
        }
        
        .history-card.threat::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle at top right, rgba(255, 51, 102, 0.15), transparent);
        }
        
        .history-card.safe {
            border-left: 3px solid #39FF14;
        }
        
        .history-card.safe::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle at top right, rgba(57, 255, 20, 0.1), transparent);
        }
        
        .history-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.8rem;
        }
        
        .history-verdict-badge {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.6rem;
            font-family: 'Anybody', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 1px;
        }
        
        .history-verdict-badge.threat {
            background: rgba(255, 51, 102, 0.15);
            border: 1px solid rgba(255, 51, 102, 0.3);
            color: #ff3366;
        }
        
        .history-verdict-badge.safe {
            background: rgba(57, 255, 20, 0.15);
            border: 1px solid rgba(57, 255, 20, 0.3);
            color: #39FF14;
        }
        
        .history-verdict-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            animation: pulse-glow 2s infinite;
        }
        
        .history-verdict-dot.threat {
            background: #ff3366;
            box-shadow: 0 0 8px rgba(255, 51, 102, 0.6);
        }
        
        .history-verdict-dot.safe {
            background: #39FF14;
            box-shadow: 0 0 8px rgba(57, 255, 20, 0.6);
        }
        
        .history-date {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #4a5568;
        }
        
        .history-content {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #e2e8f0;
            line-height: 1.5;
            margin: 1rem 0;
            padding: 0.8rem;
            background: rgba(0, 0, 0, 0.4);
            border-left: 2px solid #2a2a3a;
            word-break: break-word;
        }
        
        .history-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 0.8rem;
            border-top: 1px solid #1a1a24;
        }
        
        .history-meta-left {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        
        .history-type-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.5rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 1px;
        }
        
        .history-type-badge.type-sms {
            background: rgba(0, 255, 242, 0.1);
            border: 1px solid rgba(0, 255, 242, 0.3);
            color: #00fff2;
        }
        
        .history-type-badge.type-email {
            background: rgba(255, 170, 0, 0.1);
            border: 1px solid rgba(255, 170, 0, 0.3);
            color: #ffaa00;
        }
        
        .history-model {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #8892a0;
            padding: 0.2rem 0.4rem;
            background: rgba(255, 255, 255, 0.05);
        }
        
        .history-probability {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .history-prob-bar {
            width: 60px;
            height: 4px;
            background: #1a1a24;
            border-radius: 2px;
            overflow: hidden;
        }
        
        .history-prob-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .history-prob-fill.threat {
            background: linear-gradient(90deg, #ff3366, #ff6699);
        }
        
        .history-prob-fill.safe {
            background: linear-gradient(90deg, #39FF14, #66ff66);
        }
        
        .history-prob-text {
            font-family: 'Anybody', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
        }
        
        .history-prob-text.threat {
            color: #ff3366;
        }
        
        .history-prob-text.safe {
            color: #39FF14;
        }
        
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
    """, unsafe_allow_html=True)

    # Render history cards
    cards_html = '<div class="history-grid">'
    
    for a in analyses:
        verdict_class = "threat" if a['prediction'] == 1 else "safe"
        verdict_text = "⚠ THREAT" if a['prediction'] == 1 else "✓ SAFE"
        type_class = f"type-{a['text_type']}"
        type_icon = "📱" if a['text_type'] == 'sms' else "📧"
        prob_width = a['probability'] if a['prediction'] == 1 else (100 - a['probability'])

        # Format date
        try:
            dt = datetime.fromisoformat(a['created_at'])
            date_str = dt.strftime("%b %d, %Y • %H:%M")
        except:
            date_str = a['created_at'][:16] if a['created_at'] else "N/A"

        # Truncate text and escape HTML characters
        text_preview = a['text_input'][:100] + "..." if len(a['text_input']) > 100 else a['text_input']
        text_preview = html.escape(text_preview)
        
        # Format model name
        model_display = a['model_used'].replace('_', ' ').replace('email ', '').title()

        # Build card HTML in single line to avoid Streamlit parsing issues
        card_html = f'<div class="history-card {verdict_class}">'
        card_html += f'<div class="history-card-header">'
        card_html += f'<div class="history-verdict-badge {verdict_class}">'
        card_html += f'<span class="history-verdict-dot {verdict_class}"></span>'
        card_html += f'{verdict_text}</div>'
        card_html += f'<span class="history-date">{date_str}</span></div>'
        card_html += f'<div class="history-content">{text_preview}</div>'
        card_html += f'<div class="history-meta">'
        card_html += f'<div class="history-meta-left">'
        card_html += f'<span class="history-type-badge {type_class}">{type_icon} {a["text_type"].upper()}</span>'
        card_html += f'<span class="history-model">{model_display}</span></div>'
        card_html += f'<div class="history-probability">'
        card_html += f'<div class="history-prob-bar">'
        card_html += f'<div class="history-prob-fill {verdict_class}" style="width: {prob_width}%"></div></div>'
        card_html += f'<span class="history-prob-text {verdict_class}">{a["probability"]:.1f}%</span></div></div></div>'
        
        cards_html += card_html
    
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)



def convert_to_csv(analyses: list) -> str:
    """Convert analyses to CSV string."""
    if not analyses:
        return ""

    df = pd.DataFrame(analyses)
    df = df[['created_at', 'text_type', 'text_input', 'model_used', 'prediction', 'probability']]
    df['prediction'] = df['prediction'].map({0: 'SAFE', 1: 'THREAT'})
    df.columns = ['Date', 'Type', 'Content', 'Model', 'Verdict', 'Probability']

    return df.to_csv(index=False)


def main():
    """Main function for history page."""
    # Inject CSS
    st.markdown(get_history_css(), unsafe_allow_html=True)
    st.markdown(get_auth_css(), unsafe_allow_html=True)

    # Check authentication
    if not render_auth_page():
        return

    user = get_current_user()

    # Header
    st.markdown("""
    <div class="history-header">
        <h1 class="history-title">ANALYSIS HISTORY</h1>
        <p class="history-subtitle">// Threat Detection Archive</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation bar
    nav_cols = st.columns([1, 1, 1, 2, 1, 1])

    with nav_cols[0]:
        if st.button("🔍 Scanner", use_container_width=True, key="nav_scanner"):
            st.switch_page("streamlit_app.py")

    with nav_cols[1]:
        st.markdown('<div class="nav-active">📜 HISTORY</div>', unsafe_allow_html=True)

    with nav_cols[2]:
        # Admin link - only show for admins
        if user and user.get('role') == 'admin':
            if st.button("👑 Admin", use_container_width=True, key="nav_admin"):
                st.switch_page("pages/admin_dashboard.py")

    with nav_cols[4]:
        st.markdown(f"**{user['username']}**")

    with nav_cols[5]:
        render_logout_button()

    # Get user stats
    stats = get_user_stats(user['id'])
    render_stats_cards(stats)

    # Filters
    st.markdown('<div class="filters-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        type_filter = st.selectbox(
            "Message Type",
            ["All", "SMS", "Email"],
            key="type_filter"
        )

    with col2:
        verdict_filter = st.selectbox(
            "Verdict",
            ["All", "Threat", "Safe"],
            key="verdict_filter"
        )

    with col3:
        date_from = st.date_input(
            "From Date",
            value=datetime.now() - timedelta(days=30),
            key="date_from"
        )

    with col4:
        date_to = st.date_input(
            "To Date",
            value=datetime.now(),
            key="date_to"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Get analyses
    text_type = None if type_filter == "All" else type_filter.lower()
    analyses = get_user_analyses(user['id'], limit=100, text_type=text_type)

    # Apply additional filters
    if verdict_filter != "All":
        target_pred = 1 if verdict_filter == "Threat" else 0
        analyses = [a for a in analyses if a['prediction'] == target_pred]

    # Filter by date
    if date_from and date_to:
        analyses = [
            a for a in analyses
            if a['created_at'] and
            date_from <= datetime.fromisoformat(a['created_at']).date() <= date_to
        ]

    # Action buttons row
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_history"):
            st.rerun()
    
    with col3:
        if analyses:
            csv_data = convert_to_csv(analyses)
            st.download_button(
                label="EXPORT CSV",
                data=csv_data,
                file_name=f"shield_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="export_csv"
            )

    # Render table
    render_history_table(analyses)

    # Pagination info
    if analyses:
        st.markdown(f"""
        <div style="text-align: center; margin-top: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4a5568;">
            Showing {len(analyses)} records
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
