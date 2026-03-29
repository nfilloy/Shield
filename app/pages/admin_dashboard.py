"""
Admin Dashboard Page for SHIELD Application.

Displays global statistics, charts, and system metrics.
Only accessible to users with admin role.
Cyber-noir terminal aesthetic.
"""

import streamlit as st
import pandas as pd
import html
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database import (
    get_global_stats,
    get_recent_analyses,
    get_all_analyses_count,
    get_db_session,
    User,
    Analysis
)
from src.auth import get_all_users
from app.components.auth import (
    render_auth_page,
    is_authenticated,
    is_admin,
    get_current_user,
    render_logout_button,
    get_auth_css
)

# Page configuration
st.set_page_config(
    page_title="SHIELD // Admin Dashboard",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def get_admin_css():
    """Return CSS styles for admin dashboard."""
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
            --neon-purple: #bf5fff;
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
            max-width: 1600px;
        }

        /* ══════════════════════════════════════════
           PAGE HEADER
           ══════════════════════════════════════════ */
        .admin-header {
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 1rem;
            position: relative;
        }

        .admin-header::before {
            content: '[ ADMIN ACCESS ]';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            color: #ffaa00;
            letter-spacing: 3px;
        }

        .admin-title {
            font-family: 'Anybody', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(180deg, #ffaa00 0%, #ff6600 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        .admin-subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: #4a5568;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        /* ══════════════════════════════════════════
           MAIN STATS GRID
           ══════════════════════════════════════════ */
        .main-stats-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .main-stat-card {
            background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
            border: 1px solid #1a1a24;
            padding: 1.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .main-stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }

        .main-stat-card.users::before { background: var(--neon-purple); }
        .main-stat-card.analyses::before { background: var(--neon-cyan); }
        .main-stat-card.threats::before { background: var(--neon-red); }
        .main-stat-card.safe::before { background: var(--neon-green); }
        .main-stat-card.rate::before { background: var(--neon-amber); }

        .main-stat-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            opacity: 0.8;
        }

        .main-stat-value {
            font-family: 'Anybody', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
        }

        .main-stat-card.users .main-stat-value { color: #bf5fff; }
        .main-stat-card.analyses .main-stat-value { color: #00fff2; }
        .main-stat-card.threats .main-stat-value { color: #ff3366; }
        .main-stat-card.safe .main-stat-value { color: #39FF14; }
        .main-stat-card.rate .main-stat-value { color: #ffaa00; }

        .main-stat-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #4a5568;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        /* ══════════════════════════════════════════
           SECTION HEADERS
           ══════════════════════════════════════════ */
        .section-title {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #8892a0;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1a1a24;
        }

        .section-title::before {
            content: '// ';
            color: #00fff2;
        }

        /* ══════════════════════════════════════════
           CHART CONTAINERS
           ══════════════════════════════════════════ */
        .chart-container {
            background: #0a0a0f;
            border: 1px solid #1a1a24;
            padding: 1rem;
            margin: 1rem 0;
        }

        .chart-title {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #8892a0;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }

        /* ══════════════════════════════════════════
           RECENT ACTIVITY TABLE
           ══════════════════════════════════════════ */
        .activity-table {
            width: 100%;
            border-collapse: collapse;
        }

        .activity-table th {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #4a5568;
            letter-spacing: 1px;
            text-transform: uppercase;
            text-align: left;
            padding: 0.6rem 0.8rem;
            border-bottom: 1px solid #1a1a24;
            background: #0a0a0f;
        }

        .activity-table td {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #e2e8f0;
            padding: 0.8rem;
            border-bottom: 1px solid #12121a;
        }

        .activity-table tr:hover td {
            background: #0a0a0f;
        }

        .user-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            font-size: 0.6rem;
            background: rgba(191, 95, 255, 0.1);
            border: 1px solid rgba(191, 95, 255, 0.3);
            color: #bf5fff;
        }

        .verdict-threat { color: #ff3366; }
        .verdict-safe { color: #39FF14; }

        .type-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            font-size: 0.6rem;
        }

        .type-sms {
            background: rgba(0, 255, 242, 0.1);
            border: 1px solid rgba(0, 255, 242, 0.3);
            color: #00fff2;
        }

        .type-email {
            background: rgba(255, 170, 0, 0.1);
            border: 1px solid rgba(255, 170, 0, 0.3);
            color: #ffaa00;
        }

        /* ══════════════════════════════════════════
           ACCESS DENIED
           ══════════════════════════════════════════ */
        .access-denied {
            text-align: center;
            padding: 4rem 2rem;
            background: rgba(255, 51, 102, 0.05);
            border: 1px solid rgba(255, 51, 102, 0.2);
            margin: 2rem auto;
            max-width: 500px;
        }

        .access-denied-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }

        .access-denied-title {
            font-family: 'Anybody', sans-serif;
            font-size: 1.5rem;
            color: #ff3366;
            margin-bottom: 0.5rem;
        }

        .access-denied-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #8892a0;
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
        }

        .stButton > button:hover {
            border-color: #ffaa00 !important;
            color: #ffaa00 !important;
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
            color: #ffaa00 !important;
            border-color: #ffaa00 !important;
            background: rgba(255, 170, 0, 0.05) !important;
        }

        [data-testid="stPageLink"] a[aria-current="page"] {
            color: #ffaa00 !important;
            border-color: #ffaa00 !important;
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


def get_plotly_theme():
    """Return Plotly theme configuration for cyber-noir aesthetic."""
    return {
        'paper_bgcolor': '#0a0a0f',
        'plot_bgcolor': '#0a0a0f',
        'font': {
            'family': 'IBM Plex Mono, monospace',
            'color': '#8892a0',
            'size': 11
        },
        'xaxis': {
            'gridcolor': '#1a1a24',
            'linecolor': '#1a1a24',
            'tickfont': {'color': '#4a5568'}
        },
        'yaxis': {
            'gridcolor': '#1a1a24',
            'linecolor': '#1a1a24',
            'tickfont': {'color': '#4a5568'}
        },
        'colorway': ['#00fff2', '#39FF14', '#ff3366', '#ffaa00', '#bf5fff']
    }


def render_main_stats(stats: dict):
    """Render main statistics cards."""
    st.markdown(f"""
    <div class="main-stats-grid">
        <div class="main-stat-card users">
            <div class="main-stat-icon">👥</div>
            <div class="main-stat-value">{stats['total_users']}</div>
            <div class="main-stat-label">Total Users</div>
        </div>
        <div class="main-stat-card analyses">
            <div class="main-stat-icon">🔍</div>
            <div class="main-stat-value">{stats['total_analyses']}</div>
            <div class="main-stat-label">Total Analyses</div>
        </div>
        <div class="main-stat-card threats">
            <div class="main-stat-icon">⚠️</div>
            <div class="main-stat-value">{stats['total_threats']}</div>
            <div class="main-stat-label">Threats Detected</div>
        </div>
        <div class="main-stat-card safe">
            <div class="main-stat-icon">✓</div>
            <div class="main-stat-value">{stats['total_safe']}</div>
            <div class="main-stat-label">Safe Messages</div>
        </div>
        <div class="main-stat-card rate">
            <div class="main-stat-icon">📊</div>
            <div class="main-stat-value">{stats['threat_rate']:.1f}%</div>
            <div class="main-stat-label">Threat Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_analyses_by_date():
    """Get analysis counts grouped by date for the last 30 days."""
    try:
        with get_db_session() as session:
            analyses = session.query(Analysis).filter(
                Analysis.created_at >= datetime.now() - timedelta(days=30)
            ).all()

            # Group by date
            date_counts = {}
            for a in analyses:
                if a.created_at:
                    date_key = a.created_at.date()
                    if date_key not in date_counts:
                        date_counts[date_key] = {'threats': 0, 'safe': 0}
                    if a.prediction == 1:
                        date_counts[date_key]['threats'] += 1
                    else:
                        date_counts[date_key]['safe'] += 1

            # Create DataFrame
            dates = []
            threats = []
            safe = []
            for date in sorted(date_counts.keys()):
                dates.append(date)
                threats.append(date_counts[date]['threats'])
                safe.append(date_counts[date]['safe'])

            return pd.DataFrame({
                'Date': dates,
                'Threats': threats,
                'Safe': safe
            })
    except Exception:
        return pd.DataFrame()


def get_type_distribution():
    """Get analysis distribution by type."""
    try:
        with get_db_session() as session:
            analyses = session.query(Analysis).all()

            sms_threats = sum(1 for a in analyses if a.text_type == 'sms' and a.prediction == 1)
            sms_safe = sum(1 for a in analyses if a.text_type == 'sms' and a.prediction == 0)
            email_threats = sum(1 for a in analyses if a.text_type == 'email' and a.prediction == 1)
            email_safe = sum(1 for a in analyses if a.text_type == 'email' and a.prediction == 0)

            return pd.DataFrame({
                'Type': ['SMS Threats', 'SMS Safe', 'Email Threats', 'Email Safe'],
                'Count': [sms_threats, sms_safe, email_threats, email_safe],
                'Category': ['SMS', 'SMS', 'Email', 'Email'],
                'Status': ['Threat', 'Safe', 'Threat', 'Safe']
            })
    except Exception:
        return pd.DataFrame()


def get_model_usage():
    """Get analysis count by model."""
    try:
        with get_db_session() as session:
            analyses = session.query(Analysis).all()

            model_counts = {}
            for a in analyses:
                model = a.model_used
                if model not in model_counts:
                    model_counts[model] = 0
                model_counts[model] += 1

            return pd.DataFrame({
                'Model': list(model_counts.keys()),
                'Count': list(model_counts.values())
            })
    except Exception:
        return pd.DataFrame()


def render_activity_table(recent: list):
    """Render recent activity with modern cards design."""
    if not recent:
        st.markdown("""
        <div class="empty-activity">
            <div class="empty-icon">📭</div>
            <div class="empty-text">No recent activity detected</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Inject additional CSS for activity cards
    st.markdown("""
    <style>
        .activity-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .activity-card {
            background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
            border: 1px solid #1a1a24;
            padding: 1rem 1.2rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .activity-card:hover {
            border-color: #2a2a3a;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        
        .activity-card.threat {
            border-left: 3px solid #ff3366;
        }
        
        .activity-card.safe {
            border-left: 3px solid #39FF14;
        }
        
        .activity-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        
        .activity-user {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .activity-user-icon {
            width: 28px;
            height: 28px;
            background: rgba(191, 95, 255, 0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
        }
        
        .activity-user-name {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
            color: #bf5fff;
        }
        
        .activity-time {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #4a5568;
        }
        
        .activity-content {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #8892a0;
            line-height: 1.4;
            margin: 0.8rem 0;
            padding: 0.6rem;
            background: rgba(0, 0, 0, 0.3);
            border-left: 2px solid #2a2a3a;
        }
        
        .activity-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.8rem;
            padding-top: 0.8rem;
            border-top: 1px solid #1a1a24;
        }
        
        .activity-type-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.5rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        
        .activity-type-badge.sms {
            background: rgba(0, 255, 242, 0.1);
            border: 1px solid rgba(0, 255, 242, 0.3);
            color: #00fff2;
        }
        
        .activity-type-badge.email {
            background: rgba(255, 170, 0, 0.1);
            border: 1px solid rgba(255, 170, 0, 0.3);
            color: #ffaa00;
        }
        
        .activity-verdict {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .activity-verdict-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse-glow 2s infinite;
        }
        
        .activity-verdict-indicator.threat {
            background: #ff3366;
            box-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
        }
        
        .activity-verdict-indicator.safe {
            background: #39FF14;
            box-shadow: 0 0 10px rgba(57, 255, 20, 0.5);
        }
        
        .activity-verdict-text {
            font-family: 'Anybody', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
        }
        
        .activity-verdict-text.threat {
            color: #ff3366;
        }
        
        .activity-verdict-text.safe {
            color: #39FF14;
        }
        
        .activity-probability {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: #8892a0;
            padding: 0.2rem 0.5rem;
            background: rgba(255, 255, 255, 0.05);
        }
        
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .empty-activity {
            text-align: center;
            padding: 3rem;
            background: #0a0a0f;
            border: 1px dashed #1a1a24;
        }
        
        .empty-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            opacity: 0.5;
        }
        
        .empty-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #4a5568;
        }
    </style>
    """, unsafe_allow_html=True)

    # Render activity cards
    cards_html = '<div class="activity-grid">'
    
    for a in recent:
        verdict_class = "threat" if a['prediction'] == 1 else "safe"
        verdict_text = "THREAT DETECTED" if a['prediction'] == 1 else "SAFE"
        type_class = a['text_type']
        type_icon = "📱" if a['text_type'] == 'sms' else "📧"
        user_initial = a['username'][0].upper() if a['username'] else "?"
        
        # Escape HTML in text preview
        text_preview = html.escape(a['text_preview'])

        try:
            dt = datetime.fromisoformat(a['created_at'])
            date_str = dt.strftime("%b %d, %H:%M")
        except:
            date_str = "N/A"

        # Build card HTML in single lines to avoid Streamlit parsing issues
        card_html = f'<div class="activity-card {verdict_class}">'
        card_html += f'<div class="activity-card-header">'
        card_html += f'<div class="activity-user">'
        card_html += f'<div class="activity-user-icon">{user_initial}</div>'
        card_html += f'<span class="activity-user-name">{a["username"]}</span></div>'
        card_html += f'<span class="activity-time">{date_str}</span></div>'
        card_html += f'<div class="activity-content">"{text_preview}"</div>'
        card_html += f'<div class="activity-footer">'
        card_html += f'<span class="activity-type-badge {type_class}">{type_icon} {a["text_type"].upper()}</span>'
        card_html += f'<div class="activity-verdict">'
        card_html += f'<div class="activity-verdict-indicator {verdict_class}"></div>'
        card_html += f'<span class="activity-verdict-text {verdict_class}">{verdict_text}</span></div>'
        card_html += f'<span class="activity-probability">{a["probability"]:.1f}%</span></div></div>'
        
        cards_html += card_html
    
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


def main():
    """Main function for admin dashboard."""
    # Inject CSS
    st.markdown(get_admin_css(), unsafe_allow_html=True)
    st.markdown(get_auth_css(), unsafe_allow_html=True)

    # Check authentication
    if not render_auth_page():
        return

    # Check admin role
    if not is_admin():
        st.markdown("""
        <div class="access-denied">
            <div class="access-denied-icon">🚫</div>
            <div class="access-denied-title">ACCESS DENIED</div>
            <div class="access-denied-text">Administrator privileges required to access this dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        return

    user = get_current_user()

    # Header
    st.markdown("""
    <div class="admin-header">
        <h1 class="admin-title">CONTROL CENTER</h1>
        <p class="admin-subtitle">// System Administration Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation bar
    nav_cols = st.columns([1, 1, 1, 2, 1, 1])

    with nav_cols[0]:
        if st.button("🔍 Scanner", use_container_width=True, key="nav_scanner"):
            st.switch_page("streamlit_app.py")

    with nav_cols[1]:
        if st.button("📜 History", use_container_width=True, key="nav_history"):
            st.switch_page("pages/history.py")

    with nav_cols[2]:
        st.markdown('<div class="nav-active">👑 ADMIN</div>', unsafe_allow_html=True)

    with nav_cols[4]:
        st.markdown(f"**👑 {user['username']}**")

    with nav_cols[5]:
        render_logout_button()

    # Refresh button
    col_refresh = st.columns([5, 1])
    with col_refresh[1]:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_admin"):
            st.rerun()

    # Get global stats
    stats = get_global_stats()
    render_main_stats(stats)

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Activity Timeline</div>', unsafe_allow_html=True)

        df_dates = get_analyses_by_date()
        if not df_dates.empty:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=df_dates['Date'],
                y=df_dates['Safe'],
                name='Safe',
                marker_color='#39FF14'
            ))

            fig.add_trace(go.Bar(
                x=df_dates['Date'],
                y=df_dates['Threats'],
                name='Threats',
                marker_color='#ff3366'
            ))

            theme = get_plotly_theme()
            fig.update_layout(
                barmode='stack',
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=theme['xaxis'],
                yaxis=theme['yaxis'],
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    font=dict(size=10)
                ),
                margin=dict(l=40, r=20, t=40, b=40),
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for timeline")

    with col2:
        st.markdown('<div class="section-title">Type Distribution</div>', unsafe_allow_html=True)

        df_types = get_type_distribution()
        if not df_types.empty and df_types['Count'].sum() > 0:
            colors = {
                'SMS Threats': '#ff3366',
                'SMS Safe': '#00fff2',
                'Email Threats': '#ff6699',
                'Email Safe': '#ffaa00'
            }

            fig = px.pie(
                df_types,
                values='Count',
                names='Type',
                color='Type',
                color_discrete_map=colors,
                hole=0.5
            )

            theme = get_plotly_theme()
            fig.update_layout(
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.2,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=10)
                ),
                margin=dict(l=20, r=20, t=20, b=60),
                height=300
            )

            fig.update_traces(textposition='inside', textinfo='percent')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for distribution")

    # Model Usage Chart
    st.markdown('<div class="section-title">Model Usage</div>', unsafe_allow_html=True)

    df_models = get_model_usage()
    if not df_models.empty:
        fig = px.bar(
            df_models,
            x='Model',
            y='Count',
            color='Count',
            color_continuous_scale=['#1a1a24', '#00fff2']
        )

        theme = get_plotly_theme()
        fig.update_layout(
            paper_bgcolor=theme['paper_bgcolor'],
            plot_bgcolor=theme['plot_bgcolor'],
            font=theme['font'],
            xaxis={**theme['xaxis'], 'title': ''},
            yaxis={**theme['yaxis'], 'title': 'Analyses'},
            coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=20, b=80),
            height=250
        )

        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model usage data")

    # Recent Activity
    st.markdown('<div class="section-title">Recent Activity</div>', unsafe_allow_html=True)

    recent = get_recent_analyses(limit=15)
    render_activity_table(recent)

    # Footer stats
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #1a1a24;">
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4a5568;">
            SHIELD ADMIN DASHBOARD // Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
