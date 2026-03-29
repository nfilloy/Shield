"""
Componentes reutilizables para la aplicación Streamlit.
"""

from .xai_display import render_xai_section, render_word_importance_chart
from .auth import (
    render_auth_page,
    render_logout_button,
    render_user_menu,
    init_auth_state,
    is_authenticated,
    is_admin,
    is_guest,
    get_current_user,
    login_user,
    logout_user,
    login_as_guest,
    require_auth,
    require_admin,
    get_auth_css
)

__all__ = [
    # XAI components
    'render_xai_section',
    'render_word_importance_chart',
    # Auth components
    'render_auth_page',
    'render_logout_button',
    'render_user_menu',
    'init_auth_state',
    'is_authenticated',
    'is_admin',
    'is_guest',
    'get_current_user',
    'login_user',
    'logout_user',
    'login_as_guest',
    'require_auth',
    'require_admin',
    'get_auth_css'
]

