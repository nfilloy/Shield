"""
Authentication UI Components for SHIELD Application.

Cyber-noir styled login and registration forms integrated with Streamlit.
"""

import streamlit as st
import uuid
from typing import Optional, Callable
from src.auth import (
    register_user,
    authenticate_user,
    create_guest_user,
    AuthResult
)


def get_auth_css() -> str:
    """Return CSS styles for authentication components."""
    return """
    <style>
        /* ══════════════════════════════════════════
           AUTH CONTAINER - Centered Panel
           ══════════════════════════════════════════ */
        .auth-container {
            max-width: 420px;
            margin: 2rem auto;
            padding: 2.5rem;
            background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
            border: 1px solid #1a1a24;
            position: relative;
        }

        .auth-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00fff2, #39FF14);
            box-shadow: 0 0 20px rgba(0, 255, 242, 0.5);
        }

        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .auth-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            filter: drop-shadow(0 0 20px rgba(0, 255, 242, 0.5));
        }

        .auth-title {
            font-family: 'Anybody', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(180deg, #00fff2 0%, #39FF14 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
        }

        .auth-subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: #4a5568;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        /* ══════════════════════════════════════════
           FORM INPUTS - Terminal Style
           ══════════════════════════════════════════ */
        .auth-container .stTextInput > div > div > input {
            background: #050508 !important;
            border: 1px solid #1a1a24 !important;
            border-radius: 0 !important;
            color: #e2e8f0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.9rem !important;
            padding: 0.8rem 1rem !important;
            transition: all 0.2s ease !important;
        }

        .auth-container .stTextInput > div > div > input:focus {
            border-color: #00fff2 !important;
            box-shadow: 0 0 0 1px #00fff2, 0 0 20px rgba(0, 255, 242, 0.2) !important;
        }

        .auth-container .stTextInput > div > div > input::placeholder {
            color: #4a5568 !important;
        }

        .auth-container .stTextInput > label {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem !important;
            color: #8892a0 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
        }

        /* ══════════════════════════════════════════
           AUTH BUTTONS
           ══════════════════════════════════════════ */
        .auth-container .stButton > button {
            width: 100% !important;
            background: linear-gradient(90deg, #39FF14, #00fff2) !important;
            border: none !important;
            border-radius: 0 !important;
            color: #050508 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 0.9rem 1.5rem !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 0 20px rgba(57, 255, 20, 0.3) !important;
        }

        .auth-container .stButton > button:hover {
            box-shadow: 0 0 30px rgba(57, 255, 20, 0.5),
                        0 0 60px rgba(0, 255, 242, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        /* ══════════════════════════════════════════
           STATUS MESSAGES
           ══════════════════════════════════════════ */
        .auth-error {
            background: rgba(255, 51, 102, 0.1);
            border: 1px solid rgba(255, 51, 102, 0.3);
            border-left: 3px solid #ff3366;
            padding: 0.8rem 1rem;
            margin: 1rem 0;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #ff3366;
        }

        .auth-success {
            background: rgba(57, 255, 20, 0.1);
            border: 1px solid rgba(57, 255, 20, 0.3);
            border-left: 3px solid #39FF14;
            padding: 0.8rem 1rem;
            margin: 1rem 0;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #39FF14;
        }

        /* ══════════════════════════════════════════
           TOGGLE LINK
           ══════════════════════════════════════════ */
        .auth-toggle {
            text-align: center;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid #1a1a24;
        }

        .auth-toggle-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #4a5568;
        }

        .auth-toggle-link {
            color: #00fff2;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s ease;
        }

        .auth-toggle-link:hover {
            color: #39FF14;
            text-shadow: 0 0 10px rgba(0, 255, 242, 0.5);
        }

        /* ══════════════════════════════════════════
           USER BADGE (When logged in)
           ══════════════════════════════════════════ */
        .user-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.8rem;
            background: rgba(0, 255, 242, 0.1);
            border: 1px solid rgba(0, 255, 242, 0.3);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #00fff2;
        }

        .user-badge.admin {
            background: rgba(255, 170, 0, 0.1);
            border-color: rgba(255, 170, 0, 0.3);
            color: #ffaa00;
        }

        .user-badge.guest {
            background: rgba(138, 138, 138, 0.1);
            border-color: rgba(138, 138, 138, 0.3);
            color: #8a8a8a;
        }

        .user-badge-icon {
            font-size: 0.9rem;
        }

        /* ══════════════════════════════════════════
           PASSWORD REQUIREMENTS
           ══════════════════════════════════════════ */
        .password-requirements {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: #4a5568;
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: rgba(0, 0, 0, 0.3);
            border-left: 2px solid #1a1a24;
        }

        .password-requirements li {
            margin: 0.2rem 0;
        }

        .req-met {
            color: #39FF14;
        }

        .req-unmet {
            color: #4a5568;
        }
    </style>
    """


def init_auth_state():
    """Initialize authentication-related session state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'  # 'login' or 'register'
    if 'auth_message' not in st.session_state:
        st.session_state.auth_message = None
    if 'auth_message_type' not in st.session_state:
        st.session_state.auth_message_type = None


def set_auth_message(message: str, msg_type: str = 'error'):
    """Set authentication status message."""
    st.session_state.auth_message = message
    st.session_state.auth_message_type = msg_type


def clear_auth_message():
    """Clear authentication status message."""
    st.session_state.auth_message = None
    st.session_state.auth_message_type = None


def login_user(user):
    """Set user as logged in."""
    st.session_state.authenticated = True
    st.session_state.user = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role
    }
    clear_auth_message()


def logout_user():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.user = None
    clear_auth_message()


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)


def get_current_user() -> Optional[dict]:
    """Get current logged in user."""
    return st.session_state.get('user', None)


def is_admin() -> bool:
    """Check if current user is admin."""
    user = get_current_user()
    return user is not None and user.get('role') == 'admin'


def is_guest() -> bool:
    """Check if current user is a guest."""
    user = get_current_user()
    return user is not None and user.get('role') == 'guest'


def login_as_guest():
    """Create a guest session with a temporary user."""
    # Generate unique guest ID
    guest_id = str(uuid.uuid4())[:8].upper()
    
    # Create guest user in database
    result = create_guest_user(guest_id)
    
    if result.success:
        st.session_state.authenticated = True
        st.session_state.user = {
            'id': result.user.id,
            'username': f"GUEST_{guest_id}",
            'email': None,
            'role': 'guest'
        }
        clear_auth_message()
        return True
    return False


def render_login_form():
    """Render the login form."""
    st.markdown("""
        <div class="auth-header">
            <div class="auth-icon">🔐</div>
            <h2 class="auth-title">ACCESS TERMINAL</h2>
            <p class="auth-subtitle">// Authentication Required</p>
        </div>
    """, unsafe_allow_html=True)

    # Show messages
    if st.session_state.auth_message:
        msg_class = 'auth-success' if st.session_state.auth_message_type == 'success' else 'auth-error'
        st.markdown(f'<div class="{msg_class}">{st.session_state.auth_message}</div>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username or Email", placeholder="Enter credentials...")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        submit = st.form_submit_button("INITIALIZE SESSION", use_container_width=True)

        if submit:
            if not username or not password:
                set_auth_message("All fields are required")
                st.rerun()
            else:
                result = authenticate_user(username, password)
                if result.success:
                    login_user(result.user)
                    set_auth_message(f"Welcome back, {result.user.username}", "success")
                    st.rerun()
                else:
                    set_auth_message(result.message)
                    st.rerun()

    # Toggle to register
    st.markdown("""
        <div class="auth-toggle">
            <span class="auth-toggle-text">No credentials? </span>
        </div>
    """, unsafe_allow_html=True)

    if st.button("CREATE NEW IDENTITY", key="goto_register", use_container_width=True):
        st.session_state.auth_mode = 'register'
        clear_auth_message()
        st.rerun()

    # Guest access
    st.markdown("""
        <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #1a1a24;">
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4a5568;">Or continue as guest</span>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("👤 CONTINUE AS GUEST", key="guest_login", use_container_width=True):
        if login_as_guest():
            set_auth_message("Guest session started", "success")
            st.rerun()
        else:
            set_auth_message("Failed to create guest session")
            st.rerun()


def render_register_form():
    """Render the registration form."""
    st.markdown("""
        <div class="auth-header">
            <div class="auth-icon">🛡️</div>
            <h2 class="auth-title">NEW IDENTITY</h2>
            <p class="auth-subtitle">// Create Access Credentials</p>
        </div>
    """, unsafe_allow_html=True)

    # Show messages
    if st.session_state.auth_message:
        msg_class = 'auth-success' if st.session_state.auth_message_type == 'success' else 'auth-error'
        st.markdown(f'<div class="{msg_class}">{st.session_state.auth_message}</div>', unsafe_allow_html=True)

    with st.form("register_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Choose identifier...")
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        password_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")

        # Password requirements hint
        st.markdown("""
            <div class="password-requirements">
                <strong>// PASSWORD REQUIREMENTS:</strong>
                <ul>
                    <li>Minimum 8 characters</li>
                    <li>At least one uppercase letter</li>
                    <li>At least one lowercase letter</li>
                    <li>At least one digit</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        submit = st.form_submit_button("GENERATE IDENTITY", use_container_width=True)

        if submit:
            if not username or not email or not password or not password_confirm:
                set_auth_message("All fields are required")
                st.rerun()
            elif password != password_confirm:
                set_auth_message("Passwords do not match")
                st.rerun()
            else:
                result = register_user(username, email, password)
                if result.success:
                    set_auth_message("Identity created. You can now login.", "success")
                    st.session_state.auth_mode = 'login'
                    st.rerun()
                else:
                    set_auth_message(result.message)
                    st.rerun()

    # Toggle to login
    st.markdown("""
        <div class="auth-toggle">
            <span class="auth-toggle-text">Already registered? </span>
        </div>
    """, unsafe_allow_html=True)

    if st.button("ACCESS TERMINAL", key="goto_login", use_container_width=True):
        st.session_state.auth_mode = 'login'
        clear_auth_message()
        st.rerun()

    # Guest access
    st.markdown("""
        <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #1a1a24;">
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4a5568;">Or continue as guest</span>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("👤 CONTINUE AS GUEST", key="guest_login_reg", use_container_width=True):
        if login_as_guest():
            set_auth_message("Guest session started", "success")
            st.rerun()
        else:
            set_auth_message("Failed to create guest session")
            st.rerun()


def render_user_menu():
    """Render user menu when logged in (for sidebar or header)."""
    user = get_current_user()
    if not user:
        return

    # Determine role styling
    if user['role'] == 'admin':
        role_class = "admin"
        role_icon = "👑"
    elif user['role'] == 'guest':
        role_class = "guest"
        role_icon = "👤"
    else:
        role_class = ""
        role_icon = "👤"

    st.markdown(f"""
        <div class="user-badge {role_class}">
            <span class="user-badge-icon">{role_icon}</span>
            <span>{user['username']}</span>
        </div>
    """, unsafe_allow_html=True)


def render_auth_page():
    """
    Render the full authentication page.

    Returns:
        bool: True if user is authenticated, False otherwise
    """
    init_auth_state()

    # If already authenticated, return True
    if is_authenticated():
        return True

    # Inject auth CSS
    st.markdown(get_auth_css(), unsafe_allow_html=True)

    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)

        if st.session_state.auth_mode == 'login':
            render_login_form()
        else:
            render_register_form()

        st.markdown('</div>', unsafe_allow_html=True)

    return False


def render_logout_button():
    """Render a logout button."""
    if st.button("🚪 TERMINATE SESSION", key="logout_btn"):
        logout_user()
        st.rerun()


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for a function.

    Usage:
        @require_auth
        def my_protected_page():
            st.write("Secret content")
    """
    def wrapper(*args, **kwargs):
        if not render_auth_page():
            st.stop()
        return func(*args, **kwargs)
    return wrapper


def require_admin(func: Callable) -> Callable:
    """
    Decorator to require admin role for a function.

    Usage:
        @require_admin
        def admin_dashboard():
            st.write("Admin only content")
    """
    def wrapper(*args, **kwargs):
        if not render_auth_page():
            st.stop()
        if not is_admin():
            st.error("🚫 ACCESS DENIED: Admin privileges required")
            st.stop()
        return func(*args, **kwargs)
    return wrapper
