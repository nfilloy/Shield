"""
Authentication service for SHIELD application.

Provides secure user authentication functions:
- Password hashing with bcrypt
- User registration
- User authentication (login)
- Session management helpers
"""

import logging
import re
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

import bcrypt
from sqlalchemy.exc import IntegrityError

from src.database import get_db_session, User

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
    """Result of an authentication operation."""
    success: bool
    message: str
    user: Optional[User] = None


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Plain text password to verify
        password_hash: Stored password hash

    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength.

    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"

    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email format.

    Args:
        email: Email to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format"
    return True, ""


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate username format.

    Requirements:
    - 3-50 characters
    - Only alphanumeric and underscores
    - Must start with a letter

    Args:
        username: Username to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"

    if len(username) > 50:
        return False, "Username must be at most 50 characters long"

    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', username):
        return False, "Username must start with a letter and contain only letters, numbers, and underscores"

    return True, ""


def register_user(
    username: str,
    email: str,
    password: str,
    role: str = "user"
) -> AuthResult:
    """
    Register a new user.

    Args:
        username: Unique username
        email: Unique email address
        password: Plain text password (will be hashed)
        role: User role ('user' or 'admin')

    Returns:
        AuthResult with success status and user object if successful
    """
    # Validate inputs
    valid, msg = validate_username(username)
    if not valid:
        return AuthResult(success=False, message=msg)

    valid, msg = validate_email(email)
    if not valid:
        return AuthResult(success=False, message=msg)

    valid, msg = validate_password(password)
    if not valid:
        return AuthResult(success=False, message=msg)

    if role not in ('user', 'admin'):
        return AuthResult(success=False, message="Invalid role. Must be 'user' or 'admin'")

    try:
        with get_db_session() as session:
            # Check if username exists
            existing = session.query(User).filter_by(username=username).first()
            if existing:
                return AuthResult(success=False, message="Username already exists")

            # Check if email exists
            existing = session.query(User).filter_by(email=email).first()
            if existing:
                return AuthResult(success=False, message="Email already registered")

            # Create user
            user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                role=role
            )
            session.add(user)
            session.flush()  # Get the ID

            # Detach user from session to use outside context
            session.refresh(user)
            session.expunge(user)

            logger.info(f"User registered: {username}")
            return AuthResult(
                success=True,
                message="Registration successful",
                user=user
            )

    except IntegrityError as e:
        logger.error(f"Registration integrity error: {e}")
        return AuthResult(success=False, message="Username or email already exists")
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return AuthResult(success=False, message="Registration failed. Please try again.")


def authenticate_user(username_or_email: str, password: str) -> AuthResult:
    """
    Authenticate a user by username/email and password.

    Args:
        username_or_email: Username or email address
        password: Plain text password

    Returns:
        AuthResult with success status and user object if successful
    """
    try:
        with get_db_session() as session:
            # Find user by username or email
            user = session.query(User).filter(
                (User.username == username_or_email) |
                (User.email == username_or_email)
            ).first()

            if not user:
                return AuthResult(success=False, message="Invalid credentials")

            # Verify password
            if not verify_password(password, user.password_hash):
                return AuthResult(success=False, message="Invalid credentials")

            # Update last login
            user.last_login = datetime.utcnow()

            # Detach user from session to use outside context
            session.flush()
            session.refresh(user)
            session.expunge(user)

            logger.info(f"User authenticated: {user.username}")
            return AuthResult(
                success=True,
                message="Login successful",
                user=user
            )

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return AuthResult(success=False, message="Authentication failed. Please try again.")


def get_user_by_id(user_id: int) -> Optional[User]:
    """
    Get a user by their ID.

    Args:
        user_id: User's database ID

    Returns:
        User object if found, None otherwise
    """
    try:
        with get_db_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                # Detach from session to use outside context
                session.expunge(user)
            return user
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        return None


def get_user_by_username(username: str) -> Optional[User]:
    """
    Get a user by their username.

    Args:
        username: User's username

    Returns:
        User object if found, None otherwise
    """
    try:
        with get_db_session() as session:
            user = session.query(User).filter_by(username=username).first()
            if user:
                session.expunge(user)
            return user
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        return None


def update_password(user_id: int, old_password: str, new_password: str) -> AuthResult:
    """
    Update a user's password.

    Args:
        user_id: User's database ID
        old_password: Current password for verification
        new_password: New password to set

    Returns:
        AuthResult with success status
    """
    # Validate new password
    valid, msg = validate_password(new_password)
    if not valid:
        return AuthResult(success=False, message=msg)

    try:
        with get_db_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return AuthResult(success=False, message="User not found")

            # Verify old password
            if not verify_password(old_password, user.password_hash):
                return AuthResult(success=False, message="Current password is incorrect")

            # Update password
            user.password_hash = hash_password(new_password)

            logger.info(f"Password updated for user: {user.username}")
            return AuthResult(success=True, message="Password updated successfully")

    except Exception as e:
        logger.error(f"Password update error: {e}")
        return AuthResult(success=False, message="Failed to update password")


def is_admin(user: User) -> bool:
    """
    Check if a user has admin role.

    Args:
        user: User object to check

    Returns:
        True if user is admin, False otherwise
    """
    return user.role == "admin"


def get_all_users() -> list:
    """
    Get all users (admin function).

    Returns:
        List of all User objects
    """
    try:
        with get_db_session() as session:
            users = session.query(User).all()
            for user in users:
                session.expunge(user)
            return users
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return []


def delete_user(user_id: int) -> AuthResult:
    """
    Delete a user account.

    Args:
        user_id: User's database ID

    Returns:
        AuthResult with success status
    """
    try:
        with get_db_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return AuthResult(success=False, message="User not found")

            username = user.username
            session.delete(user)

            logger.info(f"User deleted: {username}")
            return AuthResult(success=True, message="User deleted successfully")

    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return AuthResult(success=False, message="Failed to delete user")


def create_guest_user(guest_id: str) -> AuthResult:
    """
    Create a temporary guest user.

    Guest users can use the application and their analyses are stored,
    but they have limited access (no history page, no admin).

    Args:
        guest_id: Unique identifier for the guest session

    Returns:
        AuthResult with success status and user object if successful
    """
    try:
        with get_db_session() as session:
            # Create guest user with no password
            username = f"GUEST_{guest_id}"
            
            # Check if this guest already exists (unlikely but possible)
            existing = session.query(User).filter_by(username=username).first()
            if existing:
                # Return existing guest
                session.expunge(existing)
                return AuthResult(
                    success=True,
                    message="Guest session resumed",
                    user=existing
                )

            # Create new guest user
            user = User(
                username=username,
                email=f"guest_{guest_id.lower()}@shield.local",
                password_hash="GUEST_NO_PASSWORD",  # Guests can't login with password
                role="guest"
            )
            session.add(user)
            session.flush()

            # Detach user from session
            session.refresh(user)
            session.expunge(user)

            logger.info(f"Guest user created: {username}")
            return AuthResult(
                success=True,
                message="Guest session started",
                user=user
            )

    except Exception as e:
        logger.error(f"Guest user creation error: {e}")
        return AuthResult(success=False, message="Failed to create guest session")

