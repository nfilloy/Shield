"""
Authentication module for SHIELD application.

Provides secure user authentication and authorization:
- Password hashing with bcrypt
- User registration and login
- Role-based access control
- Guest user support

Usage:
    from src.auth import register_user, authenticate_user, AuthResult

    # Register new user
    result = register_user("username", "email@example.com", "Password123")
    if result.success:
        print(f"Welcome, {result.user.username}!")

    # Login
    result = authenticate_user("username", "Password123")
    if result.success:
        print(f"Logged in as {result.user.username}")

    # Guest access
    result = create_guest_user("ABC123")
    if result.success:
        print(f"Guest session: {result.user.username}")
"""

from .auth_service import (
    # Result class
    AuthResult,
    # Password functions
    hash_password,
    verify_password,
    validate_password,
    # Validation functions
    validate_email,
    validate_username,
    # Core auth functions
    register_user,
    authenticate_user,
    update_password,
    create_guest_user,
    # User retrieval
    get_user_by_id,
    get_user_by_username,
    get_all_users,
    # User management
    delete_user,
    is_admin,
)

__all__ = [
    # Result class
    "AuthResult",
    # Password functions
    "hash_password",
    "verify_password",
    "validate_password",
    # Validation functions
    "validate_email",
    "validate_username",
    # Core auth functions
    "register_user",
    "authenticate_user",
    "update_password",
    "create_guest_user",
    # User retrieval
    "get_user_by_id",
    "get_user_by_username",
    "get_all_users",
    # User management
    "delete_user",
    "is_admin",
]

