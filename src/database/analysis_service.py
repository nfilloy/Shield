"""
Analysis service for SHIELD application.

Provides functions to save and retrieve analysis history.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from .connection import get_db_session
from .models import Analysis, User

logger = logging.getLogger(__name__)


def save_analysis(
    text_input: str,
    text_type: str,
    model_used: str,
    prediction: int,
    probability: float,
    features: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None
) -> Optional[int]:
    """
    Save an analysis result to the database.

    Args:
        text_input: Original text that was analyzed
        text_type: Type of content ('sms' or 'email')
        model_used: Name of the ML model used
        prediction: Binary prediction (0=safe, 1=threat)
        probability: Threat probability (0-100)
        features: Optional dict of extracted features
        user_id: Optional user ID (None for anonymous)

    Returns:
        Analysis ID if successful, None otherwise
    """
    try:
        with get_db_session() as session:
            # Convert features dict to JSON string
            features_json = json.dumps(features) if features else None

            analysis = Analysis(
                user_id=user_id,
                text_input=text_input,
                text_type=text_type,
                model_used=model_used,
                prediction=prediction,
                probability=probability,
                features_json=features_json
            )
            session.add(analysis)
            session.flush()

            analysis_id = analysis.id
            logger.info(f"Analysis saved: id={analysis_id}, type={text_type}, prediction={prediction}")
            return analysis_id

    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")
        return None


def get_user_analyses(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
    text_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get analysis history for a specific user.

    Args:
        user_id: User's database ID
        limit: Maximum number of results
        offset: Number of results to skip
        text_type: Optional filter by type ('sms' or 'email')

    Returns:
        List of analysis dictionaries
    """
    try:
        with get_db_session() as session:
            query = session.query(Analysis).filter(Analysis.user_id == user_id)

            if text_type:
                query = query.filter(Analysis.text_type == text_type)

            query = query.order_by(Analysis.created_at.desc())
            query = query.limit(limit).offset(offset)

            analyses = query.all()

            return [
                {
                    'id': a.id,
                    'text_input': a.text_input,
                    'text_type': a.text_type,
                    'model_used': a.model_used,
                    'prediction': a.prediction,
                    'probability': a.probability,
                    'features': json.loads(a.features_json) if a.features_json else None,
                    'created_at': a.created_at.isoformat() if a.created_at else None
                }
                for a in analyses
            ]

    except Exception as e:
        logger.error(f"Failed to get user analyses: {e}")
        return []


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific analysis by ID.

    Args:
        analysis_id: Analysis database ID

    Returns:
        Analysis dictionary if found, None otherwise
    """
    try:
        with get_db_session() as session:
            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()

            if not analysis:
                return None

            return {
                'id': analysis.id,
                'user_id': analysis.user_id,
                'text_input': analysis.text_input,
                'text_type': analysis.text_type,
                'model_used': analysis.model_used,
                'prediction': analysis.prediction,
                'probability': analysis.probability,
                'features': json.loads(analysis.features_json) if analysis.features_json else None,
                'created_at': analysis.created_at.isoformat() if analysis.created_at else None
            }

    except Exception as e:
        logger.error(f"Failed to get analysis: {e}")
        return None


def get_user_analysis_count(user_id: int) -> int:
    """
    Get total number of analyses for a user.

    Args:
        user_id: User's database ID

    Returns:
        Total count of analyses
    """
    try:
        with get_db_session() as session:
            return session.query(Analysis).filter(Analysis.user_id == user_id).count()
    except Exception as e:
        logger.error(f"Failed to count analyses: {e}")
        return 0


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """
    Get analysis statistics for a user.

    Args:
        user_id: User's database ID

    Returns:
        Dictionary with statistics
    """
    try:
        with get_db_session() as session:
            analyses = session.query(Analysis).filter(Analysis.user_id == user_id).all()

            if not analyses:
                return {
                    'total': 0,
                    'threats': 0,
                    'safe': 0,
                    'sms_count': 0,
                    'email_count': 0,
                    'threat_rate': 0.0
                }

            total = len(analyses)
            threats = sum(1 for a in analyses if a.prediction == 1)
            sms_count = sum(1 for a in analyses if a.text_type == 'sms')
            email_count = sum(1 for a in analyses if a.text_type == 'email')

            return {
                'total': total,
                'threats': threats,
                'safe': total - threats,
                'sms_count': sms_count,
                'email_count': email_count,
                'threat_rate': (threats / total * 100) if total > 0 else 0.0
            }

    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        return {'total': 0, 'threats': 0, 'safe': 0, 'sms_count': 0, 'email_count': 0, 'threat_rate': 0.0}


def delete_analysis(analysis_id: int, user_id: int) -> bool:
    """
    Delete an analysis (only if owned by user).

    Args:
        analysis_id: Analysis database ID
        user_id: User's database ID (for ownership check)

    Returns:
        True if deleted, False otherwise
    """
    try:
        with get_db_session() as session:
            analysis = session.query(Analysis).filter(
                Analysis.id == analysis_id,
                Analysis.user_id == user_id
            ).first()

            if not analysis:
                return False

            session.delete(analysis)
            logger.info(f"Analysis deleted: id={analysis_id}")
            return True

    except Exception as e:
        logger.error(f"Failed to delete analysis: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# ADMIN FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_all_analyses_count() -> int:
    """Get total number of analyses in the system."""
    try:
        with get_db_session() as session:
            return session.query(Analysis).count()
    except Exception as e:
        logger.error(f"Failed to count all analyses: {e}")
        return 0


def get_global_stats() -> Dict[str, Any]:
    """
    Get global analysis statistics (admin function).

    Returns:
        Dictionary with global statistics
    """
    try:
        with get_db_session() as session:
            analyses = session.query(Analysis).all()
            users = session.query(User).count()

            if not analyses:
                return {
                    'total_analyses': 0,
                    'total_users': users,
                    'total_threats': 0,
                    'total_safe': 0,
                    'sms_count': 0,
                    'email_count': 0,
                    'threat_rate': 0.0,
                    'avg_probability': 0.0
                }

            total = len(analyses)
            threats = sum(1 for a in analyses if a.prediction == 1)
            sms_count = sum(1 for a in analyses if a.text_type == 'sms')
            email_count = sum(1 for a in analyses if a.text_type == 'email')
            avg_prob = sum(a.probability for a in analyses) / total

            return {
                'total_analyses': total,
                'total_users': users,
                'total_threats': threats,
                'total_safe': total - threats,
                'sms_count': sms_count,
                'email_count': email_count,
                'threat_rate': (threats / total * 100) if total > 0 else 0.0,
                'avg_probability': avg_prob
            }

    except Exception as e:
        logger.error(f"Failed to get global stats: {e}")
        return {
            'total_analyses': 0, 'total_users': 0, 'total_threats': 0,
            'total_safe': 0, 'sms_count': 0, 'email_count': 0,
            'threat_rate': 0.0, 'avg_probability': 0.0
        }


def get_recent_analyses(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get most recent analyses (admin function).

    Args:
        limit: Maximum number of results

    Returns:
        List of recent analysis dictionaries with user info
    """
    try:
        with get_db_session() as session:
            analyses = session.query(Analysis).order_by(
                Analysis.created_at.desc()
            ).limit(limit).all()

            results = []
            for a in analyses:
                user = session.query(User).filter(User.id == a.user_id).first() if a.user_id else None
                results.append({
                    'id': a.id,
                    'username': user.username if user else 'Anonymous',
                    'text_preview': a.text_input[:50] + '...' if len(a.text_input) > 50 else a.text_input,
                    'text_type': a.text_type,
                    'model_used': a.model_used,
                    'prediction': a.prediction,
                    'probability': a.probability,
                    'created_at': a.created_at.isoformat() if a.created_at else None
                })

            return results

    except Exception as e:
        logger.error(f"Failed to get recent analyses: {e}")
        return []
