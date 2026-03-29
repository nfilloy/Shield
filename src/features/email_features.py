"""
Email-specific feature extraction module.

Extracts metadata and structural features specific to emails
that are useful for phishing detection.
"""

import re
import logging
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailFeatureExtractor:
    """
    Extract email-specific features for phishing detection.

    Features include:
    - Header analysis (SPF, DKIM, sender domain)
    - URL analysis (count, domains, shortened URLs)
    - Content analysis (urgency words, formatting)
    - Structural features (length ratios, attachment indicators)
    """

    # Known suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = [
        '.xyz', '.top', '.work', '.click', '.link', '.info', '.online',
        '.site', '.website', '.space', '.tech', '.store', '.shop',
        '.win', '.loan', '.party', '.review', '.science', '.gq', '.ml',
        '.cf', '.tk', '.ga'
    ]

    # Common legitimate domains (for comparison)
    LEGITIMATE_DOMAINS = [
        'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
        'facebook.com', 'twitter.com', 'linkedin.com', 'github.com',
        'paypal.com', 'ebay.com', 'netflix.com', 'spotify.com'
    ]

    # URL shortener domains
    URL_SHORTENERS = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'j.mp', 'tr.im', 'short.to', 'rebrand.ly',
        'tiny.cc', 'shorte.st', 'cutt.ly', 'shorturl.at', 'rb.gy',
        'snip.ly', 'sniply.io', 'clck.ru', 'v.gd', 'bc.vc', 'po.st'
    ]

    # Urgency and phishing keywords
    URGENCY_KEYWORDS = [
        'urgent', 'immediate', 'action required', 'verify', 'confirm',
        'suspend', 'locked', 'limited', 'expire', 'deadline', 'asap',
        'within 24 hours', 'within 48 hours', 'account will be',
        'security alert', 'unusual activity', 'unauthorized'
    ]

    PHISHING_KEYWORDS = [
        'click here', 'click below', 'click the link', 'log in',
        'sign in', 'update your', 'verify your', 'confirm your',
        'password', 'ssn', 'social security', 'credit card',
        'bank account', 'routing number', 'pin number',
        'winner', 'won', 'lottery', 'prize', 'free gift',
        'act now', 'limited time', 'offer expires'
    ]

    # Patterns
    # URL pattern that detects URLs with and without http/https prefix
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'(?:[-\w]+\.)+(?:com|org|net|co|uk|io|me|ly|to|gl|gg|cc|tv|us|es|de|fr|it|ru|cn|in|br|mx|ar|cl|pe|info|biz|xyz|top|click|link|online|site|tech|store|shop|app|dev|ai|cloud)[^\s]*',
        re.IGNORECASE
    )
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    IP_URL_PATTERN = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

    def __init__(self, include_text_stats: bool = True):
        """
        Initialize the EmailFeatureExtractor.

        Args:
            include_text_stats: Include basic text statistics
        """
        self.include_text_stats = include_text_stats
        self.feature_names = []

    def extract_url_features(self, text: str) -> Dict[str, float]:
        """
        Extract URL-related features from email text.

        Args:
            text: Email body text

        Returns:
            Dictionary of URL features
        """
        if not text:
            return self._empty_url_features()

        text_lower = text.lower()
        urls = self.URL_PATTERN.findall(text)

        features = {
            'url_count': len(urls),
            'has_url': int(len(urls) > 0),  # Estandarizado con SMS (antes: has_urls)
            'url_to_text_ratio': len(urls) / max(len(text.split()), 1),
        }

        # Analyze URLs
        shortened_count = 0
        suspicious_tld_count = 0
        ip_url_count = 0
        unique_domains = set()

        for url in urls:
            try:
                parsed = urlparse(url if url.startswith('http') else f'http://{url}')
                domain = parsed.netloc.lower()
                unique_domains.add(domain)

                # Check for URL shorteners
                if any(shortener in domain for shortener in self.URL_SHORTENERS):
                    shortened_count += 1

                # Check for suspicious TLDs
                if any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS):
                    suspicious_tld_count += 1

            except Exception:
                pass

        # Check for IP-based URLs
        ip_url_count = len(self.IP_URL_PATTERN.findall(text))

        features.update({
            'shortened_url_count': shortened_count,
            'has_shortened_url': int(shortened_count > 0),
            'suspicious_tld_count': suspicious_tld_count,
            'has_suspicious_tld': int(suspicious_tld_count > 0),
            'ip_url_count': ip_url_count,
            'has_ip_url': int(ip_url_count > 0),
            'unique_domain_count': len(unique_domains),
        })

        return features

    def _empty_url_features(self) -> Dict[str, float]:
        """Return empty URL features."""
        return {
            'url_count': 0,
            'has_url': 0,
            'url_to_text_ratio': 0,
            'shortened_url_count': 0,
            'has_shortened_url': 0,
            'suspicious_tld_count': 0,
            'has_suspicious_tld': 0,
            'ip_url_count': 0,
            'has_ip_url': 0,
            'unique_domain_count': 0,
        }

    def extract_header_features(self, headers: Dict) -> Dict[str, float]:
        """
        Extract features from email headers.

        Args:
            headers: Dictionary of email headers

        Returns:
            Dictionary of header features
        """
        if not headers:
            return self._empty_header_features()

        features = {}

        # From address analysis
        from_addr = headers.get('From', '') or headers.get('from', '')
        if from_addr:
            # Check for display name vs actual email mismatch
            email_match = self.EMAIL_PATTERN.search(from_addr)
            if email_match:
                email = email_match.group().lower()
                domain = email.split('@')[-1] if '@' in email else ''

                features['sender_domain_length'] = len(domain)
                features['sender_is_legitimate_domain'] = int(
                    any(legit in domain for legit in self.LEGITIMATE_DOMAINS)
                )
                features['sender_has_suspicious_tld'] = int(
                    any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS)
                )

                # Check for lookalike domains (e.g., paypa1.com)
                features['sender_has_numbers_in_domain'] = int(
                    bool(re.search(r'\d', domain.split('.')[0]))
                )
            else:
                features.update({
                    'sender_domain_length': 0,
                    'sender_is_legitimate_domain': 0,
                    'sender_has_suspicious_tld': 0,
                    'sender_has_numbers_in_domain': 0
                })
        else:
            features.update({
                'sender_domain_length': 0,
                'sender_is_legitimate_domain': 0,
                'sender_has_suspicious_tld': 0,
                'sender_has_numbers_in_domain': 0
            })

        # Reply-To mismatch
        reply_to = headers.get('Reply-To', '') or headers.get('reply-to', '')
        if reply_to and from_addr:
            from_email = self.EMAIL_PATTERN.search(from_addr)
            reply_email = self.EMAIL_PATTERN.search(reply_to)
            if from_email and reply_email:
                features['reply_to_mismatch'] = int(
                    from_email.group().lower() != reply_email.group().lower()
                )
            else:
                features['reply_to_mismatch'] = 0
        else:
            features['reply_to_mismatch'] = 0

        # Check for common phishing header patterns
        features['has_x_mailer'] = int('X-Mailer' in headers or 'x-mailer' in headers)
        features['has_received_spf'] = int(
            'Received-SPF' in headers or 'received-spf' in headers
        )

        return features

    def _empty_header_features(self) -> Dict[str, float]:
        """Return empty header features."""
        return {
            'sender_domain_length': 0,
            'sender_is_legitimate_domain': 0,
            'sender_has_suspicious_tld': 0,
            'sender_has_numbers_in_domain': 0,
            'reply_to_mismatch': 0,
            'has_x_mailer': 0,
            'has_received_spf': 0
        }

    def extract_content_features(self, subject: str, body: str) -> Dict[str, float]:
        """
        Extract content-related features.

        Args:
            subject: Email subject
            body: Email body

        Returns:
            Dictionary of content features
        """
        subject = subject or ''
        body = body or ''

        text = f"{subject} {body}".lower()
        body_lower = body.lower()

        features = {}

        # Urgency keywords
        urgency_count = sum(1 for kw in self.URGENCY_KEYWORDS if kw in text)
        features['urgency_keyword_count'] = urgency_count
        features['has_urgency_keywords'] = int(urgency_count > 0)

        # Phishing keywords
        phishing_count = sum(1 for kw in self.PHISHING_KEYWORDS if kw in text)
        features['phishing_keyword_count'] = phishing_count
        features['has_phishing_keywords'] = int(phishing_count > 0)

        # Subject analysis
        features['subject_length'] = len(subject)
        features['subject_word_count'] = len(subject.split())
        features['subject_has_urgency'] = int(
            any(kw in subject.lower() for kw in self.URGENCY_KEYWORDS)
        )
        features['subject_has_re_fw'] = int(
            subject.lower().startswith(('re:', 'fw:', 'fwd:'))
        )
        features['subject_all_caps'] = int(
            subject.isupper() and len(subject) > 3
        )
        features['subject_exclamation_count'] = subject.count('!')

        # Body analysis
        features['body_length'] = len(body)
        features['body_word_count'] = len(body.split())

        # Formatting features
        features['html_tag_count'] = len(re.findall(r'<[^>]+>', body))
        features['has_html'] = int(features['html_tag_count'] > 0)

        # Capitalization
        if body:
            upper_chars = sum(1 for c in body if c.isupper())
            alpha_chars = sum(1 for c in body if c.isalpha())
            features['caps_ratio'] = upper_chars / max(alpha_chars, 1)
        else:
            features['caps_ratio'] = 0

        # Special characters
        features['exclamation_count'] = body.count('!')
        features['question_count'] = body.count('?')
        features['dollar_sign_count'] = body.count('$') + body.count('£') + body.count('€')

        # Spelling/grammar indicators (simplified)
        # Count common misspellings often seen in phishing
        common_misspellings = [
            'accout', 'verifiy', 'pasword', 'securty', 'updtae',
            'suspendd', 'verfy', 'confrim'
        ]
        features['misspelling_indicator'] = sum(
            1 for word in common_misspellings if word in body_lower
        )

        return features

    def extract_structural_features(
        self,
        email_data: Dict,
        include_attachments: bool = True
    ) -> Dict[str, float]:
        """
        Extract structural features from email.

        Args:
            email_data: Dictionary containing email fields
            include_attachments: Include attachment-related features

        Returns:
            Dictionary of structural features
        """
        features = {}

        body = email_data.get('body', '') or ''
        subject = email_data.get('subject', '') or ''

        # Length ratios
        features['subject_body_length_ratio'] = (
            len(subject) / max(len(body), 1)
        )

        # Email count in body
        emails_in_body = self.EMAIL_PATTERN.findall(body)
        features['email_addresses_in_body'] = len(emails_in_body)

        # Attachment indicators
        if include_attachments:
            features['has_attachments'] = int(
                email_data.get('has_attachments', False)
            )

            # Check for attachment keywords
            attachment_keywords = ['attachment', 'attached', 'enclosed', 'see attached']
            features['mentions_attachment'] = int(
                any(kw in body.lower() for kw in attachment_keywords)
            )

        # Link text mismatch detection (simplified)
        # Look for patterns like <a href="bad.com">good.com</a>
        href_pattern = re.compile(r'href=["\']([^"\']+)["\'][^>]*>([^<]+)', re.IGNORECASE)
        href_matches = href_pattern.findall(body)
        mismatch_count = 0
        for href, text in href_matches:
            href_domain = urlparse(href).netloc.lower() if href.startswith('http') else ''
            if href_domain and href_domain not in text.lower() and '.' in text:
                mismatch_count += 1
        features['link_text_mismatch_count'] = mismatch_count

        return features

    def extract_all_features(self, email_data: Dict) -> Dict[str, float]:
        """
        Extract all features from an email.

        Args:
            email_data: Dictionary containing email fields
                Expected keys: 'subject', 'body', 'headers', 'from', etc.

        Returns:
            Dictionary of all features
        """
        features = {}

        body = email_data.get('body', '') or ''
        subject = email_data.get('subject', '') or ''
        headers = email_data.get('headers', {}) or {}

        # URL features (from body)
        features.update(self.extract_url_features(body))

        # Header features
        features.update(self.extract_header_features(headers))

        # Content features
        features.update(self.extract_content_features(subject, body))

        # Structural features
        features.update(self.extract_structural_features(email_data))

        return features

    def extract_features_batch(
        self,
        emails: Union[List[Dict], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extract features from a batch of emails.

        Args:
            emails: List of email dictionaries or DataFrame

        Returns:
            DataFrame with extracted features
        """
        if isinstance(emails, pd.DataFrame):
            emails = emails.to_dict('records')

        features_list = []
        for email_data in emails:
            features = self.extract_all_features(email_data)
            features_list.append(features)

        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not self.feature_names:
            # Generate from sample extraction
            sample = self.extract_all_features({
                'subject': 'test',
                'body': 'test body',
                'headers': {}
            })
            self.feature_names = list(sample.keys())
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    extractor = EmailFeatureExtractor()

    sample_email = {
        'subject': 'URGENT: Your account has been suspended!',
        'body': '''
        Dear Customer,

        We have detected unusual activity on your account. Your account has been
        temporarily suspended. Please click the link below to verify your identity:

        http://paypa1-security.com/verify?user=123

        If you do not verify within 24 hours, your account will be permanently closed.

        Best regards,
        Security Team
        ''',
        'from': 'security@paypa1-alerts.xyz',
        'headers': {
            'From': 'PayPal Security <security@paypa1-alerts.xyz>',
            'Reply-To': 'support@different-domain.com'
        }
    }

    features = extractor.extract_all_features(sample_email)

    print("Extracted features:")
    for name, value in sorted(features.items()):
        print(f"  {name}: {value}")
