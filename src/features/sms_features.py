"""
SMS-specific feature extraction module.

Extracts features specific to SMS messages that are useful
for smishing (SMS phishing) detection.
"""

import re
import logging
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMSFeatureExtractor:
    """
    Extract SMS-specific features for smishing detection.

    Features include:
    - Message length and structure
    - URL analysis (shortened URLs, suspicious domains)
    - Phone number patterns
    - Urgency and promotional keywords
    - Character patterns (emoji, special chars)
    """

    # URL shortener domains commonly used in smishing
    URL_SHORTENERS = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'j.mp', 'tr.im', 'short.to', 'rebrand.ly',
        'tiny.cc', 'shorte.st', 'cutt.ly', 'shorturl.at', 'rb.gy',
        'snip.ly', 'sniply.io', 'clck.ru', 'v.gd', 'bc.vc', 'po.st'
    ]

    # Suspicious TLDs
    SUSPICIOUS_TLDS = [
        '.xyz', '.top', '.work', '.click', '.link', '.info', '.online',
        '.site', '.website', '.space', '.tech', '.store', '.shop',
        '.win', '.loan', '.party', '.gq', '.ml', '.cf', '.tk', '.ga'
    ]

    # Urgency keywords specific to SMS scams
    URGENCY_KEYWORDS = [
        'urgent', 'immediate', 'now', 'today', 'expires', 'limited',
        'act fast', 'hurry', 'last chance', 'final notice', 'asap',
        'within 24', 'deadline', 'don\'t miss', 'time sensitive'
    ]

    # Promotional/scam keywords
    PROMO_KEYWORDS = [
        'free', 'winner', 'won', 'congratulations', 'prize', 'reward',
        'gift', 'cash', 'money', 'credit', 'loan', 'discount',
        'offer', 'deal', 'save', 'bonus', 'guaranteed', 'selected',
        'lucky', 'exclusive', 'vip', 'special'
    ]

    # Financial/sensitive keywords
    FINANCIAL_KEYWORDS = [
        'bank', 'account', 'password', 'pin', 'verify', 'confirm',
        'update', 'secure', 'security', 'suspend', 'block', 'locked',
        'unauthorized', 'fraud', 'transaction', 'payment', 'refund',
        'tax', 'irs', 'social security', 'ssn', 'credit card'
    ]

    # Action keywords (call to action)
    ACTION_KEYWORDS = [
        'click', 'tap', 'call', 'text', 'reply', 'send', 'visit',
        'go to', 'open', 'download', 'install', 'activate', 'register',
        'sign up', 'log in', 'login', 'enter'
    ]

    # Patterns
    # URL pattern that detects URLs with and without http/https prefix
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'(?:[-\w]+\.)+(?:com|org|net|co|uk|io|me|ly|to|gl|gg|cc|tv|us|es|de|fr|it|ru|cn|in|br|mx|ar|cl|pe|info|biz|xyz|top|click|link|online|site|tech|store|shop|app|dev|ai|cloud)[^\s]*',
        re.IGNORECASE
    )

    PHONE_PATTERN = re.compile(
        r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|'
        r'\d{5,6}|'  # Short codes
        r'(?:\+\d{1,3}[-.\s]?)?\d{8,}'
    )

    # Emoji pattern (simplified)
    EMOJI_PATTERN = re.compile(
        r'[\U0001F300-\U0001F9FF]|'
        r'[\u2600-\u26FF]|'
        r'[\u2700-\u27BF]',
        re.UNICODE
    )

    # Money pattern
    MONEY_PATTERN = re.compile(
        r'[$£€¥]\s*\d+(?:[,\.]\d+)*|'
        r'\d+(?:[,\.]\d+)*\s*(?:dollars?|euros?|pounds?|usd|gbp|eur)',
        re.IGNORECASE
    )

    def __init__(self):
        """Initialize the SMSFeatureExtractor."""
        self.feature_names = []

    def extract_length_features(self, text: str) -> Dict[str, float]:
        """
        Extract length and structure features.

        Args:
            text: SMS text

        Returns:
            Dictionary of length features
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'is_single_sms': 1,
                'is_long_sms': 0
            }

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        char_count = len(text)

        return {
            'char_count': char_count,
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len(sentences),
            'is_single_sms': int(char_count <= 160),
            'is_long_sms': int(char_count > 320),
        }

    def extract_url_features(self, text: str) -> Dict[str, float]:
        """
        Extract URL-related features.

        Args:
            text: SMS text

        Returns:
            Dictionary of URL features
        """
        if not text:
            return self._empty_url_features()

        urls = self.URL_PATTERN.findall(text)

        features = {
            'url_count': len(urls),
            'has_url': int(len(urls) > 0),
            'url_ratio': len(''.join(urls)) / max(len(text), 1),
        }

        shortened_count = 0
        suspicious_tld_count = 0
        ip_url_count = 0

        for url in urls:
            url_lower = url.lower()

            # Check for URL shorteners
            if any(shortener in url_lower for shortener in self.URL_SHORTENERS):
                shortened_count += 1

            # Check for suspicious TLDs
            if any(url_lower.endswith(tld) or f'{tld}/' in url_lower
                   for tld in self.SUSPICIOUS_TLDS):
                suspicious_tld_count += 1

            # Check for IP-based URLs
            if re.search(r'://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                ip_url_count += 1

        features.update({
            'shortened_url_count': shortened_count,
            'has_shortened_url': int(shortened_count > 0),
            'suspicious_tld_count': suspicious_tld_count,
            'has_suspicious_tld': int(suspicious_tld_count > 0),
            'ip_url_count': ip_url_count,
            'has_ip_url': int(ip_url_count > 0),
        })

        return features

    def _empty_url_features(self) -> Dict[str, float]:
        """Return empty URL features."""
        return {
            'url_count': 0,
            'has_url': 0,
            'url_ratio': 0,
            'shortened_url_count': 0,
            'has_shortened_url': 0,
            'suspicious_tld_count': 0,
            'has_suspicious_tld': 0,
            'ip_url_count': 0,
            'has_ip_url': 0,
        }

    def extract_phone_features(self, text: str) -> Dict[str, float]:
        """
        Extract phone number related features.

        Args:
            text: SMS text

        Returns:
            Dictionary of phone features
        """
        if not text:
            return {
                'phone_count': 0,
                'has_phone': 0,
                'has_short_code': 0,
                'has_premium_indicator': 0
            }

        phones = self.PHONE_PATTERN.findall(text)

        # Check for short codes (5-6 digit numbers)
        short_codes = [p for p in phones if len(re.sub(r'\D', '', p)) in [5, 6]]

        # Premium number indicators
        premium_indicators = ['900', '976', '1-900', '1900']
        has_premium = any(ind in text for ind in premium_indicators)

        return {
            'phone_count': len(phones),
            'has_phone': int(len(phones) > 0),
            'has_short_code': int(len(short_codes) > 0),
            'has_premium_indicator': int(has_premium)
        }

    def extract_keyword_features(self, text: str) -> Dict[str, float]:
        """
        Extract keyword-based features.

        Args:
            text: SMS text

        Returns:
            Dictionary of keyword features
        """
        if not text:
            return {
                'urgency_count': 0,
                'has_urgency': 0,
                'promo_count': 0,
                'has_promo': 0,
                'financial_count': 0,
                'has_financial': 0,
                'action_count': 0,
                'has_action': 0,
                'total_suspicious_keywords': 0
            }

        text_lower = text.lower()

        urgency_count = sum(1 for kw in self.URGENCY_KEYWORDS if kw in text_lower)
        promo_count = sum(1 for kw in self.PROMO_KEYWORDS if kw in text_lower)
        financial_count = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in text_lower)
        action_count = sum(1 for kw in self.ACTION_KEYWORDS if kw in text_lower)

        return {
            'urgency_count': urgency_count,
            'has_urgency': int(urgency_count > 0),
            'promo_count': promo_count,
            'has_promo': int(promo_count > 0),
            'financial_count': financial_count,
            'has_financial': int(financial_count > 0),
            'action_count': action_count,
            'has_action': int(action_count > 0),
            'total_suspicious_keywords': urgency_count + promo_count + financial_count
        }

    def extract_character_features(self, text: str) -> Dict[str, float]:
        """
        Extract character-level features.

        Args:
            text: SMS text

        Returns:
            Dictionary of character features
        """
        if not text:
            return {
                'uppercase_ratio': 0,
                'digit_ratio': 0,
                'special_char_ratio': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'emoji_count': 0,
                'has_emoji': 0,
                'consecutive_caps_max': 0,
                'money_mention_count': 0,
                'has_money': 0
            }

        # Character ratios
        alpha_count = sum(1 for c in text if c.isalpha())
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

        # Emoji count
        emojis = self.EMOJI_PATTERN.findall(text)

        # Consecutive caps
        caps_sequences = re.findall(r'[A-Z]{2,}', text)
        max_caps = max([len(s) for s in caps_sequences], default=0)

        # Money mentions
        money_mentions = self.MONEY_PATTERN.findall(text)

        return {
            'uppercase_ratio': upper_count / max(alpha_count, 1),
            'digit_ratio': digit_count / max(len(text), 1),
            'special_char_ratio': special_count / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'emoji_count': len(emojis),
            'has_emoji': int(len(emojis) > 0),
            'consecutive_caps_max': max_caps,
            'money_mention_count': len(money_mentions),
            'has_money': int(len(money_mentions) > 0)
        }

    def extract_pattern_features(self, text: str) -> Dict[str, float]:
        """
        Extract pattern-based features specific to smishing.

        Args:
            text: SMS text

        Returns:
            Dictionary of pattern features
        """
        if not text:
            return {
                'has_verification_code': 0,
                'has_tracking_number': 0,
                'has_reference_number': 0,
                'requests_reply': 0,
                'has_greeting': 0,
                'impersonation_indicator': 0
            }

        text_lower = text.lower()

        # Verification code pattern (e.g., "code: 123456", "OTP: 1234")
        verification_patterns = [
            r'\b(?:code|otp|pin|verification)\s*[:\s]\s*\d{4,8}\b',
            r'\b\d{4,6}\s+is\s+your\s+(?:code|otp|pin)\b'
        ]
        has_verification = any(
            re.search(p, text_lower) for p in verification_patterns
        )

        # Tracking number pattern
        tracking_patterns = [
            r'\b(?:tracking|shipment|order)\s*(?:#|number|no\.?)?\s*[:\s]?\s*[A-Z0-9]{8,}',
        ]
        has_tracking = any(
            re.search(p, text, re.IGNORECASE) for p in tracking_patterns
        )

        # Reference number pattern
        ref_patterns = [
            r'\b(?:ref|reference|confirmation)\s*(?:#|number|no\.?)?\s*[:\s]?\s*[A-Z0-9]{6,}'
        ]
        has_reference = any(
            re.search(p, text, re.IGNORECASE) for p in ref_patterns
        )

        # Reply request
        reply_patterns = ['reply', 'text back', 'send', 'respond']
        requests_reply = any(p in text_lower for p in reply_patterns)

        # Common greeting
        greetings = ['dear', 'hello', 'hi ', 'hey ', 'good morning', 'good evening']
        has_greeting = any(text_lower.startswith(g) or f' {g}' in text_lower
                          for g in greetings)

        # Impersonation indicators (company names followed by suspicious content)
        impersonation_keywords = [
            'amazon', 'paypal', 'apple', 'microsoft', 'google', 'netflix',
            'bank of america', 'wells fargo', 'chase', 'usps', 'fedex', 'ups',
            'irs', 'social security'
        ]
        has_impersonation = (
            any(kw in text_lower for kw in impersonation_keywords) and
            (self.extract_url_features(text)['has_url'] or
             self.extract_keyword_features(text)['has_urgency'])
        )

        return {
            'has_verification_code': int(has_verification),
            'has_tracking_number': int(has_tracking),
            'has_reference_number': int(has_reference),
            'requests_reply': int(requests_reply),
            'has_greeting': int(has_greeting),
            'impersonation_indicator': int(has_impersonation)
        }

    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from an SMS.

        Args:
            text: SMS text

        Returns:
            Dictionary of all features
        """
        features = {}

        features.update(self.extract_length_features(text))
        features.update(self.extract_url_features(text))
        features.update(self.extract_phone_features(text))
        features.update(self.extract_keyword_features(text))
        features.update(self.extract_character_features(text))
        features.update(self.extract_pattern_features(text))

        return features

    def extract_features_batch(
        self,
        messages: Union[List[str], pd.Series]
    ) -> pd.DataFrame:
        """
        Extract features from a batch of SMS messages.

        Args:
            messages: List of SMS texts or Series

        Returns:
            DataFrame with extracted features
        """
        if isinstance(messages, pd.Series):
            messages = messages.fillna('').tolist()

        features_list = []
        for text in messages:
            features = self.extract_all_features(text)
            features_list.append(features)

        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not self.feature_names:
            sample = self.extract_all_features("test message")
            self.feature_names = list(sample.keys())
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    extractor = SMSFeatureExtractor()

    sample_messages = [
        "CONGRATULATIONS! You've won £1000! Claim now: bit.ly/claim-prize",
        "Your Amazon package is delayed. Track here: amaz0n-track.xyz/123",
        "Hi! Are we still meeting for coffee at 3pm?",
        "Your verification code is 123456. Do not share with anyone.",
        "URGENT: Your bank account has been locked. Call 1-800-FAKE-NUM now!",
        "Mom: Don't forget to pick up groceries on your way home.",
    ]

    print("SMS Feature Extraction Examples:\n")
    for msg in sample_messages:
        features = extractor.extract_all_features(msg)
        print(f"Message: {msg[:60]}...")
        print(f"  URL count: {features['url_count']}")
        print(f"  Has shortened URL: {features['has_shortened_url']}")
        print(f"  Urgency count: {features['urgency_count']}")
        print(f"  Total suspicious keywords: {features['total_suspicious_keywords']}")
        print()
