"""
Email dataset loader for phishing detection.

Supports loading from phishing-specific datasets:
- Hugging Face zefang-liu/phishing-email-dataset (18,700 real phishing emails)
- Nazario Phishing Corpus
- Custom CSV/JSON formats
"""

import os
import email
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from email.parser import Parser
from email.policy import default

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailLoader:
    """Loader class for email datasets used in phishing detection."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the EmailLoader.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "email"
        self.processed_dir = self.data_dir / "processed" / "email"
        self.external_dir = self.data_dir / "external"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        self.email_parser = Parser(policy=default)

    def load_huggingface_phishing(self) -> pd.DataFrame:
        """
        Load real phishing email dataset.

        First tries to load from local file (data/external/phishing_email_dataset.csv).
        If not found, downloads from Hugging Face and saves locally.

        Dataset: zefang-liu/phishing-email-dataset
        Contains: 18,700 emails labeled as phishing or safe
        Source: Kaggle "Phishing Email Detection" dataset

        Returns:
            DataFrame with standardized columns (body, label, source)
        """
        # Local file path
        local_path = self.external_dir / "phishing_email_dataset.csv"

        # Try to load from local file first
        if local_path.exists():
            logger.info(f"Loading phishing dataset from local file: {local_path}")
            try:
                df = pd.read_csv(local_path)
                logger.info(f"Loaded {len(df)} emails from local file")
                logger.info(f"  Phishing: {(df['label'] == 1).sum()}")
                logger.info(f"  Safe: {(df['label'] == 0).sum()}")
                return df
            except Exception as e:
                logger.warning(f"Error loading local file, will download from HuggingFace: {e}")

        # Download from HuggingFace
        logger.info("Loading phishing dataset from Hugging Face (zefang-liu/phishing-email-dataset)...")

        try:
            from datasets import load_dataset

            # Load real phishing dataset
            ds = load_dataset("zefang-liu/phishing-email-dataset")

            # Convert to pandas (only 'train' split available)
            df = ds['train'].to_pandas()

            logger.info(f"Raw dataset loaded with columns: {df.columns.tolist()}")

            # Standardize columns
            # Dataset columns: 'Email Text', 'Email Type'
            if 'Email Text' in df.columns:
                df = df.rename(columns={'Email Text': 'body'})

            # Map labels: "Phishing Email" -> 1, "Safe Email" -> 0
            if 'Email Type' in df.columns:
                df['label'] = df['Email Type'].map({
                    'Phishing Email': 1,
                    'Safe Email': 0
                })
                df = df.drop(columns=['Email Type'])

            df['source'] = 'huggingface_phishing'

            # Clean up - keep only needed columns
            df = df[['body', 'label', 'source']]
            df = df.dropna(subset=['body', 'label'])
            df['body'] = df['body'].astype(str).str.strip()
            df = df[df['body'].str.len() > 10]  # Filter very short entries

            # Remove duplicates
            original_count = len(df)
            df = df.drop_duplicates(subset=['body'])
            duplicates_removed = original_count - len(df)

            logger.info(f"Loaded {len(df)} emails from Hugging Face")
            logger.info(f"  Removed {duplicates_removed} duplicates")
            logger.info(f"  Phishing: {(df['label'] == 1).sum()}")
            logger.info(f"  Safe: {(df['label'] == 0).sum()}")

            # Save to local file for future use
            try:
                df.to_csv(local_path, index=False)
                logger.info(f"Saved dataset to local file: {local_path}")
            except Exception as e:
                logger.warning(f"Could not save dataset locally: {e}")

            return df

        except ImportError:
            logger.error("'datasets' library not installed. Run: pip install datasets")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Hugging Face phishing dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def parse_email_file(self, filepath: str) -> Dict:
        """
        Parse a single email file and extract relevant fields.

        Args:
            filepath: Path to the email file

        Returns:
            Dictionary containing email fields
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            msg = self.email_parser.parsestr(content)

            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except:
                            body = str(part.get_payload())
                        break
                    elif content_type == "text/html" and not body:
                        try:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except:
                            body = str(part.get_payload())
            else:
                try:
                    body = msg.get_payload(decode=True)
                    if isinstance(body, bytes):
                        body = body.decode('utf-8', errors='ignore')
                except:
                    body = str(msg.get_payload())

            return {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'body': body if body else '',
                'headers': dict(msg.items()),
                'content_type': msg.get_content_type(),
                'has_attachments': msg.is_multipart(),
                'raw_content': content
            }
        except Exception as e:
            logger.warning(f"Error parsing email {filepath}: {e}")
            return None

    def load_directory(
        self,
        directory: str,
        label: int,
        extensions: List[str] = None
    ) -> pd.DataFrame:
        """
        Load all email files from a directory.

        Args:
            directory: Path to directory containing email files
            label: Label to assign (0=legitimate, 1=phishing)
            extensions: List of file extensions to include

        Returns:
            DataFrame with email data
        """
        if extensions is None:
            extensions = ['.eml', '.txt', '']

        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return pd.DataFrame()

        emails = []
        patterns = [f"**/*{ext}" for ext in extensions]

        all_files = []
        for pattern in patterns:
            all_files.extend(directory.glob(pattern))

        # Remove duplicates and filter files only
        all_files = list(set([f for f in all_files if f.is_file()]))

        logger.info(f"Loading {len(all_files)} emails from {directory}")

        for filepath in tqdm(all_files, desc=f"Loading emails (label={label})"):
            email_data = self.parse_email_file(str(filepath))
            if email_data:
                email_data['label'] = label
                email_data['source_file'] = str(filepath)
                emails.append(email_data)

        return pd.DataFrame(emails)

    def load_csv(
        self,
        filepath: str,
        text_column: str = 'text',
        label_column: str = 'label',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load emails from a CSV file.

        Args:
            filepath: Path to CSV file
            text_column: Name of column containing email text
            label_column: Name of column containing labels
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with standardized columns
        """
        df = pd.read_csv(filepath, **kwargs)

        # Standardize column names
        if text_column in df.columns:
            df = df.rename(columns={text_column: 'body'})
        if label_column in df.columns and label_column != 'label':
            df = df.rename(columns={label_column: 'label'})

        # Ensure required columns exist
        if 'body' not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")

        # Add missing columns with defaults
        for col in ['subject', 'from', 'to', 'date']:
            if col not in df.columns:
                df[col] = ''

        df['source'] = 'csv'

        return df

    def load_json(self, filepath: str) -> pd.DataFrame:
        """
        Load emails from a JSON file.

        Expected format: [{"text": "...", "label": 0/1}, ...]

        Args:
            filepath: Path to JSON file

        Returns:
            DataFrame with standardized columns
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Standardize column names
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'body'})
        if 'email_text' in df.columns:
            df = df.rename(columns={'email_text': 'body'})

        df['source'] = 'json'

        return df

    def load_nazario(self, nazario_path: str) -> pd.DataFrame:
        """
        Load Nazario Phishing Corpus.

        This is a real phishing email corpus maintained by Jose Nazario.

        Args:
            nazario_path: Path to Nazario corpus directory

        Returns:
            DataFrame with phishing emails (all labeled as 1)
        """
        logger.info("Loading Nazario Phishing Corpus...")
        df = self.load_directory(nazario_path, label=1)
        df['source'] = 'nazario_phishing'
        return df

    def create_combined_dataset(
        self,
        phishing_sources: List[pd.DataFrame],
        legitimate_sources: List[pd.DataFrame],
        balance: bool = True,
        target_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Combine multiple email sources into a single dataset.

        Args:
            phishing_sources: List of DataFrames with phishing emails
            legitimate_sources: List of DataFrames with legitimate emails
            balance: Whether to balance the dataset
            target_ratio: Target ratio of phishing emails (0.5 = balanced)

        Returns:
            Combined and optionally balanced DataFrame
        """
        # Combine phishing emails
        phishing_df = pd.concat(phishing_sources, ignore_index=True)
        phishing_df['label'] = 1

        # Combine legitimate emails
        legitimate_df = pd.concat(legitimate_sources, ignore_index=True)
        legitimate_df['label'] = 0

        logger.info(f"Total phishing emails: {len(phishing_df)}")
        logger.info(f"Total legitimate emails: {len(legitimate_df)}")

        if balance:
            # Balance to target ratio
            n_phishing = len(phishing_df)
            n_legitimate = len(legitimate_df)

            if target_ratio == 0.5:
                # Equal balance
                min_size = min(n_phishing, n_legitimate)
                phishing_df = phishing_df.sample(n=min_size, random_state=42)
                legitimate_df = legitimate_df.sample(n=min_size, random_state=42)
            else:
                # Custom ratio
                total_target = min(n_phishing / target_ratio, n_legitimate / (1 - target_ratio))
                n_phishing_target = int(total_target * target_ratio)
                n_legitimate_target = int(total_target * (1 - target_ratio))

                phishing_df = phishing_df.sample(n=min(n_phishing_target, n_phishing), random_state=42)
                legitimate_df = legitimate_df.sample(n=min(n_legitimate_target, n_legitimate), random_state=42)

        # Combine final dataset
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Final dataset size: {len(combined_df)}")
        logger.info(f"Phishing ratio: {combined_df['label'].mean():.2%}")

        return combined_df

    def save_processed(self, df: pd.DataFrame, filename: str = "emails_processed.csv"):
        """Save processed dataset to CSV."""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")

    def load_processed(self, filename: str = "emails_processed.csv") -> pd.DataFrame:
        """Load processed dataset from CSV."""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed file not found: {filepath}")
        return pd.read_csv(filepath)

    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the email dataset.

        Args:
            df: DataFrame with email data

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_emails': len(df),
            'phishing_count': int((df['label'] == 1).sum()),
            'legitimate_count': int((df['label'] == 0).sum()),
            'phishing_ratio': float(df['label'].mean()),
            'avg_body_length': float(df['body'].str.len().mean()),
            'avg_subject_length': float(df['subject'].str.len().mean()) if 'subject' in df.columns else 0,
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'missing_body': int(df['body'].isna().sum() + (df['body'] == '').sum()),
        }
        return stats


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample dataset for testing purposes.

    Returns:
        DataFrame with sample email data (phishing examples)
    """
    sample_data = [
        {
            'subject': 'URGENT: Your account has been compromised - Verify now!',
            'from': 'security@bankofamerica-secure.xyz',
            'body': 'Dear Customer, We have detected unusual activity on your account. '
                    'Click here to verify your identity immediately: http://secure-bank-verify.com/login '
                    'or your account will be suspended within 24 hours.',
            'label': 1,
            'source': 'sample'
        },
        {
            'subject': 'Your PayPal account needs immediate verification',
            'from': 'service@paypa1-security.com',
            'body': 'Your PayPal account has been limited due to suspicious activity. '
                    'Please click the link below to restore full access: http://paypal-verify.xyz/restore '
                    'Failure to verify will result in permanent suspension.',
            'label': 1,
            'source': 'sample'
        },
        {
            'subject': 'Meeting tomorrow at 3pm',
            'from': 'john.smith@company.com',
            'body': 'Hi team, Just a reminder that we have our weekly sync tomorrow at 3pm '
                    'in conference room B. Please come prepared with your updates.',
            'label': 0,
            'source': 'sample'
        },
        {
            'subject': 'Your Amazon order has shipped',
            'from': 'ship-confirm@amazon.com',
            'body': 'Your order #123-4567890 has shipped and is on its way. '
                    'Track your package using the link in your Amazon account.',
            'label': 0,
            'source': 'sample'
        },
        {
            'subject': 'Congratulations! You won $1,000,000 in our lottery',
            'from': 'winner@lottery-international.org',
            'body': 'You have been selected as the lucky winner of our international lottery. '
                    'To claim your prize of $1,000,000, please send us your bank details '
                    'and a processing fee of $500 to: claims@lottery-prize.xyz',
            'label': 1,
            'source': 'sample'
        },
        {
            'subject': 'Invoice #12345 attached',
            'from': 'billing@supplier.com',
            'body': 'Please find attached the invoice for last month\'s services. '
                    'Payment is due within 30 days. Contact us if you have any questions.',
            'label': 0,
            'source': 'sample'
        },
        {
            'subject': 'Your Apple ID was used to sign in to iCloud',
            'from': 'noreply@apple-id-support.xyz',
            'body': 'Your Apple ID was just used to sign in to iCloud from a new device. '
                    'If this was not you, click here immediately to secure your account: '
                    'http://apple-secure-verify.com/protect',
            'label': 1,
            'source': 'sample'
        },
        {
            'subject': 'Welcome to the team!',
            'from': 'hr@company.com',
            'body': 'Welcome aboard! We are excited to have you join our team. '
                    'Please complete your onboarding documents by Friday.',
            'label': 0,
            'source': 'sample'
        }
    ]

    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Example usage
    loader = EmailLoader()

    # Try to load Hugging Face phishing dataset
    print("Attempting to load Hugging Face phishing dataset...")
    df = loader.load_huggingface_phishing()

    if not df.empty:
        print(f"\nDataset loaded successfully!")
        print(f"Dataset stats: {loader.get_dataset_stats(df)}")
    else:
        # Fall back to sample dataset
        print("\nUsing sample dataset for demonstration:")
        sample_df = create_sample_dataset()
        print(sample_df[['subject', 'label']].to_string())
        print("\nDataset stats:", loader.get_dataset_stats(sample_df))
