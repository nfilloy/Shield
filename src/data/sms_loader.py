"""
SMS dataset loader for smishing detection.

Supports loading from phishing/smishing-specific datasets:
- Mendeley SMS Phishing Dataset (f45bkkt8pr)
- Hugging Face ealvaradob/phishing-dataset (SMS subset)
- Custom CSV/JSON formats
"""

import os
import json
import logging
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMSLoader:
    """Loader class for SMS datasets used in smishing detection."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the SMSLoader.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "sms"
        self.processed_dir = self.data_dir / "processed" / "sms"
        self.external_dir = self.data_dir / "external"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

    def download_mendeley_smishing(self) -> str:
        """
        Download the Mendeley SMS Phishing Dataset.

        Dataset: https://data.mendeley.com/datasets/f45bkkt8pr/1
        Contains: 5,971 messages (638 smishing + 489 spam + 4,844 ham)

        NOTE: Mendeley requires manual download. This method provides instructions.

        Returns:
            Path to the downloaded file or None if not available
        """
        output_path = self.external_dir / "smishing_mendeley.csv"

        if output_path.exists():
            # Verify it's a valid CSV (not an error page)
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if 'error' in first_line.lower() or '404' in first_line or '{' in first_line:
                        logger.warning("Existing Mendeley file appears to be invalid, removing...")
                        output_path.unlink()
                    else:
                        logger.info(f"Mendeley SMS dataset already exists at {output_path}")
                        return str(output_path)
            except Exception:
                pass

        logger.info("=" * 60)
        logger.info("MENDELEY DATASET REQUIRES MANUAL DOWNLOAD")
        logger.info("=" * 60)
        logger.info("1. Go to: https://data.mendeley.com/datasets/f45bkkt8pr/1")
        logger.info("2. Download the CSV file (Dataset_5971.csv)")
        logger.info(f"3. Save it as: {output_path}")
        logger.info("=" * 60)

        return None

    def load_mendeley_smishing(self, filepath: str = None) -> pd.DataFrame:
        """
        Load the Mendeley SMS Phishing Dataset.

        Filters to include only SMISHING vs HAM (excludes generic spam).

        Dataset format (Dataset_5971.csv):
        - Column 'LABEL': ham, Smishing, smishing, spam, Spam
        - Column 'TEXT': SMS content
        - Column 'URL': yes/No (contains URL)
        - Column 'EMAIL': yes/No (contains email)
        - Column 'PHONE': yes/No (contains phone)

        Args:
            filepath: Path to the Mendeley CSV file

        Returns:
            DataFrame with standardized columns (body, label, source)
        """
        if filepath is None:
            # Buscar en la carpeta del dataset descargado manualmente
            dataset_dir = self.external_dir / "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION"
            filepath = dataset_dir / "Dataset_5971.csv"

            # Fallback a la ubicación antigua si no existe
            if not filepath.exists():
                filepath = self.external_dir / "smishing_mendeley.csv"

        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Mendeley dataset not found at {filepath}")
            download_result = self.download_mendeley_smishing()
            if download_result is None:
                return pd.DataFrame()
            filepath = Path(download_result)

        logger.info(f"Loading Mendeley SMS Phishing Dataset from {filepath}")

        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Total rows in file: {len(df)}")

            # Identify columns (handle various naming conventions)
            text_col = None
            label_col = None

            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['text', 'message', 'sms', 'body', 'content']:
                    text_col = col
                elif col_lower in ['label', 'class', 'category', 'type']:
                    label_col = col

            if text_col is None or label_col is None:
                # Try positional if standard names not found
                if len(df.columns) >= 2:
                    label_col = df.columns[0]
                    text_col = df.columns[1]
                else:
                    logger.error(f"Could not identify columns. Found: {df.columns.tolist()}")
                    return pd.DataFrame()

            logger.info(f"Using columns: text='{text_col}', label='{label_col}'")

            # Normalizar etiquetas (case-insensitive)
            original_count = len(df)
            df[label_col] = df[label_col].astype(str).str.lower().str.strip()

            # Mostrar distribución de etiquetas original
            label_dist = df[label_col].value_counts()
            logger.info(f"Original label distribution:\n{label_dist}")

            # Filtrar: solo smishing y ham (excluir spam genérico)
            # smishing -> 1 (positive class - what we want to detect)
            # ham -> 0 (legitimate)
            # spam -> exclude (generic spam, not phishing)
            df = df[df[label_col].isin(['smishing', 'ham'])]

            # Convertir a binario
            df['label'] = (df[label_col] == 'smishing').astype(int)

            filtered_count = len(df)
            excluded = original_count - filtered_count

            logger.info(f"Filtered dataset: {filtered_count} messages (excluded {excluded} generic spam)")

            # Standardize columns
            df = df.rename(columns={text_col: 'body'})
            df['source'] = 'mendeley_smishing'

            # Clean up
            df = df[['body', 'label', 'source']]
            df = df.dropna(subset=['body'])
            df['body'] = df['body'].astype(str).str.strip()
            df = df[df['body'].str.len() > 0]

            logger.info(f"Loaded {len(df)} SMS messages from Mendeley")
            logger.info(f"  Smishing: {(df['label'] == 1).sum()}")
            logger.info(f"  Legitimate: {(df['label'] == 0).sum()}")

            return df

        except Exception as e:
            logger.error(f"Error loading Mendeley dataset: {e}")
            return pd.DataFrame()

    def load_huggingface_sms(self) -> pd.DataFrame:
        """
        Load SMS spam dataset from Hugging Face.

        Dataset: ucirvine/sms_spam (UCI SMS Spam Collection on HuggingFace)
        Contains 5,574 SMS messages labeled as spam/ham.

        Returns:
            DataFrame with standardized columns (body, label, source)
        """
        logger.info("Loading SMS dataset from Hugging Face (ucirvine/sms_spam)...")

        try:
            from datasets import load_dataset

            # Load UCI SMS Spam dataset from HuggingFace
            ds = load_dataset("ucirvine/sms_spam")

            # Convert to pandas
            df = ds['train'].to_pandas()

            logger.info(f"Raw dataset columns: {df.columns.tolist()}")

            # Standardize columns
            # Expected columns: 'sms' (message) and 'label' (0=ham, 1=spam)
            if 'sms' in df.columns:
                df = df.rename(columns={'sms': 'body'})
            elif 'text' in df.columns:
                df = df.rename(columns={'text': 'body'})
            elif 'message' in df.columns:
                df = df.rename(columns={'message': 'body'})

            # Ensure label column exists
            if 'label' not in df.columns:
                for col in ['Label', 'CLASS', 'class', 'is_spam']:
                    if col in df.columns:
                        df = df.rename(columns={col: 'label'})
                        break

            df['label'] = df['label'].astype(int)
            df['source'] = 'huggingface_sms_spam'

            # Clean up
            df = df[['body', 'label', 'source']]
            df = df.dropna(subset=['body'])
            df['body'] = df['body'].astype(str).str.strip()
            df = df[df['body'].str.len() > 0]

            logger.info(f"Loaded {len(df)} SMS messages from Hugging Face")
            logger.info(f"  Spam/Smishing: {(df['label'] == 1).sum()}")
            logger.info(f"  Legitimate: {(df['label'] == 0).sum()}")

            return df

        except ImportError:
            logger.error("'datasets' library not installed. Run: pip install datasets")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Hugging Face SMS dataset: {e}")
            return pd.DataFrame()

    def load_combined_smishing(self) -> pd.DataFrame:
        """
        Load and combine all available smishing datasets.

        Combines:
        1. Mendeley SMS Phishing Dataset
        2. Hugging Face ealvaradob/phishing-dataset (SMS)

        Removes duplicates based on message content.

        Returns:
            Combined DataFrame with standardized columns
        """
        logger.info("Loading combined smishing dataset...")

        all_dataframes = []

        # 1. Try to load Mendeley dataset
        try:
            df_mendeley = self.load_mendeley_smishing()
            if not df_mendeley.empty:
                all_dataframes.append(df_mendeley)
                logger.info(f"Added {len(df_mendeley)} messages from Mendeley")
        except Exception as e:
            logger.warning(f"Could not load Mendeley dataset: {e}")

        # 2. Try to load Hugging Face dataset
        try:
            df_hf = self.load_huggingface_sms()
            if not df_hf.empty:
                all_dataframes.append(df_hf)
                logger.info(f"Added {len(df_hf)} messages from Hugging Face")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face SMS dataset: {e}")

        # Combine all datasets
        if not all_dataframes:
            logger.error("No SMS datasets could be loaded!")
            return pd.DataFrame()

        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Remove duplicates based on message content (normalized)
        original_count = len(combined_df)
        combined_df['body_normalized'] = combined_df['body'].str.lower().str.strip()
        combined_df = combined_df.drop_duplicates(subset=['body_normalized'])
        combined_df = combined_df.drop(columns=['body_normalized'])
        duplicates_removed = original_count - len(combined_df)

        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(combined_df)} unique messages")
        logger.info(f"  Removed {duplicates_removed} duplicates")
        logger.info(f"  Smishing: {(combined_df['label'] == 1).sum()}")
        logger.info(f"  Legitimate: {(combined_df['label'] == 0).sum()}")
        logger.info(f"  Sources: {combined_df['source'].value_counts().to_dict()}")

        return combined_df

    def load_csv(
        self,
        filepath: str,
        text_column: str = 'text',
        label_column: str = 'label',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load SMS data from a generic CSV file.

        Args:
            filepath: Path to CSV file
            text_column: Name of column containing SMS text
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

        if 'body' not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")

        df['source'] = 'csv'

        return df[['body', 'label', 'source']]

    def load_json(self, filepath: str) -> pd.DataFrame:
        """
        Load SMS data from a JSON file.

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
        if 'message' in df.columns:
            df = df.rename(columns={'message': 'body'})

        df['source'] = 'json'

        return df[['body', 'label', 'source']]

    def create_combined_dataset(
        self,
        smishing_sources: List[pd.DataFrame],
        legitimate_sources: List[pd.DataFrame],
        balance: bool = True,
        target_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Combine multiple SMS sources into a single dataset.

        Args:
            smishing_sources: List of DataFrames with smishing SMS
            legitimate_sources: List of DataFrames with legitimate SMS
            balance: Whether to balance the dataset
            target_ratio: Target ratio of smishing messages (0.5 = balanced)

        Returns:
            Combined and optionally balanced DataFrame
        """
        # Combine smishing SMS
        smishing_df = pd.concat(smishing_sources, ignore_index=True)
        smishing_df['label'] = 1

        # Combine legitimate SMS
        legitimate_df = pd.concat(legitimate_sources, ignore_index=True)
        legitimate_df['label'] = 0

        logger.info(f"Total smishing SMS: {len(smishing_df)}")
        logger.info(f"Total legitimate SMS: {len(legitimate_df)}")

        if balance:
            n_smishing = len(smishing_df)
            n_legitimate = len(legitimate_df)

            if target_ratio == 0.5:
                min_size = min(n_smishing, n_legitimate)
                smishing_df = smishing_df.sample(n=min_size, random_state=42)
                legitimate_df = legitimate_df.sample(n=min_size, random_state=42)
            else:
                total_target = min(n_smishing / target_ratio, n_legitimate / (1 - target_ratio))
                n_smishing_target = int(total_target * target_ratio)
                n_legitimate_target = int(total_target * (1 - target_ratio))

                smishing_df = smishing_df.sample(n=min(n_smishing_target, n_smishing), random_state=42)
                legitimate_df = legitimate_df.sample(n=min(n_legitimate_target, n_legitimate), random_state=42)

        combined_df = pd.concat([smishing_df, legitimate_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Final dataset size: {len(combined_df)}")
        logger.info(f"Smishing ratio: {combined_df['label'].mean():.2%}")

        return combined_df

    def save_processed(self, df: pd.DataFrame, filename: str = "sms_processed.csv"):
        """Save processed dataset to CSV."""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")

    def load_processed(self, filename: str = "sms_processed.csv") -> pd.DataFrame:
        """Load processed dataset from CSV."""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed file not found: {filepath}")
        return pd.read_csv(filepath)

    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the SMS dataset.

        Args:
            df: DataFrame with SMS data

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_sms': len(df),
            'smishing_count': int((df['label'] == 1).sum()),
            'legitimate_count': int((df['label'] == 0).sum()),
            'smishing_ratio': float(df['label'].mean()),
            'avg_message_length': float(df['body'].str.len().mean()),
            'min_message_length': int(df['body'].str.len().min()),
            'max_message_length': int(df['body'].str.len().max()),
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'empty_messages': int(df['body'].isna().sum() + (df['body'] == '').sum()),
        }
        return stats


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample SMS dataset for testing purposes.

    Returns:
        DataFrame with sample SMS data (smishing examples)
    """
    sample_data = [
        {
            'body': 'URGENT: Your bank account has been compromised. Verify your identity at secure-bank-verify.com/login',
            'label': 1,
            'source': 'sample'
        },
        {
            'body': 'Congratulations! You won $1000 gift card. Claim now: bit.ly/claim-prize-now',
            'label': 1,
            'source': 'sample'
        },
        {
            'body': 'Your package could not be delivered. Update delivery info: track.delivery-notice.xyz',
            'label': 1,
            'source': 'sample'
        },
        {
            'body': 'Hey! Are we still meeting for coffee at 3pm?',
            'label': 0,
            'source': 'sample'
        },
        {
            'body': 'Your verification code is 123456. Do not share this with anyone.',
            'label': 0,
            'source': 'sample'
        },
        {
            'body': 'Reminder: Your dentist appointment is tomorrow at 10am.',
            'label': 0,
            'source': 'sample'
        },
        {
            'body': 'ALERT: Suspicious login detected. Confirm identity: secure-verify.net/auth',
            'label': 1,
            'source': 'sample'
        },
        {
            'body': 'Mom said dinner is at 7. Don\'t be late!',
            'label': 0,
            'source': 'sample'
        },
        {
            'body': 'Your Netflix account will be suspended. Update payment: netflix-billing.co/update',
            'label': 1,
            'source': 'sample'
        },
        {
            'body': 'Thanks for your order! Your confirmation number is ORD-789456.',
            'label': 0,
            'source': 'sample'
        }
    ]

    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Example usage
    loader = SMSLoader()

    # Try to load combined smishing dataset
    print("Attempting to load combined smishing dataset...")
    df = loader.load_combined_smishing()

    if not df.empty:
        print(f"\nDataset loaded successfully!")
        print(f"Dataset stats: {loader.get_dataset_stats(df)}")
    else:
        # Fall back to sample dataset
        print("\nUsing sample dataset for demonstration:")
        sample_df = create_sample_dataset()
        print(sample_df[['body', 'label']].to_string())
        print("\nDataset stats:", loader.get_dataset_stats(sample_df))
