"""
Transformer-based models for phishing/smishing detection.

Implements fine-tuning of pre-trained transformer models:
- BERT
- DistilBERT
- RoBERTa
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required libraries
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        BertTokenizer,
        BertForSequenceClassification,
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        RobertaTokenizer,
        RobertaForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    from datasets import Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Install with: pip install transformers datasets")


# Model configurations
MODEL_CONFIGS = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'tokenizer_class': 'BertTokenizer' if TRANSFORMERS_AVAILABLE else None,
        'model_class': 'BertForSequenceClassification' if TRANSFORMERS_AVAILABLE else None
    },
    'distilbert': {
        'model_name': 'distilbert-base-uncased',
        'tokenizer_class': 'DistilBertTokenizer' if TRANSFORMERS_AVAILABLE else None,
        'model_class': 'DistilBertForSequenceClassification' if TRANSFORMERS_AVAILABLE else None
    },
    'roberta': {
        'model_name': 'roberta-base',
        'tokenizer_class': 'RobertaTokenizer' if TRANSFORMERS_AVAILABLE else None,
        'model_class': 'RobertaForSequenceClassification' if TRANSFORMERS_AVAILABLE else None
    },
    'bert-multilingual': {
        'model_name': 'bert-base-multilingual-uncased',
        'tokenizer_class': 'BertTokenizer' if TRANSFORMERS_AVAILABLE else None,
        'model_class': 'BertForSequenceClassification' if TRANSFORMERS_AVAILABLE else None
    }
}


if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    class PhishingDataset(Dataset):
        """Custom Dataset for transformer models."""

        def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer,
            max_length: int = 256
        ):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


class TransformerModels:
    """
    Manager class for transformer-based models.

    Provides fine-tuning, evaluation, and inference utilities
    for BERT, DistilBERT, and RoBERTa models.
    """

    def __init__(
        self,
        model_dir: str = "models",
        device: Optional[str] = None
    ):
        """
        Initialize TransformerModels.

        Args:
            model_dir: Directory for saving models
            device: Device to use ('cuda' or 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library required. Install with: pip install transformers datasets"
            )

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.training_histories: Dict[str, Dict] = {}

    def load_pretrained(
        self,
        model_type: str,
        num_labels: int = 2,
        custom_model_path: Optional[str] = None
    ) -> Tuple[Any, Any]:
        """
        Load a pre-trained transformer model.

        Args:
            model_type: Type of model ('bert', 'distilbert', 'roberta')
            num_labels: Number of classification labels
            custom_model_path: Path to custom/fine-tuned model

        Returns:
            Tuple of (tokenizer, model)
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_type]
        model_name = custom_model_path or config['model_name']

        logger.info(f"Loading {model_type} from {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        model = model.to(self.device)

        self.tokenizers[model_type] = tokenizer
        self.models[model_type] = model

        return tokenizer, model

    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        model_type: str,
        max_length: int = 256,
        batch_size: int = 16
    ) -> DataLoader:
        """
        Prepare data for training/evaluation.

        Args:
            texts: List of texts
            labels: List of labels
            model_type: Model type (for tokenizer selection)
            max_length: Maximum sequence length
            batch_size: Batch size

        Returns:
            DataLoader
        """
        if model_type not in self.tokenizers:
            self.load_pretrained(model_type)

        dataset = PhishingDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizers[model_type],
            max_length=max_length
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        model_type: str,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 256,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        early_stopping_patience: int = 3,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Fine-tune a transformer model.

        Args:
            model_type: Type of model
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            early_stopping_patience: Early stopping patience
            output_dir: Output directory for checkpoints

        Returns:
            Training history
        """
        if output_dir is None:
            output_dir = str(self.model_dir / f"{model_type}_finetuned")

        # Load model if not already loaded
        if model_type not in self.models:
            self.load_pretrained(model_type)

        tokenizer = self.tokenizers[model_type]
        model = self.models[model_type]

        # Prepare datasets
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )

        train_data = {
            'text': train_texts,
            'label': train_labels
        }
        train_dataset = HFDataset.from_dict(train_data)
        train_dataset = train_dataset.map(tokenize_function, batched=True)

        eval_dataset = None
        if val_texts and val_labels:
            val_data = {
                'text': val_texts,
                'label': val_labels
            }
            eval_dataset = HFDataset.from_dict(val_data)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            save_total_limit=2,
            report_to="none"
        )

        # Metrics function
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary'
            )
            accuracy = accuracy_score(labels, predictions)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Initialize trainer
        callbacks = []
        if eval_dataset and early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )

        # Train
        logger.info(f"Starting fine-tuning of {model_type}...")
        train_result = trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Update stored model
        self.models[model_type] = trainer.model

        # Store history
        history = {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics.get('train_runtime'),
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second')
        }

        if eval_dataset:
            eval_result = trainer.evaluate()
            history['eval_loss'] = eval_result.get('eval_loss')
            history['eval_accuracy'] = eval_result.get('eval_accuracy')
            history['eval_f1'] = eval_result.get('eval_f1')

        self.training_histories[model_type] = history
        logger.info(f"Fine-tuning complete. Model saved to {output_dir}")

        return history

    def predict(
        self,
        model_type: str,
        texts: List[str],
        max_length: int = 256,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions with a fine-tuned model.

        Args:
            model_type: Type of model
            texts: Texts to classify
            max_length: Maximum sequence length
            batch_size: Batch size

        Returns:
            Predicted labels
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")

        model = self.models[model_type]
        tokenizer = self.tokenizers[model_type]
        model.eval()

        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                batch_preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(batch_preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(
        self,
        model_type: str,
        texts: List[str],
        max_length: int = 256,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            model_type: Type of model
            texts: Texts to classify
            max_length: Maximum sequence length
            batch_size: Batch size

        Returns:
            Probability predictions
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")

        model = self.models[model_type]
        tokenizer = self.tokenizers[model_type]
        model.eval()

        probabilities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                probs = torch.softmax(outputs.logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def save_model(self, model_type: str, path: Optional[str] = None):
        """
        Save a fine-tuned model.

        Args:
            model_type: Type of model
            path: Save path
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")

        if path is None:
            path = str(self.model_dir / f"{model_type}_saved")

        self.models[model_type].save_pretrained(path)
        self.tokenizers[model_type].save_pretrained(path)

        logger.info(f"Saved {model_type} to {path}")

    def load_model(self, model_type: str, path: str):
        """
        Load a fine-tuned model.

        Args:
            model_type: Type of model (for registry)
            path: Path to saved model
        """
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model = model.to(self.device)

        self.tokenizers[model_type] = tokenizer
        self.models[model_type] = model

        logger.info(f"Loaded model from {path}")

    def get_attention_weights(
        self,
        model_type: str,
        text: str,
        max_length: int = 256
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get attention weights for interpretability.

        Args:
            model_type: Type of model
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Tuple of (tokens, attention_weights)
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")

        model = self.models[model_type]
        tokenizer = self.tokenizers[model_type]
        model.eval()

        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding, output_attentions=True)

        # Get attention from last layer
        attention = outputs.attentions[-1]  # (batch, heads, seq, seq)
        attention = attention.mean(dim=1).squeeze()  # Average over heads

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze())

        # Get CLS token attention (first token attending to all others)
        cls_attention = attention[0].cpu().numpy()

        return tokens, cls_attention


def get_available_models() -> List[str]:
    """Get list of available transformer models."""
    return list(MODEL_CONFIGS.keys())


if __name__ == "__main__":
    if TRANSFORMERS_AVAILABLE:
        print("Available transformer models:")
        for model_type, config in MODEL_CONFIGS.items():
            print(f"  - {model_type}: {config['model_name']}")

        # Example usage (requires significant resources)
        sample_texts = [
            "Your account has been suspended. Click here to verify.",
            "Meeting tomorrow at 3pm. Please confirm attendance.",
            "You won $1000000! Claim your prize now!",
            "Quarterly report attached for your review."
        ]
        sample_labels = [1, 0, 1, 0]

        print("\nTo fine-tune a model:")
        print("  transformer_models = TransformerModels()")
        print("  transformer_models.load_pretrained('distilbert')")
        print("  transformer_models.train('distilbert', texts, labels)")
    else:
        print("Transformers library not available. Install with:")
        print("  pip install transformers datasets")
