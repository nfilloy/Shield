"""
Deep Learning models for phishing/smishing detection.

Implements neural network models:
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)
- CNN 1D (Convolutional Neural Network)
- BiLSTM with Attention
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


if TORCH_AVAILABLE:
    class TextDataset(Dataset):
        """Custom Dataset for text data."""

        def __init__(self, texts: np.ndarray, labels: np.ndarray):
            self.texts = torch.LongTensor(texts)
            self.labels = torch.FloatTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    class LSTMClassifier(nn.Module):
        """LSTM-based text classifier."""

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = False,
            pretrained_embeddings: Optional[np.ndarray] = None
        ):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
                self.embedding.weight.requires_grad = True  # Fine-tune embeddings

            # LSTM layer
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            # Fully connected layers
            fc_input_dim = hidden_dim * self.num_directions
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # x: (batch_size, seq_len)
            embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

            # LSTM
            lstm_out, (hidden, cell) = self.lstm(embedded)
            # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)

            # Use last hidden state
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                hidden = hidden[-1, :, :]

            # Fully connected
            output = self.fc(hidden)
            return output.squeeze()

    class CNNClassifier(nn.Module):
        """1D CNN-based text classifier."""

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            num_filters: int = 128,
            filter_sizes: List[int] = None,
            dropout: float = 0.3,
            pretrained_embeddings: Optional[np.ndarray] = None
        ):
            super().__init__()

            if filter_sizes is None:
                filter_sizes = [2, 3, 4, 5]

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))

            # Convolutional layers with different filter sizes
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
                for fs in filter_sizes
            ])

            self.dropout = nn.Dropout(dropout)

            # Fully connected layer
            self.fc = nn.Sequential(
                nn.Linear(num_filters * len(filter_sizes), 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # x: (batch_size, seq_len)
            embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
            embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

            # Apply convolutions and max pooling
            conv_outputs = []
            for conv in self.convs:
                conv_out = torch.relu(conv(embedded))  # (batch_size, num_filters, seq_len - fs + 1)
                pooled = torch.max(conv_out, dim=2)[0]  # (batch_size, num_filters)
                conv_outputs.append(pooled)

            # Concatenate all pooled outputs
            cat_output = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
            cat_output = self.dropout(cat_output)

            # Fully connected
            output = self.fc(cat_output)
            return output.squeeze()

    class AttentionLayer(nn.Module):
        """Self-attention layer."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attention = nn.Linear(hidden_dim, 1)

        def forward(self, lstm_output):
            # lstm_output: (batch_size, seq_len, hidden_dim)
            attention_weights = torch.softmax(
                self.attention(lstm_output).squeeze(-1), dim=1
            )  # (batch_size, seq_len)

            # Weighted sum
            weighted = torch.bmm(
                attention_weights.unsqueeze(1), lstm_output
            ).squeeze(1)  # (batch_size, hidden_dim)

            return weighted, attention_weights

    class BiLSTMAttentionClassifier(nn.Module):
        """BiLSTM with Attention mechanism."""

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3,
            pretrained_embeddings: Optional[np.ndarray] = None
        ):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))

            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

            self.attention = AttentionLayer(hidden_dim * 2)

            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)

            # Apply attention
            attended, attention_weights = self.attention(lstm_out)

            # Fully connected
            output = self.fc(attended)
            return output.squeeze()

        def forward_with_attention(self, x):
            """Forward pass returning attention weights for interpretability."""
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            attended, attention_weights = self.attention(lstm_out)
            output = self.fc(attended)
            return output.squeeze(), attention_weights


class DeepLearningModels:
    """
    Manager class for deep learning models.

    Provides training, evaluation, and inference utilities
    for LSTM, CNN, and attention-based models.
    """

    def __init__(
        self,
        model_dir: str = "models",
        device: Optional[str] = None
    ):
        """
        Initialize DeepLearningModels.

        Args:
            model_dir: Directory for saving models
            device: Device to use ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.models: Dict[str, nn.Module] = {}
        self.histories: Dict[str, Dict] = {}

    def create_model(
        self,
        model_type: str,
        vocab_size: int,
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance.

        Args:
            model_type: Type of model ('lstm', 'bilstm', 'cnn', 'bilstm_attention')
            vocab_size: Vocabulary size for embedding
            **kwargs: Additional model parameters

        Returns:
            Model instance
        """
        if model_type == 'lstm':
            model = LSTMClassifier(vocab_size, bidirectional=False, **kwargs)
        elif model_type == 'bilstm':
            model = LSTMClassifier(vocab_size, bidirectional=True, **kwargs)
        elif model_type == 'cnn':
            model = CNNClassifier(vocab_size, **kwargs)
        elif model_type == 'bilstm_attention':
            model = BiLSTMAttentionClassifier(vocab_size, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model.to(self.device)

    def train(
        self,
        model_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 3,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Train a deep learning model.

        Args:
            model_name: Name for the model
            model: Model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Patience for early stopping
            class_weights: Class weights for imbalanced data

        Returns:
            Training history
        """
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if class_weights is not None:
            criterion = nn.BCELoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.BCELoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        predictions = (outputs > 0.5).float()
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)

                scheduler.step(avg_val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        self.models[model_name] = model
        self.histories[model_name] = history

        return history

    def predict(
        self,
        model_name: str,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions with a trained model.

        Args:
            model_name: Name of the model
            X: Input data
            batch_size: Batch size for prediction

        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        model.eval()

        dataset = TensorDataset(torch.LongTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size)

        predictions = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def predict_proba(
        self,
        model_name: str,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            model_name: Name of the model
            X: Input data
            batch_size: Batch size

        Returns:
            Probability predictions (2D array with shape [n_samples, 2])
        """
        probs = self.predict(model_name, X, batch_size)
        return np.column_stack([1 - probs, probs])

    def save_model(self, model_name: str, filename: Optional[str] = None):
        """Save a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if filename is None:
            filename = f"{model_name}.pt"

        filepath = self.model_dir / filename
        torch.save({
            'model_state_dict': self.models[model_name].state_dict(),
            'model_class': type(self.models[model_name]).__name__,
            'history': self.histories.get(model_name, {})
        }, filepath)

        logger.info(f"Saved {model_name} to {filepath}")

    def load_model(
        self,
        model_name: str,
        model: nn.Module,
        filename: Optional[str] = None
    ):
        """Load a trained model."""
        if filename is None:
            filename = f"{model_name}.pt"

        filepath = self.model_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        self.models[model_name] = model
        self.histories[model_name] = checkpoint.get('history', {})

        logger.info(f"Loaded {model_name} from {filepath}")


class TextTokenizer:
    """Simple text tokenizer for deep learning models."""

    def __init__(
        self,
        max_vocab_size: int = 10000,
        max_seq_length: int = 256,
        oov_token: str = '<OOV>',
        pad_token: str = '<PAD>'
    ):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.oov_token = oov_token
        self.pad_token = pad_token

        self.word_to_idx = {pad_token: 0, oov_token: 1}
        self.idx_to_word = {0: pad_token, 1: oov_token}
        self.word_counts = {}
        self._is_fitted = False

    def fit(self, texts: List[str]) -> 'TextTokenizer':
        """Build vocabulary from texts."""
        for text in texts:
            for word in text.lower().split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Sort by frequency and keep top words
        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_vocab_size - 2]  # Reserve for PAD and OOV

        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to sequences of indices."""
        if not self._is_fitted:
            raise ValueError("Tokenizer not fitted")

        sequences = []
        for text in texts:
            words = text.lower().split()
            seq = [
                self.word_to_idx.get(word, 1)  # 1 is OOV
                for word in words[:self.max_seq_length]
            ]
            # Pad sequence
            if len(seq) < self.max_seq_length:
                seq.extend([0] * (self.max_seq_length - len(seq)))
            sequences.append(seq)

        return np.array(sequences)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_idx)


if __name__ == "__main__" and TORCH_AVAILABLE:
    # Example usage
    sample_texts = [
        "urgent verify your account now click here",
        "meeting tomorrow at office please confirm",
        "you won a prize claim your reward",
        "project update report attached",
        "your bank account suspended verify immediately"
    ]
    sample_labels = [1, 0, 1, 0, 1]

    # Tokenize
    tokenizer = TextTokenizer(max_vocab_size=100, max_seq_length=20)
    X = tokenizer.fit_transform(sample_texts)
    y = np.array(sample_labels)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Input shape: {X.shape}")

    # Create and train model
    dl_models = DeepLearningModels()
    model = dl_models.create_model(
        'bilstm',
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1
    )

    print(f"\nModel architecture:\n{model}")
