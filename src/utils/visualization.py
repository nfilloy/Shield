"""
Visualization utilities for phishing/smishing detection project.

Provides plotting functions for:
- Model performance metrics
- Confusion matrices
- ROC and PR curves
- Feature importance
- Dataset analysis
- Training history
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available")


class Visualizer:
    """
    Visualization toolkit for phishing/smishing detection analysis.

    Supports both static (matplotlib) and interactive (plotly) visualizations.
    """

    # Color schemes
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#28A745',
        'danger': '#DC3545',
        'warning': '#FFC107',
        'info': '#17A2B8',
        'legitimate': '#28A745',
        'phishing': '#DC3545',
        'smishing': '#FD7E14'
    }

    def __init__(
        self,
        output_dir: str = "reports/figures",
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100
    ):
        """
        Initialize the Visualizer.

        Args:
            output_dir: Directory for saving figures
            style: Matplotlib style
            figsize: Default figure size
            dpi: Figure resolution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figsize = figsize
        self.dpi = dpi

        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except:
                plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str] = None,
        title: str = "Confusion Matrix",
        normalize: bool = False,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix array
            labels: Class labels
            title: Plot title
            normalize: Normalize values
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for this visualization")
            return None

        if labels is None:
            labels = ['Legitimate', 'Phishing/Smishing']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={'size': 14}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {self.output_dir / save_path}")

        if show:
            plt.show()

        return fig

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        model_name: str = "Model",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot ROC curve.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            model_name: Model name for legend
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for this visualization")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(fpr, tpr, color=self.COLORS['primary'], lw=2,
                label=f'{model_name} (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
                label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_roc_curves_comparison(
        self,
        models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        title: str = "ROC Curve Comparison",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot multiple ROC curves for comparison.

        Args:
            models_data: Dict of {model_name: (fpr, tpr, auc)}
            title: Plot title
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.Set2(np.linspace(0, 1, len(models_data)))

        for (model_name, (fpr, tpr, auc)), color in zip(models_data.items(), colors):
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_name} (AUC = {auc:.4f})')

        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        ap_score: float,
        model_name: str = "Model",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot Precision-Recall curve.

        Args:
            precision: Precision values
            recall: Recall values
            ap_score: Average precision score
            model_name: Model name for legend
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(recall, precision, color=self.COLORS['secondary'], lw=2,
                label=f'{model_name} (AP = {ap_score:.4f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_metrics_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot bar chart comparing model metrics.

        Args:
            comparison_df: DataFrame with model metrics
            metrics: Metrics to include
            title: Plot title
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        # Filter to available metrics
        metrics = [m for m in metrics if m in comparison_df.columns]

        df_plot = comparison_df.melt(
            id_vars=['model'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=df_plot,
            x='model',
            y='Score',
            hue='Metric',
            ax=ax
        )

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot feature importance bar chart.

        Args:
            feature_names: Feature names
            importances: Importance scores
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Sort and select top N
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_features)))[::-1]

        ax.barh(range(len(top_features)), top_importances, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot training history (loss and accuracy curves).

        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            title: Plot title
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Training Loss',
                         color=self.COLORS['primary'])
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss',
                         color=self.COLORS['secondary'])

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Training Accuracy',
                         color=self.COLORS['primary'])
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Validation Accuracy',
                         color=self.COLORS['secondary'])

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_class_distribution(
        self,
        labels: np.ndarray,
        class_names: List[str] = None,
        title: str = "Class Distribution",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot class distribution pie/bar chart.

        Args:
            labels: Array of labels
            class_names: Names for each class
            title: Plot title
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if class_names is None:
            class_names = ['Legitimate', 'Phishing/Smishing']

        unique, counts = np.unique(labels, return_counts=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        colors = [self.COLORS['legitimate'], self.COLORS['phishing']]
        axes[0].pie(counts, labels=class_names, autopct='%1.1f%%',
                    colors=colors, startangle=90)
        axes[0].set_title('Class Distribution (Pie)')

        # Bar chart
        axes[1].bar(class_names, counts, color=colors)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Class Distribution (Bar)')

        for i, (c, count) in enumerate(zip(class_names, counts)):
            axes[1].text(i, count + 50, str(count), ha='center', fontsize=12)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_text_length_distribution(
        self,
        texts: List[str],
        labels: np.ndarray,
        class_names: List[str] = None,
        title: str = "Text Length Distribution",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot distribution of text lengths by class.

        Args:
            texts: List of texts
            labels: Array of labels
            class_names: Names for each class
            title: Plot title
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if class_names is None:
            class_names = ['Legitimate', 'Phishing/Smishing']

        lengths = [len(text) for text in texts]
        df = pd.DataFrame({'length': lengths, 'label': labels})

        fig, ax = plt.subplots(figsize=self.figsize)

        for label, name, color in zip([0, 1], class_names,
                                       [self.COLORS['legitimate'], self.COLORS['phishing']]):
            subset = df[df['label'] == label]['length']
            ax.hist(subset, bins=50, alpha=0.6, label=name, color=color)

        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_wordcloud(
        self,
        texts: List[str],
        title: str = "Word Cloud",
        max_words: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Generate and plot word cloud.

        Args:
            texts: List of texts
            title: Plot title
            max_words: Maximum number of words
            save_path: Path to save figure
            show: Display the plot

        Returns:
            Matplotlib figure
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("WordCloud not available. Install with: pip install wordcloud")
            return None

        if not MATPLOTLIB_AVAILABLE:
            return None

        text = ' '.join(texts)

        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color='white',
            colormap='viridis'
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def create_interactive_metrics_plot(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None
    ):
        """
        Create interactive metrics comparison using Plotly.

        Args:
            comparison_df: DataFrame with model metrics
            metrics: Metrics to include

        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly required for interactive plots")
            return None

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        metrics = [m for m in metrics if m in comparison_df.columns]

        fig = go.Figure()

        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=comparison_df['model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(4),
                textposition='auto'
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis_range=[0, 1],
            legend_title='Metrics'
        )

        return fig


if __name__ == "__main__":
    # Example usage
    visualizer = Visualizer()

    # Sample confusion matrix
    cm = np.array([[85, 15], [10, 90]])
    visualizer.plot_confusion_matrix(cm, title="Sample Confusion Matrix", show=True)

    # Sample ROC curve
    fpr = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    tpr = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    visualizer.plot_roc_curve(fpr, tpr, auc_score=0.92, model_name="Test Model", show=True)

    print("Visualizations complete!")
