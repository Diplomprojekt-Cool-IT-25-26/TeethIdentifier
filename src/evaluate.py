"""
Model Evaluation Script for TeethIdentifier.

This script provides comprehensive evaluation of the trained model:
- Accuracy, precision, recall, F1-score, AUC
- Confusion matrix visualization
- ROC curve
- Per-patient analysis
- Sample predictions

Usage:
    python src/evaluate.py

Expected output:
    results/classification_report.txt
    results/confusion_matrix.png
    results/roc_curve.png
    results/per_patient_analysis.csv
    results/sample_predictions/
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import TeethDataLoader


class TeethEvaluator:
    """
    Evaluation pipeline for TeethNet CNN.

    Computes metrics, generates visualizations, and analyzes per-patient performance.
    """

    def __init__(self, config_path: str = 'config.yaml',
                 model_path: str = None):
        """
        Initialize the evaluator.

        Args:
            config_path: Path to configuration YAML file
            model_path: Path to trained model (default: models/teeth_classifier.keras)
        """
        self.config = self.load_config(config_path)

        # Paths
        self.results_dir = Path(self.config['paths'].get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'sample_predictions').mkdir(parents=True, exist_ok=True)

        # Model
        if model_path is None:
            model_path = Path(self.config['paths']['model_dir']) / 'teeth_classifier.keras'
        self.model_path = model_path
        self.model = None

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self) -> keras.Model:
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Train the model first: python src/train.py"
            )

        self.model = keras.models.load_model(self.model_path)
        print(f"[OK] Model loaded successfully")

        return self.model

    def load_test_data(self):
        """
        Load test dataset.

        Returns:
            Tuple of (images, labels, metadata)
        """
        print("\nLoading test data...")
        loader = TeethDataLoader(self.config)
        images, labels, metadata = loader.load_test_data()
        print(f"[OK] Loaded {len(labels):,} test samples")

        return images, labels, metadata

    def predict(self, images: np.ndarray):
        """
        Run predictions on test images.

        Args:
            images: Test images array (n_samples, 100, 100, 3)

        Returns:
            Tuple of (predicted_labels, predicted_probabilities)
        """
        print("\nRunning predictions...")

        # Predict in batches
        batch_size = self.config['inference'].get('batch_size', 128)
        y_pred_proba = self.model.predict(images, batch_size=batch_size, verbose=1)

        # Convert probabilities to binary predictions
        threshold = self.config['evaluation'].get('threshold', 0.5)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()

        print(f"[OK] Predictions complete")

        return y_pred, y_pred_proba

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> dict:
        """
        Calculate evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        print("\nCalculating metrics...")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Per-class metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        print("[OK] Metrics calculated")

        return metrics

    def print_metrics(self, metrics: dict) -> None:
        """Print metrics to console."""
        print("\n" + "=" * 70)
        print("Evaluation Metrics")
        print("=" * 70)
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"                Predicted")
        print(f"              Gingiva  Tooth")
        print(f"  Actual Gingiva  {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"         Tooth    {cm[1,0]:6d}  {cm[1,1]:6d}")

        print("\nDetailed Breakdown:")
        print(f"  True Negatives:  {metrics['true_negatives']:,}")
        print(f"  False Positives: {metrics['false_positives']:,}")
        print(f"  False Negatives: {metrics['false_negatives']:,}")
        print(f"  True Positives:  {metrics['true_positives']:,}")

        print("=" * 70)

    def save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   metrics: dict) -> None:
        """Save detailed classification report to file."""
        report_path = self.results_dir / 'classification_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TeethIdentifier Classification Report\n")
            f.write("=" * 70 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")

            f.write("Confusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(f"                Predicted\n")
            f.write(f"              Gingiva  Tooth\n")
            f.write(f"  Actual Gingiva  {cm[0,0]:6d}  {cm[0,1]:6d}\n")
            f.write(f"         Tooth    {cm[1,0]:6d}  {cm[1,1]:6d}\n\n")

            f.write("Scikit-learn Classification Report:\n")
            f.write(classification_report(
                y_true, y_pred,
                target_names=['Gingiva', 'Tooth'],
                digits=4
            ))

        print(f"[OK] Classification report saved to: {report_path}")

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot and save confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Gingiva', 'Tooth'],
            yticklabels=['Gingiva', 'Tooth'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 14, 'fontweight': 'bold'}
        )

        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - Teeth vs Gingiva Classification',
                 fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        cm_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved to: {cm_path}")

        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Plot and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Teeth vs Gingiva Classification',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        roc_path = self.results_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        print(f"[OK] ROC curve saved to: {roc_path}")

        plt.close()

    def analyze_per_patient(self, y_true: np.ndarray, y_pred: np.ndarray,
                           patient_ids: list) -> None:
        """Analyze performance per patient."""
        if not patient_ids:
            print("⚠ No patient IDs available for per-patient analysis")
            return

        print("\nAnalyzing per-patient performance...")

        # Group by patient
        patient_results = {}
        for pid, true, pred in zip(patient_ids, y_true, y_pred):
            if pid not in patient_results:
                patient_results[pid] = {'true': [], 'pred': []}
            patient_results[pid]['true'].append(true)
            patient_results[pid]['pred'].append(pred)

        # Calculate per-patient metrics
        patient_data = []
        for pid, data in patient_results.items():
            true = np.array(data['true'])
            pred = np.array(data['pred'])

            patient_data.append({
                'patient_id': pid,
                'num_samples': len(true),
                'accuracy': accuracy_score(true, pred),
                'precision': precision_score(true, pred, zero_division=0),
                'recall': recall_score(true, pred, zero_division=0),
                'f1_score': f1_score(true, pred, zero_division=0)
            })

        # Create DataFrame
        df = pd.DataFrame(patient_data)
        df = df.sort_values('accuracy', ascending=False)

        # Save to CSV
        csv_path = self.results_dir / 'per_patient_analysis.csv'
        df.to_csv(csv_path, index=False)
        print(f"[OK] Per-patient analysis saved to: {csv_path}")

        # Print summary
        print("\nPer-Patient Summary:")
        print(f"  Total patients: {len(patient_results)}")
        print(f"  Mean accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"  Min accuracy: {df['accuracy'].min():.4f} (Patient: {df.iloc[-1]['patient_id']})")
        print(f"  Max accuracy: {df['accuracy'].max():.4f} (Patient: {df.iloc[0]['patient_id']})")

    def save_sample_predictions(self, images: np.ndarray, y_true: np.ndarray,
                                y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                n_samples: int = 20) -> None:
        """Save sample predictions with images."""
        print(f"\nSaving {n_samples} sample predictions...")

        # Select random samples
        indices = np.random.choice(len(y_true), size=min(n_samples, len(y_true)),
                                  replace=False)

        # Create grid
        n_cols = 5
        n_rows = (len(indices) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]

            # Display image
            ax.imshow(images[idx])

            # Title with prediction
            true_label = "Tooth" if y_true[idx] == 1 else "Gingiva"
            pred_label = "Tooth" if y_pred[idx] == 1 else "Gingiva"
            confidence = y_pred_proba[idx] if y_pred[idx] == 1 else (1 - y_pred_proba[idx])

            # Color based on correctness
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'

            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2%})",
                        color=color, fontsize=9, fontweight='bold')
            ax.axis('off')

        # Hide extra subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        samples_path = self.results_dir / 'sample_predictions' / 'predictions_grid.png'
        plt.savefig(samples_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Sample predictions saved to: {samples_path}")

        plt.close()

    def run(self) -> int:
        """Run the complete evaluation pipeline."""
        print("\n" + "=" * 70)
        print("TeethIdentifier Model Evaluation")
        print("=" * 70 + "\n")

        try:
            # Load model
            self.load_model()

            # Load test data
            images, y_true, metadata = self.load_test_data()

            # Predict
            y_pred, y_pred_proba = self.predict(images)

            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)

            # Print metrics
            self.print_metrics(metrics)

            # Save results
            self.save_classification_report(y_true, y_pred, metrics)
            self.plot_confusion_matrix(metrics['confusion_matrix'])
            self.plot_roc_curve(y_true, y_pred_proba)

            # Per-patient analysis
            if metadata.get('patient_ids'):
                self.analyze_per_patient(y_true, y_pred, metadata['patient_ids'])

            # Sample predictions
            n_samples = self.config['evaluation'].get('num_sample_predictions', 20)
            self.save_sample_predictions(images, y_true, y_pred, y_pred_proba, n_samples)

            # Final summary
            print("\n" + "=" * 70)
            print("Evaluation Complete!")
            print("=" * 70)
            print(f"\nResults saved to: {self.results_dir}")
            print("\nGenerated files:")
            print(f"  - classification_report.txt")
            print(f"  - confusion_matrix.png")
            print(f"  - roc_curve.png")
            print(f"  - per_patient_analysis.csv")
            print(f"  - sample_predictions/predictions_grid.png")
            print("=" * 70 + "\n")

            return 0

        except Exception as e:
            print(f"\n[ERROR] Evaluation failed")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main entry point."""
    evaluator = TeethEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(main())
