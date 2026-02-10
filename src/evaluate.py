"""Model Evaluation Script for TeethIdentifier."""

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import TeethDataLoader


class TeethEvaluator:
    """Evaluation pipeline for TeethNet CNN."""

    def __init__(self, config_path: str = 'config.yaml', model_path: str = None):
        self.config = self.load_config(config_path)
        self.results_dir = Path(self.config['paths'].get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'sample_predictions').mkdir(parents=True, exist_ok=True)

        if model_path is None:
            model_path = Path(self.config['paths']['model_dir']) / 'teeth_classifier.keras'
        self.model_path = model_path
        self.model = None

    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self) -> keras.Model:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        return self.model

    def load_test_data(self):
        loader = TeethDataLoader(self.config)
        images, labels, metadata = loader.load_test_data()
        return images, labels, metadata

    def predict(self, images: np.ndarray):
        batch_size = self.config['inference'].get('batch_size', 128)
        y_pred_proba = self.model.predict(images, batch_size=batch_size, verbose=1)
        threshold = self.config['evaluation'].get('threshold', 0.5)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        return y_pred, y_pred_proba.flatten()

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> dict:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        return metrics

    def print_metrics(self, metrics: dict) -> None:
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%, Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"Confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    def save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   metrics: dict) -> None:
        report_path = self.results_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("TeethIdentifier Classification Report\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
            f.write(classification_report(y_true, y_pred, target_names=['Gingiva', 'Tooth'], digits=4))

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Gingiva', 'Tooth'], yticklabels=['Gingiva', 'Tooth'],
                    annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_per_patient(self, y_true: np.ndarray, y_pred: np.ndarray,
                           patient_ids: list) -> None:
        if not patient_ids:
            return
        patient_results = {}
        for pid, true, pred in zip(patient_ids, y_true, y_pred):
            if pid not in patient_results:
                patient_results[pid] = {'true': [], 'pred': []}
            patient_results[pid]['true'].append(true)
            patient_results[pid]['pred'].append(pred)

        patient_data = []
        for pid, data in patient_results.items():
            true = np.array(data['true'])
            pred = np.array(data['pred'])
            patient_data.append({
                'patient_id': pid, 'num_samples': len(true),
                'accuracy': accuracy_score(true, pred),
                'precision': precision_score(true, pred, zero_division=0),
                'recall': recall_score(true, pred, zero_division=0),
                'f1_score': f1_score(true, pred, zero_division=0)
            })

        df = pd.DataFrame(patient_data).sort_values('accuracy', ascending=False)
        df.to_csv(self.results_dir / 'per_patient_analysis.csv', index=False)
        print(f"Per-patient: {len(patient_results)} patients, mean_acc={df['accuracy'].mean():.4f}")

    def save_sample_predictions(self, images: np.ndarray, y_true: np.ndarray,
                                y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                n_samples: int = 20) -> None:
        indices = np.random.choice(len(y_true), size=min(n_samples, len(y_true)), replace=False)
        n_cols = 5
        n_rows = (len(indices) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            ax.imshow(images[idx])
            true_label = "Tooth" if y_true[idx] == 1 else "Gingiva"
            pred_label = "Tooth" if y_pred[idx] == 1 else "Gingiva"
            confidence = y_pred_proba[idx] if y_pred[idx] == 1 else (1 - y_pred_proba[idx])
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.0%})",
                        color=color, fontsize=9, fontweight='bold')
            ax.axis('off')

        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_predictions' / 'predictions_grid.png', dpi=150, bbox_inches='tight')
        plt.close()

    def run(self) -> int:
        try:
            self.load_model()
            images, y_true, metadata = self.load_test_data()
            print(f"Evaluating on {len(y_true):,} test samples")

            y_pred, y_pred_proba = self.predict(images)
            metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
            self.print_metrics(metrics)

            self.save_classification_report(y_true, y_pred, metrics)
            self.plot_confusion_matrix(metrics['confusion_matrix'])
            self.plot_roc_curve(y_true, y_pred_proba)

            if metadata.get('patient_ids'):
                self.analyze_per_patient(y_true, y_pred, metadata['patient_ids'])

            n_samples = self.config['evaluation'].get('num_sample_predictions', 20)
            self.save_sample_predictions(images, y_true, y_pred, y_pred_proba, n_samples)

            print(f"Results saved to: {self.results_dir}")
            return 0

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    evaluator = TeethEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(main())
