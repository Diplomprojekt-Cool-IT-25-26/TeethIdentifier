"""Streamlit Dashboard for TeethIdentifier."""

import streamlit as st
import yaml
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="TeethIdentifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight: bold; }
.metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)


def load_config():
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def show_data_overview():
    st.header("Step 1: Data Overview")

    config = load_config()
    if not config:
        return

    training_data_dir = Path(config['paths']['training_data_dir'])

    st.subheader("Dataset Availability")
    cols = st.columns(3)
    for i, split in enumerate(['train', 'val', 'test']):
        pkl_path = training_data_dir / f'{split}.pkl'
        with cols[i]:
            if pkl_path.exists():
                st.success(f"{split.upper()} dataset found")
            else:
                st.error(f"{split.upper()} dataset not found")

    st.subheader("View Dataset")
    uploaded_file = st.file_uploader("Upload a pickle file to view", type=['pkl'])

    if uploaded_file is not None:
        data = pickle.load(uploaded_file)

        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(data['labels']):,}")
        with col2:
            n_gingiva = np.sum(data['labels'] == 0)
            st.metric("Gingiva Samples", f"{n_gingiva:,}")
        with col3:
            n_teeth = np.sum(data['labels'] == 1)
            st.metric("Tooth Samples", f"{n_teeth:,}")

        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels_unique, counts = np.unique(data['labels'], return_counts=True)
        ax.bar(['Gingiva (0)', 'Tooth (1)'], counts, color=['#ff9999', '#9999ff'])
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        for i, count in enumerate(counts):
            ax.text(i, count + max(counts)*0.02, f'{count:,}', ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig)
        plt.close()

        st.subheader("Sample Patches")
        n_show = st.slider("Number of samples to display", 4, 20, 12)
        indices = np.random.choice(len(data['labels']), min(n_show, len(data['labels'])), replace=False)

        cols_per_row = 4
        n_rows = (len(indices) + cols_per_row - 1) // cols_per_row

        for row in range(n_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx_in_list = row * cols_per_row + col_idx
                if idx_in_list < len(indices):
                    idx = indices[idx_in_list]
                    with cols[col_idx]:
                        label = "Tooth" if data['labels'][idx] == 1 else "Gingiva"
                        st.image(data['images'][idx], caption=label, use_container_width=True)


def show_model_config():
    st.header("Step 2: Model Configuration")

    config = load_config()
    if not config:
        return

    st.subheader("Current Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Settings**")
        st.write(f"- Batch Size: {config['training']['batch_size']}")
        st.write(f"- Epochs: {config['training']['epochs']}")
        st.write(f"- Learning Rate: {config['training']['learning_rate']}")
        st.write(f"- Optimizer: {config['training']['optimizer']}")
        st.write(f"- Loss: {config['training']['loss']}")

    with col2:
        st.markdown("**Model Architecture**")
        st.write(f"- Input Shape: {config['model']['input_shape']}")
        st.write(f"- Conv Filters: {config['model']['conv_filters']}")
        st.write(f"- Dense Units: {config['model']['dense_units']}")
        st.write(f"- Dropout Rates: {config['model']['dropout_rates']}")

    st.subheader("Model Architecture")
    arch_text = """
    Input (100, 100, 3) - RGB patches
    -> Rescaling (normalize to [0, 1])
    -> Block 1: Conv2D(32) -> BatchNorm -> MaxPool -> Dropout(0.25)
    -> Block 2: Conv2D(64) -> BatchNorm -> MaxPool -> Dropout(0.25)
    -> Block 3: Conv2D(128) -> BatchNorm -> MaxPool -> Dropout(0.3)
    -> Flatten
    -> Dense(256) -> BatchNorm -> Dropout(0.5)
    -> Dense(64) -> Dropout(0.4)
    -> Dense(1, Sigmoid)
    -> Output: Binary probability [0, 1]
    """
    st.code(arch_text, language='text')
    st.info("Estimated parameters: ~2.5M (fits in 8GB VRAM)")


def show_training():
    st.header("Step 3: Training")
    st.info("For live training, use: python src/train.py")

    config = load_config()
    if not config:
        return

    history_path = Path(config['paths']['model_dir']) / 'training_history.json'

    if history_path.exists():
        st.success("Training history found")

        with open(history_path, 'r') as f:
            history = json.load(f)

        st.subheader("Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Final Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Accuracy", f"{history['accuracy'][-1]:.4f}")
        with col2:
            st.metric("Val Accuracy", f"{history['val_accuracy'][-1]:.4f}")
        with col3:
            st.metric("Train Loss", f"{history['loss'][-1]:.4f}")
        with col4:
            st.metric("Val Loss", f"{history['val_loss'][-1]:.4f}")
    else:
        st.warning("No training history found. Train the model first:")
        st.code("python src/train.py", language='bash')


def show_evaluation():
    st.header("Step 4: Evaluation Results")

    config = load_config()
    if not config:
        return

    results_dir = Path(config['paths'].get('results_dir', 'results'))
    report_path = results_dir / 'classification_report.txt'
    cm_path = results_dir / 'confusion_matrix.png'
    roc_path = results_dir / 'roc_curve.png'

    if not report_path.exists():
        st.warning("No evaluation results found. Run evaluation first:")
        st.code("python src/evaluate.py", language='bash')
        return

    st.subheader("Classification Report")
    with open(report_path, 'r') as f:
        report_text = f.read()
    st.code(report_text, language='text')

    if cm_path.exists():
        st.subheader("Confusion Matrix")
        st.image(str(cm_path), use_container_width=True)

    if roc_path.exists():
        st.subheader("ROC Curve")
        st.image(str(roc_path), use_container_width=True)

    patient_csv = results_dir / 'per_patient_analysis.csv'
    if patient_csv.exists():
        st.subheader("Per-Patient Analysis")
        df = pd.read_csv(patient_csv)
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Accuracy", f"{df['accuracy'].mean():.4f}")
        with col2:
            st.metric("Min Accuracy", f"{df['accuracy'].min():.4f}")
        with col3:
            st.metric("Max Accuracy", f"{df['accuracy'].max():.4f}")


def show_inference():
    st.header("Step 5: Inference")

    config = load_config()
    if not config:
        return

    model_path = Path(config['paths']['model_dir']) / 'teeth_classifier.keras'

    if not model_path.exists():
        st.error("Trained model not found. Train the model first:")
        st.code("python src/train.py", language='bash')
        return

    st.success(f"Model found: {model_path}")

    st.subheader("Upload OBJ File")
    uploaded_file = st.file_uploader("Choose an OBJ file", type=['obj'])

    if uploaded_file is not None:
        st.info("For inference, use the command line:")
        st.code(f"python src/predict.py --obj {uploaded_file.name} --output exports/predictions", language='bash')

        st.markdown("---")
        st.subheader("Expected Output")
        st.markdown("""
        The inference pipeline will generate a JSON side-car file with:
        ```json
        {
          "name": "PATIENT_001_upper.obj",
          "labels": [0, 1, 1, 0, ...],
          "instances": [],
          "jaw": "upper",
          "prediction_metadata": {
            "model_version": "v1.0",
            "timestamp": "2026-01-05T...",
            "threshold": 0.5
          }
        }
        ```
        """)

        st.markdown("---")
        st.subheader("Batch Processing")
        st.info("To process multiple OBJ files:")
        st.code("""
# Process all OBJ files in a directory
python src/predict.py --obj data/data_part_1/upper/ --output exports/predictions

# Process specific files
python src/predict.py --obj scan1.obj --output exports/
        """, language='bash')


def main():
    st.title("TeethIdentifier - Neural Network Training System")
    st.markdown("**Binary classification of teeth vs gingiva from 3D dental scans**")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Step",
        ["1. Data Overview", "2. Model Configuration", "3. Training", "4. Evaluation", "5. Inference"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Actions")
    if st.sidebar.button("Reload Configuration"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Info")
    config = load_config()
    if config:
        st.sidebar.info(f"**Project:** {config.get('project_name', 'TeethIdentifier')}")
        st.sidebar.info(f"**Version:** {config.get('version', 'v1.0')}")

    if page == "1. Data Overview":
        show_data_overview()
    elif page == "2. Model Configuration":
        show_model_config()
    elif page == "3. Training":
        show_training()
    elif page == "4. Evaluation":
        show_evaluation()
    elif page == "5. Inference":
        show_inference()


if __name__ == "__main__":
    main()
