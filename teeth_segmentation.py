"""
Generate training data from 3DTeethSeg dataset.

This script loads multiple dental scans and generates training samples
for neural network training. It creates (image_patch, label) pairs where:
- image_patch: 2D projection of local mesh surface (I(p_i))
- label: binary classification - 0=gingiva, 1=tooth (O(p_i))

Usage:
    python teeth_segmentation.py
"""

import os
import sys
import numpy as np
from dataset_sampler import DatasetSampler


# Configuration
DATASET_ROOT = "."  # Current directory (where data_part folders are)
N_SCANS_TO_PROCESS = 10  # Number of scans to process
SAMPLES_PER_SCAN = 100  # Number of samples per scan
OUTPUT_FILE = "training_data.pkl"


def main():
    """Generate training data from multiple scans."""
    
    print("=" * 60)
    print("3DTeethSeg Training Data Generation")
    print("=" * 60)
    
    # Initialize sampler
    print(f"\nInitializing dataset sampler...")
    print(f"Dataset root: {os.path.abspath(DATASET_ROOT)}")
    
    try:
        sampler = DatasetSampler(
            dataset_root=DATASET_ROOT,
            patch_size=100,
            patch_radius=5.0
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the dataset is in the current directory:")
        print("  data_part_1/, data_part_2/, etc.")
        return
    
    if len(sampler.scan_files) == 0:
        print("\n❌ No scans found!")
        print("\nRun verify_dataset_structure.py to check your dataset.")
        return
    
    print(f"Found {len(sampler.scan_files)} total scans")
    
    # Determine how many scans to process
    n_scans = min(N_SCANS_TO_PROCESS, len(sampler.scan_files))
    print(f"\nProcessing {n_scans} scans...")
    print(f"Samples per scan: {SAMPLES_PER_SCAN}")
    print(f"Total samples: {n_scans * SAMPLES_PER_SCAN}")
    
    # Collect all training samples
    all_samples = []
    
    for scan_idx in range(n_scans):
        print(f"\n[{scan_idx + 1}/{n_scans}] Processing scan...")
        
        # Load scan
        if not sampler.load_scan(scan_idx):
            print("  Skipping scan (failed to load)")
            continue
        
        try:
            # Sample points
            sample_indices = sampler.sample_points(
                n_samples=SAMPLES_PER_SCAN,
                balance_classes=True
            )
            
            # Generate training samples
            samples = sampler.generate_samples(sample_indices)
            all_samples.extend(samples)
            
            print(f"  ✓ Generated {len(samples)} samples")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if not all_samples:
        print("\n❌ No samples were generated!")
        return
    
    # Save all training data
    print(f"\n{'=' * 60}")
    print("Saving training data...")
    sampler.save_training_data(all_samples, OUTPUT_FILE)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total samples: {len(all_samples)}")
    
    gingiva_count = sum(1 for s in all_samples if s.label == 0)
    tooth_count = len(all_samples) - gingiva_count
    
    print(f"  Gingiva: {gingiva_count} ({gingiva_count/len(all_samples)*100:.1f}%)")
    print(f"  Teeth: {tooth_count} ({tooth_count/len(all_samples)*100:.1f}%)")
    
    patients = set(s.patient_id for s in all_samples)
    print(f"  Unique patients: {len(patients)}")
    
    jaws = [s.jaw for s in all_samples]
    upper_count = jaws.count('upper')
    lower_count = jaws.count('lower')
    print(f"  Upper jaw: {upper_count}, Lower jaw: {lower_count}")
    
    print(f"\n✓ Training data saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    
    # Show how to load the data
    print("\nTo load this data for training:")
    print("  import pickle")
    print("  with open('training_data.pkl', 'rb') as f:")
    print("      data = pickle.load(f)")
    print("  X_train = data['images']  # Shape: (n_samples, 100, 100, 3)")
    print("  y_train = data['labels']  # Shape: (n_samples,)")


if __name__ == "__main__":
    main()