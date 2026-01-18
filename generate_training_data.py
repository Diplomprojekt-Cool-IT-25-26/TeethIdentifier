"""
Generate training data for TeethNet neural network.

Creates train/validation/test datasets with 70/15/15 split by patient.
Ensures equal class distribution (balanced gingiva vs teeth samples).
"""

import yaml
import sys
import random
from pathlib import Path
from tqdm import tqdm
from dataset_sampler import DatasetSampler


def split_by_patient(scan_files, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split scans by patient to prevent data leakage."""
    random.seed(seed)

    # Group scans by patient
    patients = {}
    for scan in scan_files:
        patient_id = scan['patient_id']
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(scan)

    # Shuffle patients
    patient_ids = list(patients.keys())
    random.shuffle(patient_ids)

    # Split patients
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]

    # Get scans for each split
    train_scans = [scan for pid in train_patients for scan in patients[pid]]
    val_scans = [scan for pid in val_patients for scan in patients[pid]]
    test_scans = [scan for pid in test_patients for scan in patients[pid]]

    return train_scans, val_scans, test_scans


def main():
    """Generate training, validation, and test datasets."""

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("TeethIdentifier Training Data Generation")
    print("=" * 70)
    print()

    # Initialize dataset sampler
    print("Initializing dataset sampler...")
    sampler = DatasetSampler(
        dataset_root=config['paths']['data_root'],
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius']
    )

    print(f"Found {len(sampler.scan_files)} total scans")

    # Limit number of scans if configured
    max_scans = config['data_generation']['scans_for_training']
    if max_scans and max_scans < len(sampler.scan_files):
        sampler.scan_files = sampler.scan_files[:max_scans]
        print(f"Limiting to {max_scans} scans (configured in config.yaml)")

    print()

    # Split data by patient (prevents data leakage)
    split_ratios = config['data_generation']['split_ratios']
    train_scans, val_scans, test_scans = split_by_patient(
        scan_files=sampler.scan_files,
        train_ratio=split_ratios['train'],
        val_ratio=split_ratios['val'],
        test_ratio=split_ratios['test'],
        seed=config['data_generation']['random_seed']
    )

    # Count unique patients
    train_patients = set(s['patient_id'] for s in train_scans)
    val_patients = set(s['patient_id'] for s in val_scans)
    test_patients = set(s['patient_id'] for s in test_scans)

    print("Patient-level split:")
    print(f"  Train: {len(train_patients)} patients ({len(train_scans)} scans)")
    print(f"  Val: {len(val_patients)} patients ({len(val_scans)} scans)")
    print(f"  Test: {len(test_patients)} patients ({len(test_scans)} scans)")
    print()

    # Verify no patient appears in multiple splits
    assert len(train_patients & val_patients) == 0, "Patient appears in both train and val!"
    assert len(train_patients & test_patients) == 0, "Patient appears in both train and test!"
    assert len(val_patients & test_patients) == 0, "Patient appears in both val and test!"

    # Create output directory
    output_dir = Path(config['paths']['training_data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_per_scan = config['data_generation']['samples_per_scan']

    # Get boundary weighting parameters
    boundary_weighted = config['data_generation'].get('boundary_weighted', False)
    boundary_weight = config['data_generation'].get('boundary_weight', 0.0)
    boundary_radius = config['data_generation'].get('boundary_radius', 5.0)

    # Generate training data for each split
    for split_name, scan_list in [('train', train_scans), ('val', val_scans), ('test', test_scans)]:
        print("=" * 70)
        print(f"Generating {split_name.upper()} Data")
        print("=" * 70)
        print(f"  Scans: {len(scan_list)}")
        print(f"  Samples per scan: {samples_per_scan}")
        if boundary_weighted:
            print(f"  Boundary-weighted: YES ({boundary_weight*100:.0f}% boundary, {(1-boundary_weight)*100:.0f}% clear)")
            print(f"  Boundary radius: {boundary_radius}")
        else:
            print(f"  Boundary-weighted: NO (uniform sampling)")
        print(f"  Expected total: {len(scan_list) * samples_per_scan} samples")
        print()

        all_samples = []

        # Create a temporary list with scan indices
        scan_indices = []
        for i, scan in enumerate(sampler.scan_files):
            if scan in scan_list:
                scan_indices.append(i)

        # Process each scan
        for scan_idx in tqdm(scan_indices, desc=f"Processing {split_name} scans"):
            # Load the scan
            if not sampler.load_scan(scan_idx):
                print(f"\n[WARNING] Failed to load scan {scan_idx}")
                continue

            # Sample points (balanced between gingiva and teeth)
            # Apply boundary weighting if enabled
            sample_point_indices = sampler.sample_points(
                n_samples=samples_per_scan,
                balance_classes=True,
                boundary_weight=boundary_weight if boundary_weighted else 0.0,
                boundary_radius=boundary_radius
            )

            # Generate training samples
            samples = sampler.generate_samples(sample_point_indices)
            all_samples.extend(samples)

        # Save to file
        output_path = output_dir / f"{split_name}.pkl"
        print(f"\nSaving to {output_path}...")
        sampler.save_training_data(all_samples, str(output_path))

        # Statistics
        gingiva_count = sum(1 for s in all_samples if s.label == 0)
        teeth_count = sum(1 for s in all_samples if s.label == 1)

        print(f"\n{split_name.upper()} Statistics:")
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Gingiva: {gingiva_count} ({gingiva_count/len(all_samples)*100:.1f}%)")
        print(f"  Teeth: {teeth_count} ({teeth_count/len(all_samples)*100:.1f}%)")
        print()

    print("=" * 70)
    print("Training Data Generation Complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  {output_dir}/train.pkl")
    print(f"  {output_dir}/val.pkl")
    print(f"  {output_dir}/test.pkl")
    print()
    print("Next step: Run training with train_gpu.bat")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
