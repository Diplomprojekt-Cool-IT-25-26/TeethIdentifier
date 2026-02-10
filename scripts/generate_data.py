"""Generate training data for TeethNet neural network."""

import os
import sys
import yaml
import random
from pathlib import Path
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dataset_sampler import DatasetSampler


def split_by_patient(scan_files, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split scans by patient to prevent data leakage."""
    random.seed(seed)

    patients = {}
    for scan in scan_files:
        patient_id = scan['patient_id']
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(scan)

    patient_ids = list(patients.keys())
    random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]

    train_scans = [scan for pid in train_patients for scan in patients[pid]]
    val_scans = [scan for pid in val_patients for scan in patients[pid]]
    test_scans = [scan for pid in test_patients for scan in patients[pid]]

    return train_scans, val_scans, test_scans


def main():
    """Generate training, validation, and test datasets."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Training Data Generation")

    sampler = DatasetSampler(
        dataset_root=config['paths']['data_root'],
        patch_size=config['data_generation']['patch_size'],
        patch_radius=config['data_generation']['patch_radius']
    )

    max_scans = config['data_generation']['scans_for_training']
    if max_scans and max_scans < len(sampler.scan_files):
        sampler.scan_files = sampler.scan_files[:max_scans]
        print(f"Using {max_scans} of {len(sampler.scan_files)} scans")

    split_ratios = config['data_generation']['split_ratios']
    train_scans, val_scans, test_scans = split_by_patient(
        scan_files=sampler.scan_files,
        train_ratio=split_ratios['train'],
        val_ratio=split_ratios['val'],
        test_ratio=split_ratios['test'],
        seed=config['data_generation']['random_seed']
    )

    train_patients = set(s['patient_id'] for s in train_scans)
    val_patients = set(s['patient_id'] for s in val_scans)
    test_patients = set(s['patient_id'] for s in test_scans)

    print(f"Split: train={len(train_patients)} patients ({len(train_scans)} scans), "
          f"val={len(val_patients)} ({len(val_scans)}), test={len(test_patients)} ({len(test_scans)})")

    assert len(train_patients & val_patients) == 0, "Patient in both train and val!"
    assert len(train_patients & test_patients) == 0, "Patient in both train and test!"
    assert len(val_patients & test_patients) == 0, "Patient in both val and test!"

    output_dir = Path(config['paths']['training_data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_per_scan = config['data_generation']['samples_per_scan']
    boundary_weighted = config['data_generation'].get('boundary_weighted', False)
    boundary_weight = config['data_generation'].get('boundary_weight', 0.0)
    boundary_radius = config['data_generation'].get('boundary_radius', 5.0)

    for split_name, scan_list in [('train', train_scans), ('val', val_scans), ('test', test_scans)]:
        print(f"\n{split_name.upper()}: {len(scan_list)} scans, {samples_per_scan} samples/scan"
              + (f", boundary={boundary_weight:.0%}" if boundary_weighted else ""))

        all_samples = []
        scan_indices = [i for i, scan in enumerate(sampler.scan_files) if scan in scan_list]

        for scan_idx in tqdm(scan_indices, desc=split_name):
            if not sampler.load_scan(scan_idx):
                continue

            sample_point_indices = sampler.sample_points(
                n_samples=samples_per_scan,
                balance_classes=True,
                boundary_weight=boundary_weight if boundary_weighted else 0.0,
                boundary_radius=boundary_radius
            )
            samples = sampler.generate_samples(sample_point_indices)
            all_samples.extend(samples)

        output_path = output_dir / f"{split_name}.pkl"
        sampler.save_training_data(all_samples, str(output_path))

        gingiva = sum(1 for s in all_samples if s.label == 0)
        teeth = len(all_samples) - gingiva
        print(f"Saved {len(all_samples)} samples ({gingiva} gingiva, {teeth} teeth) -> {output_path}")

    print(f"\nComplete. Run: scripts/train_gpu.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
