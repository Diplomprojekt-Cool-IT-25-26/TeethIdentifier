"""
Verify 3DTeethSeg dataset structure and contents.

This script checks that the dataset was downloaded and extracted correctly.
It displays basic statistics about the dataset.
"""

import os
import json
import sys
from pathlib import Path


def find_all_scans(dataset_root: str):
    """Find all OBJ and JSON file pairs in the dataset."""
    scans = []
    
    for root, dirs, files in os.walk(dataset_root):
        obj_files = [f for f in files if f.endswith('.obj')]
        
        for obj_file in obj_files:
            obj_path = os.path.join(root, obj_file)
            json_file = obj_file.replace('.obj', '.json')
            json_path = os.path.join(root, json_file)
            
            if os.path.exists(json_path):
                base_name = obj_file.replace('.obj', '')
                parts = base_name.split('_')
                jaw = parts[-1].lower() if len(parts) > 1 else 'unknown'
                patient_id = '_'.join(parts[:-1]) if len(parts) > 1 else base_name
                
                scans.append({
                    'obj_file': obj_path,
                    'json_file': json_path,
                    'patient_id': patient_id,
                    'jaw': jaw,
                    'base_name': base_name
                })
    
    return scans


def check_scan_data(json_path: str):
    """Check if scan has valid data."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        labels = data.get('labels', [])
        instances = data.get('instances', [])
        
        n_vertices = len(labels)
        gingiva = sum(1 for l in labels if l == 0)
        teeth = n_vertices - gingiva
        unique_teeth = len(set(l for l in labels if l != 0))
        
        return {
            'valid': True,
            'vertices': n_vertices,
            'gingiva': gingiva,
            'teeth': teeth,
            'unique_teeth': unique_teeth
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def main():
    """Main verification function."""
    
    # Get dataset path from command line or use current directory
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "data"

    if not os.path.exists(dataset_path):
        print(f"[FAIL] Path does not exist: {dataset_path}")
        return
    
    print("=" * 60)
    print("3DTeethSeg Dataset Verification")
    print("=" * 60)
    print(f"\nSearching: {os.path.abspath(dataset_path)}")
    
    # Find all scans
    scans = find_all_scans(dataset_path)
    
    if not scans:
        print("\n[FAIL] No scans found!")
        print("\nExpected structure:")
        print("  dataset_root/")
        print("  ├── data_part_1/")
        print("  │   ├── upper/")
        print("  │   │   └── PATIENT_ID/")
        print("  │   │       ├── PATIENT_ID_upper.obj")
        print("  │   │       └── PATIENT_ID_upper.json")
        print("  │   └── lower/")
        print("  │       └── PATIENT_ID/")
        print("  │           ├── PATIENT_ID_lower.obj")
        print("  │           └── PATIENT_ID_lower.json")
        return
    
    print(f"\n[OK] Found {len(scans)} scans")
    
    # Count by jaw type
    upper_scans = [s for s in scans if s['jaw'] == 'upper']
    lower_scans = [s for s in scans if s['jaw'] == 'lower']
    
    print(f"  Upper jaw: {len(upper_scans)}")
    print(f"  Lower jaw: {len(lower_scans)}")
    
    # Count unique patients
    patients = set(s['patient_id'] for s in scans)
    print(f"  Unique patients: {len(patients)}")
    
    # Check a few sample scans
    print(f"\nChecking sample scans...")
    
    samples_to_check = min(5, len(scans))
    for i in range(samples_to_check):
        scan = scans[i]
        print(f"\n[{i+1}] {scan['base_name']}")
        
        result = check_scan_data(scan['json_file'])
        
        if result['valid']:
            print(f"    Vertices: {result['vertices']:,}")
            print(f"    Teeth: {result['teeth']:,} ({result['unique_teeth']} unique)")
            print(f"    Gingiva: {result['gingiva']:,}")
        else:
            print(f"    [FAIL] {result['error']}")
    
    # Data parts found
    data_parts = set()
    for scan in scans:
        path_parts = Path(scan['obj_file']).parts
        for part in path_parts:
            if 'data_part' in part:
                data_parts.add(part)
    
    if data_parts:
        print(f"\nData parts found: {', '.join(sorted(data_parts))}")
    
    print("\n" + "=" * 60)
    print("Dataset verification complete.")
    print("=" * 60)
    print("\nNext step: python scripts/generate_data.py")


if __name__ == "__main__":
    main()