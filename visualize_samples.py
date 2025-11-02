"""
Visualize training sample patches for presentation.

This script generates and displays the 2D image patches (I(p_i)) 
that are created from 3D dental scans for neural network training.
"""

import os
import matplotlib.pyplot as plt
from dataset_sampler import DatasetSampler


def visualize_training_samples(dataset_root=".", scan_index=0, n_samples=16):
    """
    Generate and visualize training sample patches.
    
    Args:
        dataset_root: Path to dataset
        scan_index: Which scan to load (0 = first scan)
        n_samples: Number of samples to display (default: 16 for 4x4 grid)
    """
    print("=" * 60)
    print("Training Sample Visualization")
    print("=" * 60)
    
    # Initialize sampler
    print(f"\nLoading dataset from: {os.path.abspath(dataset_root)}")
    sampler = DatasetSampler(dataset_root)
    
    if len(sampler.scan_files) == 0:
        print("❌ No scans found!")
        return
    
    # Load scan
    print(f"\nLoading scan {scan_index}...")
    if not sampler.load_scan(scan_index):
        print("❌ Failed to load scan")
        return
    
    # Generate samples
    print(f"\nGenerating {n_samples} sample patches...")
    sample_indices = sampler.sample_points(n_samples, balance_classes=True)
    samples = sampler.generate_samples(sample_indices)
    
    # Create visualization
    print("Creating visualization...")
    
    # Determine grid size
    if n_samples <= 4:
        rows, cols = 2, 2
    elif n_samples <= 9:
        rows, cols = 3, 3
    elif n_samples <= 16:
        rows, cols = 4, 4
    elif n_samples <= 25:
        rows, cols = 5, 5
    else:
        rows, cols = 6, 6
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    # Plot each sample
    for i, sample in enumerate(samples[:rows*cols]):
        axes[i].imshow(sample.image_patch)
        
        # Create title with label
        label_text = "Gingiva" if sample.label == 0 else "Tooth"
        color = "red" if sample.label == 0 else "blue"
        
        axes[i].set_title(f"{label_text}\n(vertex {sample.point_index})", 
                         fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(samples), rows*cols):
        axes[i].axis('off')
    
    # Add main title
    patient_id = samples[0].patient_id if samples else "Unknown"
    jaw = samples[0].jaw if samples else "Unknown"
    fig.suptitle(f"Training Sample Patches (I(p_i))\n"
                 f"Patient: {patient_id} | Jaw: {jaw.upper()}", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = "sample_patches_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")
    
    # Show figure
    plt.show()
    
    # Print summary
    print("\nSample Summary:")
    gingiva_count = sum(1 for s in samples if s.label == 0)
    tooth_count = len(samples) - gingiva_count
    print(f"  Gingiva samples: {gingiva_count} (red labels)")
    print(f"  Tooth samples: {tooth_count} (blue labels)")
    print(f"  Patch size: {samples[0].image_patch.shape}")
    print(f"  Patient: {patient_id}")
    print(f"  Jaw: {jaw}")


def visualize_comparison(dataset_root=".", scan_index=0):
    """
    Create side-by-side comparison of gingiva vs tooth patches.
    
    Shows 8 gingiva patches on the left and 8 tooth patches on the right.
    """
    print("=" * 60)
    print("Gingiva vs Tooth Comparison")
    print("=" * 60)
    
    # Initialize sampler
    sampler = DatasetSampler(dataset_root)
    
    if len(sampler.scan_files) == 0:
        print("❌ No scans found!")
        return
    
    # Load scan
    print(f"\nLoading scan {scan_index}...")
    if not sampler.load_scan(scan_index):
        print("❌ Failed to load scan")
        return
    
    # Generate more samples to ensure we get both classes
    print(f"\nGenerating samples...")
    sample_indices = sampler.sample_points(50, balance_classes=True)
    samples = sampler.generate_samples(sample_indices)
    
    # Separate by class
    gingiva_samples = [s for s in samples if s.label == 0][:8]
    tooth_samples = [s for s in samples if s.label == 1][:8]
    
    # Create figure with two sections
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    # Plot gingiva samples (top row)
    for i in range(8):
        if i < len(gingiva_samples):
            axes[0, i].imshow(gingiva_samples[i].image_patch)
            axes[0, i].set_title(f"Gingiva\n{i+1}", fontsize=9, color='red')
        axes[0, i].axis('off')
    
    # Plot tooth samples (bottom row)
    for i in range(8):
        if i < len(tooth_samples):
            axes[1, i].imshow(tooth_samples[i].image_patch)
            axes[1, i].set_title(f"Tooth\n{i+1}", fontsize=9, color='blue')
        axes[1, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'GINGIVA\n(Class 0)', fontsize=14, 
             fontweight='bold', color='red', va='center', rotation=90)
    fig.text(0.02, 0.25, 'TOOTH\n(Class 1)', fontsize=14, 
             fontweight='bold', color='blue', va='center', rotation=90)
    
    plt.suptitle("Training Samples: Gingiva vs Tooth Classification", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = "gingiva_vs_tooth_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {output_file}")
    
    plt.show()


def main():
    """Main function with menu."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "Sample Patch Visualizer" + " " * 23 + "║")
    print("╚" + "═" * 58 + "╝")
    
    print("\nChoose visualization type:")
    print("  1. Standard grid (16 samples)")
    print("  2. Gingiva vs Tooth comparison")
    print("  3. Custom number of samples")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    dataset_root = "."
    
    if choice == "1":
        visualize_training_samples(dataset_root, scan_index=0, n_samples=16)
    
    elif choice == "2":
        visualize_comparison(dataset_root, scan_index=0)
    
    elif choice == "3":
        try:
            n = int(input("How many samples? (1-36): "))
            n = max(1, min(36, n))  # Limit between 1 and 36
            visualize_training_samples(dataset_root, scan_index=0, n_samples=n)
        except ValueError:
            print("Invalid number, using default (16)")
            visualize_training_samples(dataset_root, scan_index=0, n_samples=16)
    
    else:
        print("Invalid choice, using standard grid")
        visualize_training_samples(dataset_root, scan_index=0, n_samples=16)
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()