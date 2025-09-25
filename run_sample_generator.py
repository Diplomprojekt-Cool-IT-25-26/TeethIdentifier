"""
Runner script to test the dental mesh sample generator.

This script creates a test mesh and runs the sample generator to
demonstrate the complete pipeline.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def modify_sample_generator_for_test():
    """
    Modify the sampleGenerator to load labels from numpy file if needed.
    """
    # This is a helper to patch the label loading
    # In production, labels would come from your annotation tool
    pass


def run_pipeline():
    """
    Run the complete sample generation pipeline.
    """
    print("=" * 60)
    print("DENTAL MESH SAMPLE GENERATOR - TEST PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate test mesh
    print("\n[Step 1] Creating test mesh...")
    from create_test_mesh import create_dental_test_mesh
    mesh_file = create_dental_test_mesh("test_dental_mesh.ply")
    
    # Step 2: Import and initialize the sampler
    print("\n[Step 2] Initializing sampler...")
    
    # We need to modify the import to handle the label loading
    import sampleGenerator
    
    # Monkey-patch the label loading to use our saved labels
    original_load_labels = sampleGenerator.DentalMeshSampler._load_vertex_labels
    
    def load_labels_with_numpy(self):
        """Modified label loader that checks for numpy file."""
        labels_file = "test_dental_mesh_labels.npy"
        if os.path.exists(labels_file):
            print(f"Loading labels from {labels_file}")
            return np.load(labels_file)
        else:
            return original_load_labels(self)
    
    sampleGenerator.DentalMeshSampler._load_vertex_labels = load_labels_with_numpy
    
    # Initialize sampler
    sampler = sampleGenerator.DentalMeshSampler(
        mesh_file,
        patch_size=100,
        patch_radius=5.0
    )
    
    # Step 3: Test different sampling strategies
    print("\n[Step 3] Testing sampling strategies...")
    
    strategies = ['random', 'uniform_surface', 'boundary_focused']
    n_samples = 100  # Small number for testing
    
    all_samples = {}
    
    for strategy in strategies:
        print(f"\n  Testing '{strategy}' sampling...")
        sample_indices = sampler.sample_training_points(
            n_samples=n_samples,
            sampling_strategy=strategy,
            balance_classes=True
        )
        
        print(f"    Generated {len(sample_indices)} sample indices")
        
        # Generate actual training samples
        training_samples = sampler.generate_training_samples(
            sample_indices,
            save_patches=True,
            patch_save_dir=f"patches_{strategy}"
        )
        
        all_samples[strategy] = training_samples
        
        # Calculate statistics
        labels = [s.label for s in training_samples]
        tooth_count = sum(1 for l in labels if l == 1)
        gingiva_count = sum(1 for l in labels if l == 0)
        
        print(f"    Tooth samples: {tooth_count}")
        print(f"    Gingiva samples: {gingiva_count}")
    
    # Step 4: Save training data
    print("\n[Step 4] Saving training data...")
    sampler.save_training_data(
        all_samples['boundary_focused'],
        "dental_training_data.pkl"
    )
    
    # Step 5: Visualize results
    print("\n[Step 5] Generating visualizations...")
    visualize_results(sampler, all_samples)
    
    # Step 6: Show sample statistics
    print("\n[Step 6] Sample Statistics:")
    print("-" * 40)
    
    # Load and inspect saved data
    import pickle
    with open("dental_training_data.pkl", 'rb') as f:
        saved_data = pickle.load(f)
    
    print(f"Saved data shape:")
    print(f"  Images: {saved_data['images'].shape}")
    print(f"  Labels: {saved_data['labels'].shape}")
    print(f"  Point indices: {saved_data['point_indices'].shape}")
    print(f"  Point coords: {saved_data['point_coords'].shape}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return sampler, all_samples


def visualize_results(sampler, all_samples):
    """
    Create comprehensive visualization of the results.
    
    Args:
        sampler: The DentalMeshSampler instance
        all_samples: Dictionary of samples by strategy
    """
    # Create figure with subplots for each strategy
    fig = plt.figure(figsize=(18, 12))
    
    strategies = list(all_samples.keys())
    n_strategies = len(strategies)
    n_examples = 4  # Examples per strategy
    
    for strat_idx, strategy in enumerate(strategies):
        samples = all_samples[strategy][:n_examples]
        
        for sample_idx, sample in enumerate(samples):
            # Calculate subplot position
            plot_idx = strat_idx * n_examples + sample_idx + 1
            ax = fig.add_subplot(n_strategies, n_examples, plot_idx)
            
            # Display the patch
            ax.imshow(sample.image_patch)
            
            # Add title
            label_name = "Tooth" if sample.label == 1 else "Gingiva"
            if sample_idx == 0:
                ax.set_title(f"{strategy}\n{label_name}", fontsize=10)
            else:
                ax.set_title(label_name, fontsize=10)
            
            ax.axis('off')
    
    plt.suptitle("Sample Patches by Sampling Strategy", fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig("sample_visualization.png", dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'sample_visualization.png'")
    
    plt.show()
    
    # Create a histogram of patch intensities
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (strategy, samples) in enumerate(all_samples.items()):
        ax = axes[idx]
        
        # Collect intensity statistics
        tooth_intensities = []
        gingiva_intensities = []
        
        for sample in samples:
            mean_intensity = np.mean(sample.image_patch)
            if sample.label == 1:  # Tooth
                tooth_intensities.append(mean_intensity)
            else:  # Gingiva
                gingiva_intensities.append(mean_intensity)
        
        # Plot histograms
        bins = np.linspace(0, 255, 30)
        ax.hist(tooth_intensities, bins=bins, alpha=0.5, 
                label='Tooth', color='blue')
        ax.hist(gingiva_intensities, bins=bins, alpha=0.5,
                label='Gingiva', color='red')
        
        ax.set_xlabel('Mean Patch Intensity')
        ax.set_ylabel('Count')
        ax.set_title(f'{strategy.title()} Sampling')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Patch Intensity Distribution by Class and Strategy', 
                 fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig("intensity_distribution.png", dpi=150, bbox_inches='tight')
    print("  Saved intensity distribution to 'intensity_distribution.png'")
    
    plt.show()


def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required = ['numpy', 'trimesh', 'cv2', 'scipy', 'sklearn', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


if __name__ == "__main__":
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    print("All dependencies found!\n")
    
    # Run the pipeline
    sampler, samples = run_pipeline()
    
    print("\nOutputs created:")
    print("  - test_dental_mesh.ply: The test mesh file")
    print("  - test_dental_mesh_labels.npy: Vertex labels")
    print("  - patches_*/: Folders with generated image patches")
    print("  - dental_training_data.pkl: Pickled training data")
    print("  - sample_visualization.png: Visualization of patches")
    print("  - intensity_distribution.png: Statistical analysis")
    
    print("\nYou can now use 'dental_training_data.pkl' to train your neural network!")