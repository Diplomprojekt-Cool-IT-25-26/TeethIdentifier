"""
Creates a simple test mesh file for testing the sampleGenerator.

This script generates a synthetic dental arch-like mesh with simulated
tooth and gingiva regions for testing purposes.
"""

import numpy as np
import trimesh
import os


def create_dental_test_mesh(output_file="test_dental_mesh.ply"):
    """
    Create a synthetic dental arch mesh for testing.
    
    Args:
        output_file: Path to save the mesh file
    """
    # Create a simple arch-like structure
    print("Generating synthetic dental mesh...")
    
    # Parameters for the arch
    n_points_length = 50  # Points along the arch
    n_points_width = 20   # Points across the width
    arch_radius = 30      # Radius of the dental arch
    arch_width = 10       # Width of the dental arch
    
    vertices = []
    colors = []
    labels = []
    
    # Generate vertices in an arch pattern
    for i in range(n_points_length):
        # Angle along the arch (from -pi/2 to pi/2)
        theta = -np.pi/2 + (np.pi * i / (n_points_length - 1))
        
        for j in range(n_points_width):
            # Vary radius to create thickness
            r = arch_radius + (j - n_points_width/2) * 0.5
            
            # Calculate position
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Add some height variation to simulate teeth
            if i % 8 < 4:  # Create tooth-like bumps
                z = 5 * np.sin(i * np.pi / 4) + np.random.normal(0, 0.2)
                is_tooth = True
            else:  # Gingiva areas
                z = np.random.normal(0, 0.5)
                is_tooth = False
            
            vertices.append([x, y, z])
            
            # Assign colors based on tooth/gingiva
            if is_tooth:
                # Whitish color for teeth
                colors.append([240 + np.random.randint(-10, 10),
                             240 + np.random.randint(-10, 10),
                             230 + np.random.randint(-10, 10)])
                labels.append(1)  # Tooth label
            else:
                # Pinkish color for gingiva
                colors.append([255, 
                             180 + np.random.randint(-20, 20),
                             180 + np.random.randint(-20, 20)])
                labels.append(0)  # Gingiva label
    
    vertices = np.array(vertices)
    colors = np.array(colors, dtype=np.uint8)
    labels = np.array(labels)
    
    # Create faces using Delaunay triangulation
    print("Creating mesh faces...")
    
    # Simple grid-based face creation
    faces = []
    for i in range(n_points_length - 1):
        for j in range(n_points_width - 1):
            # Current vertex index
            idx = i * n_points_width + j
            
            # Create two triangles for each grid square
            # Triangle 1
            faces.append([idx, idx + 1, idx + n_points_width])
            # Triangle 2
            faces.append([idx + 1, idx + n_points_width + 1, idx + n_points_width])
    
    faces = np.array(faces)
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Add vertex colors
    mesh.visual.vertex_colors = colors
    
    # Store labels as vertex attributes (if possible)
    # Note: This might not be preserved in all file formats
    if hasattr(mesh.visual, 'vertex_attributes'):
        mesh.visual.vertex_attributes['labels'] = labels
    
    # Calculate vertex normals
    mesh.vertex_normals
    
    # Save mesh
    mesh.export(output_file)
    
    print(f"Mesh saved to: {output_file}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"Tooth vertices: {np.sum(labels == 1)}")
    print(f"Gingiva vertices: {np.sum(labels == 0)}")
    
    # Also save labels separately for easier loading
    labels_file = output_file.replace('.ply', '_labels.npy')
    np.save(labels_file, labels)
    print(f"Labels saved to: {labels_file}")
    
    return output_file


def visualize_mesh(mesh_file):
    """
    Quick visualization of the generated mesh.
    
    Args:
        mesh_file: Path to the mesh file
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        mesh = trimesh.load(mesh_file)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot vertices with colors
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = 'blue'
        
        ax.scatter(mesh.vertices[:, 0],
                  mesh.vertices[:, 1],
                  mesh.vertices[:, 2],
                  c=colors, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Generated Dental Test Mesh')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Create the test mesh
    mesh_file = create_dental_test_mesh()
    
    # Try to visualize it
    print("\nAttempting to visualize mesh...")
    visualize_mesh(mesh_file)