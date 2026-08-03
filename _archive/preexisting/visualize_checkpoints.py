import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.path as mpath
import re

def create_marker_from_svg(svg_paths):
    """
    Parses SVG path data to create a matplotlib Path object for use as a custom marker.
    
    Args:
        svg_paths (list of str): A list containing the 'd' attribute strings
                                from the SVG <path> elements.
    
    Returns:
        matplotlib.path.Path: A Path object for the custom marker.
    """
    all_verts = []
    all_codes = []
    
    for path_d in svg_paths:
        # Use regex to find all commands and their coordinate sequences
        path_segments = re.findall(r'([Mmc])\s*([^Mmc]*)', path_d)
        
        current_pos = np.array([0., 0.])
        should_close_path = False
        
        for command, coords_str in path_segments:
            # Check for the close path command 'z' or 'Z' at the end of coordinates
            if 'z' in coords_str.lower():
                should_close_path = True
                # Clean the string to remove the character before parsing numbers
                coords_str = coords_str.strip().rstrip('zZ')

            # Skip if the coordinate string is empty after cleaning
            if not coords_str.strip():
                continue
            
            coords = [float(c) for c in coords_str.strip().split()]
            
            if command.lower() == 'm': # moveto
                # For the first point, it's an absolute position
                current_pos = np.array([coords[0], coords[1]])
                all_verts.append(tuple(current_pos))
                all_codes.append(mpath.Path.MOVETO)
                # Subsequent points in a moveto are treated as lineto
                for i in range(2, len(coords), 2):
                    current_pos = np.array([coords[i], coords[i+1]])
                    all_verts.append(tuple(current_pos))
                    all_codes.append(mpath.Path.LINETO)

            elif command.lower() == 'c': # cubic bezier curve
                # The 'c' command uses relative coordinates for 3 points
                points = np.array(coords).reshape(-1, 2)
                for i in range(0, len(points), 3):
                    # Convert relative control and end points to absolute
                    ctrl1 = current_pos + points[i]
                    ctrl2 = current_pos + points[i+1]
                    end = current_pos + points[i+2]
                    
                    all_verts.extend([tuple(ctrl1), tuple(ctrl2), tuple(end)])
                    all_codes.extend([mpath.Path.CURVE4] * 3)
                    current_pos = end # Update current position
        
        # If a 'z' was found, add the CLOSEPOLY code for the current path
        if should_close_path:
            # The vertex for CLOSEPOLY is ignored, but one must be provided.
            # Using the first vertex of the entire path as a placeholder.
            all_verts.append(all_verts[0] if all_verts else (0,0))
            all_codes.append(mpath.Path.CLOSEPOLY)
    
    if not all_verts:
        return None

    # --- Normalize the path ---
    verts_array = np.array(all_verts)
    # 1. Flip Y-axis because SVG origin is top-left, Matplotlib is bottom-left
    verts_array[:, 1] *= -1 
    # 2. Find the bounding box
    min_x, min_y = verts_array.min(axis=0)
    max_x, max_y = verts_array.max(axis=0)
    # 3. Center the shape around (0,0)
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    verts_array -= center
    # 4. Scale it to fit in a standard marker box (e.g., -1 to 1)
    scale = max(max_x - min_x, max_y - min_y)
    verts_array /= scale
    
    return mpath.Path(verts_array, all_codes)

def load_checkpoint(checkpoint_path):
    """Load a checkpoint file (.pkl) and determine if it's from regression or PINN."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Check if it's a PINN checkpoint
    if isinstance(checkpoint, dict) and 'problem_name' in checkpoint:
        checkpoint_type = 'pinn'
        return checkpoint, checkpoint_type
    
    # It's a regression checkpoint
    checkpoint_type = 'regression'
    
    # Print basic info about the checkpoint for debugging
    print(f"Loaded regression checkpoint with keys: {list(checkpoint.keys())}")
    
    return checkpoint, checkpoint_type

def get_model_architecture_str(model_type, size, checkpoint_type='regression'):
    """Generate a string describing the model architecture based on model type and size."""
    if model_type == 'splat':
        if checkpoint_type == 'pinn':
            # For PINN, the size is directly the number of splats
            return f"[{size}] SRM"
        else:
            # For regression, estimate number of splats from parameter count
            # Assuming 2D input, 1D output: size = k * (1 + 2*(2+3)//2) = k * 6
            estimated_k = max(int(size / 6), 1)  # Avoid division errors
            return f"[{estimated_k}] SRM"
    
    elif model_type == 'kan':
        # KAN architectures from regression_test.py: [[10], [100], [300], [400], [20, 20]]
        if size == 521:  # Replace with actual parameter count for [10] KAN
            return "[10] KAN"
        elif size == 5201:  # Replace with actual parameter count for [100] KAN
            return "[100] KAN"
        elif size == 7861:  # Replace with actual parameter count for [300] KAN
            return "[300] KAN"
        elif size == 15601:  # Replace with actual parameter count for [400] KAN
            return "[400] KAN"
        elif size == 20801:  # Replace with actual parameter count for [20,20] KAN
            return "[20,20] KAN"
        # KAN architectures from pinn_test.py: [[20], [50], [100], [20, 20]]
        elif size == 1041:  # Replace with actual parameter count for [20] KAN
            return "[20] KAN"
        elif size == 2601:  # Replace with actual parameter count for [50] KAN
            return "[50] KAN"
        else:
            return f"KAN (size: {size})"
    
    elif model_type == 'mlp':
        # MLP architectures from regression_test.py: [[200], [500], [1000], [200, 200], [500, 500]]
        if size == 801:  # Replace with actual parameter count for [200] MLP
            return "[200] MLP"
        elif size == 2001:  # Replace with actual parameter count for [500] MLP
            return "[500] MLP"
        elif size == 4001:  # Replace with actual parameter count for [1000] MLP
            return "[1000] MLP"
        elif size == 41001:  # Replace with actual parameter count for [200,200] MLP
            return "[200,200] MLP"
        elif size == 252501:  # Replace with actual parameter count for [500,500] MLP
            return "[500,500] MLP"
        # MLP architectures from pinn_test.py: [[20], [50], [100], [20, 20]]
        elif size == 81:  # Replace with actual parameter count for [20] MLP
            return "[20] MLP"
        elif size == 201:  # Replace with actual parameter count for [50] MLP
            return "[50] MLP"
        elif size == 401:  # Replace with actual parameter count for [100] MLP
            return "[100] MLP"
        elif size == 501:  # Replace with actual parameter count for [20,20] MLP
            return "[20,20] MLP"
        else:
            return f"MLP (size: {size})"
    
    return f"{model_type} (size: {size})"

def visualize_ground_truth_pinn(problem_name, ax):
    """Visualize the ground truth solution for a PINN problem."""
    # Define the domain
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T
    
    # Compute analytical solution based on problem type
    if problem_name == "Poisson":
        Z = np.sin(np.pi * grid_points[:, 0]) * np.sin(np.pi * grid_points[:, 1])
        axtitle="Poisson Equation"
    elif problem_name == "AdvectionDiffusion":
        epsilon = 0.01
        velocity = (1.0, 1.0)
        exp_term1 = np.exp((grid_points[:, 0] - 0) * velocity[0] / epsilon)
        exp_term2 = np.exp((grid_points[:, 1] - 0) * velocity[1] / epsilon)
        denom1 = np.exp((1 - 0) * velocity[0] / epsilon) - 1
        denom2 = np.exp((1 - 0) * velocity[1] / epsilon) - 1
        Z = (exp_term1 - 1) / denom1 * (exp_term2 - 1) / denom2
        axtitle="Advection-Diffusion Equation"
    elif problem_name == "AllenCahn":
        epsilon = 0.1
        z = (grid_points[:, 0] + grid_points[:, 1] - 1.0) / (np.sqrt(2) * epsilon)
        Z = np.tanh(z)
        axtitle="Allen-Cahn Equation"
    elif problem_name == "Burgers":
        nu = 0.01
        z = (grid_points[:, 0] + grid_points[:, 1] - 1.0) / (2 * nu)
        Z = 1 - np.tanh(z)
        axtitle="Burgers Equation"
    else:
        Z = np.zeros(grid_points.shape[0])
    
    Z = Z.reshape(100, 100)
    
    # Create plot
    im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax.set_title(f'Ground Truth: {axtitle}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

def visualize_ground_truth_regression(ax, function_name=None):
    """Create a visualization for regression ground truth."""
    # Create a simple visualization since we can't recover the exact function
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Use different visualizations based on function name
    if function_name and "sine_cosine_product" in function_name:
        Z = np.sin(3 * np.pi * np.sqrt(X)) * np.cos(3 * np.pi * Y)
        title = "Ground Truth: sine(3π√x) × cos(3πy)"
    else:
        # Default visualization
        Z = np.sin(3 * np.pi * X) * np.sin(3 * np.pi * Y)
        title = "Ground Truth Function (Approximation)"
    
    # Create plot
    im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
def plot_results(checkpoint, checkpoint_type, output_path=None):
    """Plot results from checkpoint."""
    # Extract relevant data based on checkpoint type
    if checkpoint_type == 'pinn':
        results = checkpoint['results']
        problem_name = checkpoint['problem_name']
        function_name = None
    else:  # regression
        # For regression, extract results and function_name correctly
        if 'results' in checkpoint:
            results = checkpoint['results']
            function_name = checkpoint.get('function_name', None)
        else:
            # In case of old format checkpoints
            results = checkpoint
            function_name = None
            print("WARNING: Old checkpoint format detected without 'results' key.")
    
    # Verify results structure
    if not results or not isinstance(results, dict):
        print(f"ERROR: Invalid results structure: {type(results)}")
        print(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Plot ground truth (left subplot)
    if checkpoint_type == 'pinn':
        visualize_ground_truth_pinn(problem_name, axes[0])
    else:
        visualize_ground_truth_regression(axes[0], function_name)
    
    # 2. Plot validation curves (middle subplot)
    model_types = ['splat', 'kan', 'mlp']
    colors = {'splat': 'blue', 'kan': 'green', 'mlp': 'red'}
    markers = {'splat': 'o', 'kan': 's', 'mlp': '^'}
    
    # Check for model types in results
    valid_model_types = [m for m in model_types if m in results]
    if not valid_model_types:
        print(f"WARNING: No valid model types found. Available keys: {list(results.keys())}")
    
    # For each model type and architecture
    for model_type in valid_model_types:
        if not results[model_type]:
            print(f"WARNING: No data for {model_type}")
            continue
            
        # Sort architectures by size for consistent color gradient
        arch_sizes = sorted(results[model_type].keys())
        
        # Create color gradients
        cmap = plt.get_cmap('Blues' if model_type == 'splat' else 
                           'Greens' if model_type == 'kan' else 'Reds')
        color_norm = plt.Normalize(vmin=-1, vmax=max(1, len(arch_sizes)-1))
        
        for i, size in enumerate(arch_sizes):
            metrics = results[model_type][size]
            color = cmap(color_norm(i))
            
            # Plot validation curve - choose correct field names based on checkpoint type
            if checkpoint_type == 'pinn':
                x_values = metrics['val_steps']
                y_values = metrics['val_errors']
            else:
                # Check for correct keys in metrics
                if 'val_steps' not in metrics or 'val_mse' not in metrics:
                    print(f"WARNING: Missing validation data for {model_type} size {size}")
                    print(f"Available keys: {list(metrics.keys())}")
                    continue
                    
                x_values = metrics['val_steps']
                y_values = metrics['val_mse']
            
            # Create a more descriptive label based on architecture
            arch_str = get_model_architecture_str(model_type, size, checkpoint_type)
            label = f"{model_type.upper()} {arch_str}"
            
            axes[1].semilogy(
                x_values, y_values, 
                label=label,
                color=color,
                marker=markers[model_type],
                markevery=max(1, len(x_values)//10),
                markersize=6,
                alpha=0.8
            )
    
    title = 'Validation Error' if checkpoint_type == 'pinn' else 'Validation MSE'
    axes[1].set_title(f'{title} during Training')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel(f'{title} (log scale)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='upper right')
    
    # 3. Plot architecture size vs final MSE/error (right subplot)
    # Initialize lists to store data for scatter plot
    sizes_by_type = {model_type: [] for model_type in model_types}
    mse_by_type = {model_type: [] for model_type in model_types}
    labels_by_type = {model_type: [] for model_type in model_types}
    
    for model_type in valid_model_types:
        if model_type in results:
            for size, metrics in results[model_type].items():
                sizes_by_type[model_type].append(size)
                
                # Get final MSE/error value based on checkpoint type
                if checkpoint_type == 'pinn':
                    final_value = metrics['final_val_error']
                else:
                    if 'final_val_mse' in metrics:
                        final_value = metrics['final_val_mse']
                    elif 'val_mse' in metrics and len(metrics['val_mse']) > 0:
                        # If final_val_mse is missing, use the last value in val_mse
                        final_value = metrics['val_mse'][-1]
                    else:
                        print(f"WARNING: Cannot determine final MSE for {model_type} size {size}")
                        continue
                    
                mse_by_type[model_type].append(final_value)
                labels_by_type[model_type].append(
                    get_model_architecture_str(model_type, size, checkpoint_type)
                )
    svg_path_strings = [
        "M7355 12005 c-248 -54 -425 -237 -495 -510 -57 -219 -24 -641 79 -1044 27 -102 71 -274 99 -381 55 -214 76 -328 92 -495 45 -493 -60 -892 -293 -1112 -177 -167 -398 -234 -708 -212 -288 20 -495 108 -680 289 -192 189 -283 363 -476 919 -47 135 -106 293 -132 351 -147 330 -336 480 -603 480 -109 0 -165 -19 -231 -80 -101 -93 -143 -214 -133 -385 11 -189 70 -348 189 -505 83 -109 176 -198 410 -389 361 -296 456 -434 457 -666 0 -172 -39 -275 -147 -381 -101 -101 -265 -174 -486 -219 -159 -31 -334 -50 -647 -70 -590 -37 -774 -77 -865 -189 -28 -35 -30 -44 -30 -119 0 -101 22 -151 104 -239 72 -78 164 -148 401 -303 243 -159 336 -229 430 -324 88 -89 145 -171 179 -261 24 -60 26 -77 25 -215 -1 -132 -4 -161 -28 -238 -123 -410 -455 -609 -1181 -711 -60 -9 -220 -27 -355 -41 -135 -14 -299 -33 -365 -41 -459 -57 -680 -150 -772 -325 -26 -49 -28 -63 -28 -159 0 -90 3 -111 23 -147 48 -91 126 -144 269 -185 67 -19 102 -23 238 -22 223 0 305 23 744 210 307 132 475 194 621 233 210 55 287 65 505 65 174 0 212 -2 289 -22 254 -64 425 -191 522 -387 60 -124 77 -207 77 -385 -1 -243 -42 -430 -142 -635 -164 -337 -370 -539 -1025 -1006 -502 -357 -735 -590 -875 -874 -104 -211 -146 -407 -138 -645 3 -121 9 -158 31 -227 62 -190 206 -324 408 -379 77 -21 248 -29 350 -15 423 57 714 297 953 786 105 216 162 367 349 935 170 516 262 764 356 950 294 588 660 854 1205 877 233 10 460 -30 633 -112 325 -154 523 -433 642 -900 51 -200 82 -398 130 -835 58 -524 97 -723 172 -875 58 -118 129 -189 226 -226 51 -19 71 -21 187 -16 103 4 146 10 210 30 212 69 353 218 390 413 19 95 19 141 0 228 -22 103 -72 193 -214 385 -71 96 -152 213 -180 259 -142 238 -196 514 -173 879 20 316 64 508 162 708 94 194 211 315 385 400 135 66 227 86 435 92 409 13 924 -49 1264 -152 404 -123 633 -254 1126 -646 360 -286 586 -379 919 -379 87 0 145 5 182 16 179 53 303 186 356 384 25 90 24 309 -1 399 -68 248 -255 411 -596 519 -148 47 -302 78 -745 152 -599 100 -850 164 -1094 281 -269 128 -478 323 -594 554 -202 404 -207 1039 -11 1463 85 185 247 376 430 508 105 76 364 206 519 263 61 22 242 80 403 130 395 123 516 173 631 259 68 51 105 96 139 169 25 53 27 68 26 173 -1 137 -18 211 -77 325 -121 232 -372 356 -692 343 -130 -6 -217 -35 -336 -114 -106 -69 -290 -259 -475 -489 -198 -245 -290 -349 -380 -427 -218 -188 -337 -218 -749 -187 -759 56 -1149 310 -1335 868 -75 223 -110 491 -110 831 0 383 38 546 225 965 179 402 228 613 229 990 1 158 -2 190 -22 263 -46 170 -115 286 -236 399 -155 144 -366 203 -571 158z",
        "M545 8539 c-99 -13 -238 -54 -309 -92 -118 -62 -203 -173 -226 -296 -33 -173 11 -318 136 -451 103 -109 231 -181 419 -236 141 -41 259 -56 480 -61 312 -7 520 24 690 106 95 46 155 95 193 159 23 40 27 58 27 118 -1 62 -6 82 -37 145 -121 243 -472 483 -853 584 -86 23 -124 27 -275 30 -96 2 -206 -1 -245 -6z",
        "M10180 1384 c-210 -37 -353 -117 -469 -262 -103 -128 -145 -256 -145 -437 1 -102 5 -132 27 -200 36 -110 89 -196 172 -279 82 -83 168 -136 280 -173 70 -23 96 -27 210 -27 114 0 140 4 210 27 111 37 198 90 280 172 82 82 135 169 172 280 22 68 27 98 27 200 0 85 -5 139 -17 185 -67 244 -259 434 -499 494 -68 17 -204 28 -248 20z"
    ]
    
    splat_marker = create_marker_from_svg(svg_path_strings)
    
    # Plot scatter points for each model type
    for model_type in valid_model_types:
        if sizes_by_type[model_type]:  # Only plot if there's data
            # Use custom splat marker for splat models
            marker = splat_marker if model_type == 'splat' else markers[model_type]
            marker_size = 200 if model_type == 'splat' else 80  # Larger marker for splat
            
            axes[2].scatter(
                sizes_by_type[model_type], 
                mse_by_type[model_type], 
                color=colors[model_type], 
                marker=marker,
                s=marker_size, 
                alpha=0.7,
                label=model_type.upper()
            )
            
            # Add labels to points
            for i, (size, mse, label) in enumerate(zip(
                    sizes_by_type[model_type], 
                    mse_by_type[model_type], 
                    labels_by_type[model_type])):
                
                # Position labels to avoid overlap
                h_align = 'left'
                offset_x = 10
                if model_type == 'mlp':  # Position MLP labels to the left
                    h_align = 'right'
                    offset_x = -10
                
                axes[2].annotate(
                    label, 
                    (size, mse),
                    xytext=(offset_x, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    ha=h_align,
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec=colors[model_type])
                )
    
    title = 'Final Test Error' if checkpoint_type == 'pinn' else 'Final Validation MSE'
    axes[2].set_title(f'{title} vs Architecture Size')
    axes[2].set_xlabel('Architecture Size (parameters)')
    axes[2].set_ylabel(f'{title} (log scale)')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10, loc='upper right')
    
    # Add overall title
    checkpoint_name = Path(output_path).stem if output_path else "Checkpoint"
    if function_name:
        pass#fig.suptitle(f"Regression Results: {function_name}", fontsize=16)
    elif checkpoint_type == 'pinn':
        pass#fig.suptitle(f"PINN Results: {problem_name}", fontsize=16)
    else:
        fig.suptitle(f"Model Performance Visualization", fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save or display the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize model checkpoint results.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the .pkl checkpoint file')
    parser.add_argument('--output', type=str, help='Path to save the output plot (optional)')
    parser.add_argument('--debug', action='store_true', help='Print extra debugging information')
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint, checkpoint_type = load_checkpoint(args.checkpoint_path)
    
    if args.debug:
        print(f"Checkpoint type: {checkpoint_type}")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if checkpoint_type == 'regression' and 'results' in checkpoint:
            print("Model types present:", list(checkpoint['results'].keys()))
            for model_type, data in checkpoint['results'].items():
                print(f"  {model_type}: {list(data.keys()) if data else 'Empty'}")
    
    # Determine output path if not specified
    output_path = args.output
    if not output_path:
        # Generate a default output path based on the checkpoint filename
        checkpoint_filename = Path(args.checkpoint_path).stem
        output_path = f"{checkpoint_filename}_visualization.pdf"
    
    # Plot results
    plot_results(checkpoint, checkpoint_type, output_path)

if __name__ == "__main__":
    main()