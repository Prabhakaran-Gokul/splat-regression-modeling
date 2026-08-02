import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
import time
import pandas as pd
from tqdm import tqdm
from functools import partial
import optax
import os
from tabulate import tabulate
import seaborn as sns
import pickle
from datetime import datetime
from pathlib import Path

# Configure JAX for GPU usage
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Configure JAX for optimal GPU performance
jax.config.update('jax_enable_x64', False)  # Use float32 for better GPU performance
jax.config.update('jax_platform_name', 'gpu')  # Force GPU usage

# Import model functions from existing files
from lib.splat import eval_splat, gd_splat_regression
from lib.nets import gd_net_regression

# JIT-compiled utility functions for efficiency
@jax.jit
def compute_mse(y_pred, y_true):
    """Compute mean squared error between predictions and true values."""
    return jnp.mean((y_pred - y_true)**2)

def warmup_gpu():
    """Warm up GPU by running a dummy computation to force compilation."""
    print("Warming up GPU...")
    key = jr.PRNGKey(42)
    
    # Dummy operations to warm up GPU
    dummy_data = jr.normal(key, (1000, 10))
    dummy_target = jr.normal(key, (1000, 1))
    
    # Force compilation and GPU memory allocation
    _ = compute_mse(dummy_data[:, :1], dummy_target)
    _ = jnp.dot(dummy_data, dummy_data.T)
    _ = jax.nn.relu(dummy_data)
    _ = jnp.sin(dummy_data)
    _ = jnp.cos(dummy_data)
    
    # Block until computation is complete
    _.block_until_ready()
    print("GPU warmup complete.")

@jax.jit
def generate_data(key, n_samples, function, input_dim, noise_level):
    """
    Generate data for regression task.
    """
    # Split key for inputs and noise
    key_x, key_noise = jr.split(key)
    
    # Generate uniform inputs in [0, 1]^d
    X = jr.uniform(key_x, (n_samples, input_dim))
    
    # Evaluate function
    Y_true = function(X)
    
    # Add noise
    noise = jr.normal(key_noise, Y_true.shape) * noise_level
    Y = Y_true + noise
    
    return X, Y


def train_and_evaluate_splat(init_splat, train_X, train_Y, test_X, test_Y, 
                           num_steps, lr, adam, validation_interval):
    """
    Train a splat model and track training/validation errors.
    """
    start_time = time.time()
    
    # Define a JIT-compiled function to compute training loss
    @jax.jit
    def compute_loss(params, x, y):
        y_pred = eval_splat(x, params)
        return compute_mse(y_pred, y)
    
    # JIT-compiled evaluation function
    @jax.jit 
    def evaluate_model(params, x):
        return eval_splat(x, params)
    
    # Train the model
    splat_trajectory = gd_splat_regression(
        init_splat, train_X, train_Y, 
        lr=lr, num_steps=num_steps, adam=adam, verbose=True
    )
    
    # Compute training loss at each step (vectorized for efficiency)
    @jax.jit
    def compute_all_losses(trajectory_params, x, y):
        """Compute losses for entire trajectory at once."""
        losses = []
        for params in trajectory_params:
            y_pred = eval_splat(x, params)
            loss = compute_mse(y_pred, y)
            losses.append(loss)
        return jnp.array(losses)
    
    train_losses = []
    for params in splat_trajectory:
        loss = compute_loss(params, train_X, train_Y)
        loss.block_until_ready()  # Ensure computation completes
        train_losses.append(float(loss))
    
    # Compute validation MSE at specified intervals
    val_steps = list(range(0, num_steps, validation_interval))
    if (num_steps-1) not in val_steps:
        val_steps.append(num_steps-1)
    
    @jax.jit
    def compute_validation_mse(params, x, y):
        """Compute validation MSE for given parameters."""
        y_pred = evaluate_model(params, x)
        return compute_mse(y_pred, y)
    
    val_mse = []
    for step in val_steps:
        params = splat_trajectory[step]
        mse = compute_validation_mse(params, test_X, test_Y)
        mse.block_until_ready()  # Ensure computation completes
        val_mse.append(float(mse))
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    return {
        'train_losses': train_losses,
        'val_steps': val_steps,
        'val_mse': val_mse,
        'final_val_mse': val_mse[-1],
        'wall_time': wall_time
    }

def train_and_evaluate_nn_model(model, train_X, train_Y, test_X, test_Y, 
                               num_steps, lr, adam, validation_interval, adam_params=(0.9, 0.999, 1e-8)):
    """
    Train a neural network model (KAN or MLP) and track training/validation errors.
    """
    start_time = time.time()
    
    # Initialize results
    train_losses = []
    val_steps = []
    val_mse = []
    

    if adam:
        b1, b2, eps = adam_params
        optimizer = nnx.Optimizer(model, optax.adam(lr, b1, b2, eps=eps), wrt=nnx.Param)
    else:
        optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)

    @jax.jit
    def loss_fn(kan_model):
        y_pred = kan_model(train_X)
        return compute_mse(y_pred, train_Y)
    
    # JIT-compiled evaluation function
    @jax.jit
    def evaluate_model(model, x):
        return model(x)

    grad_fn = nnx.grad(loss_fn)
    R = tqdm(range(num_steps), desc=f"Training {'KAN' if 'KAN' in str(type(model)) else 'MLP'}")

    
    # Training loop with validation at intervals
    for step in R:
        grads = grad_fn(model)
        optimizer.update(model, grads)
        
        # Force computation to complete and update progress
        current_loss = loss_fn(model)
        current_loss.block_until_ready()
        R.set_description(f"Training {'KAN' if 'KAN' in str(type(model)) else 'MLP'} – log(MSE) = {jnp.log10(current_loss):.4f}")
        
        # Validate at specified intervals
        if step % validation_interval == 0 or step == num_steps - 1:
            val_steps.append(step)
            test_pred = evaluate_model(model, test_X)
            test_pred.block_until_ready()  # Ensure computation completes
            test_loss = float(compute_mse(test_pred, test_Y))
            val_mse.append(test_loss)
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    return {
        'train_losses': train_losses,
        'val_steps': val_steps,
        'val_mse': val_mse,
        'final_val_mse': val_mse[-1],
        'wall_time': wall_time
    }

def save_snapshot(snapshot_file, results, function_name, config):
    """
    Save current training results to a snapshot file.
    
    Args:
        snapshot_file: Path to the snapshot file
        results: Results dictionary from training
        function_name: Name of the regression function
        config: Configuration parameters
    """
    # Create directory if it doesn't exist
    Path(snapshot_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save current timestamp
    snapshot = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'results': results,
        'function_name': function_name,
        'config': config
    }
    
    # Save to file
    with open(snapshot_file, 'wb') as f:
        pickle.dump(snapshot, f)
        
    print(f"Snapshot saved to {snapshot_file}")

def load_snapshot(snapshot_file):
    """
    Load training results from a snapshot file.
    
    Args:
        snapshot_file: Path to the snapshot file
        
    Returns:
        Dictionary containing loaded results and configuration
    """
    with open(snapshot_file, 'rb') as f:
        snapshot = pickle.load(f)
    
    print(f"Loaded snapshot from {snapshot_file}")
    print(f"Created on: {snapshot['timestamp']}")
    print(f"Function: {snapshot.get('function_name', 'Unknown')}")
    
    return snapshot

def run_regression_comparison(
    train_size,
    test_size,
    function,
    noise_level,
    input_dim,
    splat_architectures,
    kan_architectures,
    mlp_architectures,
    num_steps=1000,
    lr=1e-3,
    adam=True,
    validation_interval=100,
    seed=0,
    snapshot_file=None,
    function_name="regression_function"
):
    """
    Run regression comparison experiments with snapshot support.
    """
    # Initialize random key
    key = jr.PRNGKey(seed)
    key_train, key_test = jr.split(key)
    
    # Generate data and ensure it's on GPU
    train_X, train_Y = generate_data(key_train, train_size, function, input_dim, noise_level)
    test_X, test_Y = generate_data(key_test, test_size, function, input_dim, noise_level)
    
    # Force data to GPU by calling a JIT function that returns the same data
    @jax.jit
    def ensure_gpu_data(x, y):
        return x, y
    
    train_X, train_Y = ensure_gpu_data(train_X, train_Y)
    test_X, test_Y = ensure_gpu_data(test_X, test_Y)
    
    print(f"Data moved to device: {train_X.device()}")
    
    # Initialize results dictionary
    results = {
        'splat': {},
        'kan': {},
        'mlp': {}
    }
    
    # Store configuration
    config = {
        'train_size': train_size,
        'test_size': test_size,
        'noise_level': noise_level,
        'input_dim': input_dim,
        'num_steps': num_steps,
        'lr': lr,
        'adam': adam,
        'validation_interval': validation_interval,
        'seed': seed
    }
    
    # Train and evaluate splat models
    print(f"Training {len(splat_architectures)} splat models...")
    for i, init_splat in enumerate(splat_architectures):
        print(f"  Model {i+1}/{len(splat_architectures)}")
        n,d = train_X.shape
        n,p = train_Y.shape
        architecture_size = init_splat[0].shape[0] * (p + d * (d+3)//2)  # Number of splats (k)
        
        result = train_and_evaluate_splat(
            init_splat, train_X, train_Y, test_X, test_Y,
            num_steps, lr, adam, validation_interval
        )
        
        results['splat'][architecture_size] = result
        
        # Update snapshot after each model is trained
        if snapshot_file:
            save_snapshot(snapshot_file, results, function_name, config)
    
    # Train and evaluate KAN models
    print(f"Training {len(kan_architectures)} KAN models...")
    for i, model in enumerate(kan_architectures):
        print(f"  Model {i+1}/{len(kan_architectures)}")
        
        # Get architecture size (approximate number of parameters)
        architecture_size = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        result = train_and_evaluate_nn_model(
            model, train_X, train_Y, test_X, test_Y,
            num_steps, lr, adam, validation_interval
        )
        
        results['kan'][architecture_size] = result
        
        # Update snapshot after each model is trained
        if snapshot_file:
            save_snapshot(snapshot_file, results, function_name, config)
    
    

    # Train and evaluate MLP models
    print(f"Training {len(mlp_architectures)} MLP models...")
    for i, model in enumerate(mlp_architectures):
        print(f"  Model {i+1}/{len(mlp_architectures)}")
        
        # Get architecture size (approximate number of parameters)
        architecture_size = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))

        result = train_and_evaluate_nn_model(
            model, train_X, train_Y, test_X, test_Y,
            num_steps, lr, adam, validation_interval
        )
        
        results['mlp'][architecture_size] = result
        
        # Update snapshot after each model is trained
        if snapshot_file:
            save_snapshot(snapshot_file, results, function_name, config)
    
    return results

def plot_training_loss(results):
    """
    Plot training loss vs. steps for each model type and architecture.
    
    Args:
        results: results dictionary from run_regression_comparison
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each model type with different line styles
    line_styles = {'splat': '-', 'kan': '--', 'mlp': '-.'}
    
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            label = f"{model_type} (size: {arch_size})"
            plt.plot(metrics['train_losses'], 
                    line_styles[model_type], 
                    label=label, 
                    alpha=0.8)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss vs. Steps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_validation_mse(results):
    """
    Plot validation MSE vs. steps for each model type and architecture.
    
    Args:
        results: results dictionary from run_regression_comparison
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each model type with different line styles
    line_styles = {'splat': '-', 'kan': '--', 'mlp': '-.'}
    markers = {'splat': 'o', 'kan': 's', 'mlp': '^'}
    
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            label = f"{model_type} (size: {arch_size})"
            plt.plot(metrics['val_steps'], metrics['val_mse'], 
                    line_styles[model_type], 
                    marker=markers[model_type],
                    label=label, 
                    alpha=0.8,
                    markersize=6)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs. Steps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('validation_mse.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_mse_vs_size(results):
    """
    Plot final validation MSE vs. architecture size for each model type.
    
    Args:
        results: results dictionary from run_regression_comparison
    """
    plt.figure(figsize=(10, 6))
    
    markers = {'splat': 'o', 'kan': 's', 'mlp': '^'}
    colors = {'splat': 'blue', 'kan': 'green', 'mlp': 'red'}
    
    for model_type, architectures in results.items():
        sizes = []
        mse_values = []
        
        for arch_size, metrics in architectures.items():
            sizes.append(arch_size)
            mse_values.append(metrics['final_val_mse'])
        
        # Sort by size for proper line plotting
        sorted_idx = np.argsort(sizes)
        sizes = [sizes[i] for i in sorted_idx]
        mse_values = [mse_values[i] for i in sorted_idx]
        
        plt.plot(sizes, mse_values, 'o-', 
                marker=markers[model_type],
                color=colors[model_type],
                label=model_type,
                markersize=8)
    
    plt.xlabel('Architecture Size')
    plt.ylabel('Final Validation MSE')
    plt.title('Final Validation MSE vs. Architecture Size')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('final_mse_vs_size.png', dpi=300)
    plt.show()

def generate_timing_table(results):
    """
    Generate a LaTeX table with wall clock training times.
    
    Args:
        results: results dictionary from run_regression_comparison
        
    Returns:
        str: LaTeX table code
    """
    # Prepare data for the table
    data = []
    
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            data.append({
                'Model Type': model_type.upper(),
                'Architecture Size': arch_size,
                'Wall Time (s)': metrics['wall_time'],
                'Final MSE': metrics['final_val_mse']
            })
    
    # Create pandas DataFrame
    df = pd.DataFrame(data)
    
    # Sort by model type and architecture size
    df = df.sort_values(['Model Type', 'Architecture Size'])
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f")
    
    # Save to file
    with open('timing_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Print table in terminal
    print("\nWall Clock Training Times:")
    print(tabulate(df, headers='keys', tablefmt='grid'))
    
    return latex_table

def example_usage(load_from=None, name=None, seed=42):
    """
    Example usage of the regression comparison framework.
    
    Args:
        load_from: Optional path to load snapshot from
    """
    if load_from:
        # Load from snapshot instead of training
        snapshot = load_snapshot(load_from)
        results = snapshot['results']
        function_name = snapshot.get('function_name', 'Unknown')
        config = snapshot['config']
        
        print(f"Loaded results for {function_name} function")
        print(f"Configuration: {config}")
        
        # Generate plots from loaded results
        plot_training_loss(results)
        plot_validation_mse(results)
        plot_final_mse_vs_size(results)
        generate_timing_table(results)
        
        return results
    
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jaxkan.KAN import KAN
    
    # Set random seed
    seed = seed
    key = jr.PRNGKey(seed)
    key1, key2, key3 = jr.split(key, 3)
    
    # Define function to fit (example: sine function with two inputs)
    @jax.jit
    def target_function(X):
        return jnp.sin(3 * jnp.pi * jnp.sqrt(X[:, 0:1])) * jnp.cos(3 * jnp.pi * X[:, 1:2])
    
    function_name = "sine_cosine_product"
    # Define experiment parameters
    input_dim = 2
    
    # Create splat architectures with different numbers of components
    splat_architectures = []
    for k in [10, 50, 100, 200, 300, 400]:
        # Initialize splat parameters (V, A, B)
        key_splat = jr.PRNGKey(k)
        key_v, key_a, key_b = jr.split(key_splat, 3)
        
        # V: weights of shape [k, 1]
        V = jnp.zeros((k, 1))
        
        # A: covariance matrices of shape [k, input_dim, input_dim]
        A = jnp.tile(jnp.eye(input_dim)[None, :, :] * 0.1, (k, 1, 1))
        
        # B: centers of shape [k, input_dim]
        B = jr.uniform(key_b, (k, input_dim), minval=0, maxval=1)
        
        splat_architectures.append((V, A, B))
    
    # Create KAN architectures with different sizes
    kan_architectures = []
    for hidden_neurons in [[10], [100], [300], [400], [20, 20]]:
        kan_model = KAN(
            layer_dims=[input_dim] + hidden_neurons + [1],
            layer_type='base',
            required_parameters={'k': 10, 'G': 5},
            seed=jr.PRNGKey(sum(hidden_neurons))
        )
        kan_architectures.append(kan_model)
    
    # Create MLP architectures with different sizes using nnx
    class SimpleMLP(nnx.Module):
        def __init__(self, hidden_dims, input_dim):
            super().__init__()
            self.hidden_dims = hidden_dims
            
            # First layer takes input_dim as in_features
            in_features = input_dim
            
            # Create hidden layers with proper dimensions
            self.layers = []
            for i, dim in enumerate(hidden_dims):
                layer = nnx.Linear(in_features=in_features, out_features=dim, rngs=nnx.Rngs(0))
                self.layers.append(layer)
                # Register layer as attribute to make it findable by nnx
                setattr(self, f"layer_{i}", layer)
                in_features = dim
                
            # Output layer with 1 output feature
            self.output_layer = nnx.Linear(in_features=in_features, out_features=1, rngs=nnx.Rngs(0))
            
        def __call__(self, x):
            for layer in self.layers:
                x = jax.nn.relu(layer(x))
            return self.output_layer(x)
    
    mlp_architectures = []
    for hidden_dims in [[200], [500], [1000], [200, 200], [500, 500]]:
        # Create model and initialize parameters
        mlp_model = SimpleMLP(hidden_dims, input_dim)
        
        mlp_architectures.append(mlp_model)
    
    # Create snapshot filename based on function and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = name
    snapshot_file = f"logs/regression_{function_name}_{name}_key={seed}_{timestamp}.pkl"
    
    # Warm up GPU for optimal performance
    warmup_gpu()

    
    train_size = 1000
    test_size = 200
    noise_level = 0.01
    num_steps = 20000
    validation_interval = 1000
    # Run comparison with snapshot support
    results = run_regression_comparison(
        train_size=train_size,
        test_size=test_size,
        function=target_function,
        noise_level=noise_level,
        input_dim=input_dim,
        splat_architectures=splat_architectures,
        kan_architectures=[],#kan_architectures,
        mlp_architectures=[],#mlp_architectures,
        num_steps=num_steps,
        validation_interval=validation_interval,
        adam=True,
        lr=1e-4,
        snapshot_file=snapshot_file,  # Add snapshot file
        function_name=function_name   # Add function name for identification
    )
    
    # Generate plots and table
    plot_training_loss(results)
    plot_validation_mse(results)
    plot_final_mse_vs_size(results)
    generate_timing_table(results)
    
    return results

if __name__ == "__main__":
    # Add command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description='Regression Model Training and Evaluation')
    parser.add_argument('--load', type=str, help='Load results from snapshot file')
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    
    # Run example usage with optional snapshot loading
    results = example_usage(args.load, args.name, args.seed)