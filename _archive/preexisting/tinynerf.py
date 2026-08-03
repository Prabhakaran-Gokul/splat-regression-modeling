'''
You are going to write an implementation of the tinynerf experiment where you will compare the performance of a few different network architectures. Let's walk through everything that will be involved.

1. **Data Preparation**: You will need to load and preprocess the dataset that contains the 3D models used in the tinynerf experiments. This includes loading the images, camera poses, and any other relevant data. Make sure to normalize the data appropriately for training. Create a method whose input is the number of training samples and test samples, and whose output is datasets of different renders of the given object from different angles, of appropriate sizes.
2. **Model Definition**: Define the neural network architectures that will be compared. First write two different methods that can be used to initialize MLPs and KANs respectively. These two methods should have a reasonable set of arguments for defining the topology of simple toy networks. Then, write a third method that can initialize a Splat network. This method should take as input the number of splats, and the dimensionality of the input space. It should also take optional arguments for initializing the splat parameters (e.g., means, variances, amplitudes). 
3. **Training Loop**: Implement a training loop that can train each model on the dataset. This should include:
   - Forward pass
   - Loss computation via renderer
   - Backward pass and parameter updates
   - include an optional 'verbose' argument that will display a progress bar using tqdm if set to True.
4. **Evaluation**: after training, you will need to evaluate the performance of each model on the test dataset. This should include:
   - Computing metrics such as PSNR, SSIM, or other relevant measures to quantify the quality of the rendered images.
   - Visualizing the results by rendering the test scenes with each model and comparing the outputs.
'''

import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax import nnx
import time
from tqdm import tqdm
import optax
from jaxkan.KAN import KAN
import pickle
from datetime import datetime
from pathlib import Path
from lib.splat import eval_splat, gd_splat_regression
from scipy.stats import norm

def generate_rays(H, W, focal, c2w):
    """Generate camera rays for pinhole camera."""
    i, j = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing='xy')
    
    # Normalized device coordinates in range [-1, 1]
    dirs = jnp.stack([(i - W * 0.5) / focal,
                     -(j - H * 0.5) / focal,
                     -jnp.ones_like(i)], axis=-1)
    
    # Rotate ray directions from camera to world
    rays_d = jnp.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    
    # Origin of all rays is the camera origin in world space
    rays_o = jnp.broadcast_to(c2w[:3, -1], rays_d.shape)
    
    return rays_o, rays_d

def positional_encoding(x, num_freqs=10, include_input=True):
    """Apply positional encoding to input coordinates."""
    if num_freqs == 0:
        return x
        
    encoded = []
    if include_input:
        encoded.append(x)
        
    # Apply sin(2^i * pi * x) and cos(2^i * pi * x) for i in [0, num_freqs-1]
    for i in range(num_freqs):
        freq = 2.0 ** i
        encoded.append(jnp.sin(freq * jnp.pi * x))
        encoded.append(jnp.cos(freq * jnp.pi * x))
        
    return jnp.concatenate(encoded, axis=-1)

# Define MLP using nnx - CORRECTED VERSION
class SimpleMLP(nnx.Module):
    def __init__(self, input_dim, hidden_dims, output_dims, activation=jax.nn.relu, rngs=None):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.activation = activation
        
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Create hidden layers with proper in_features and out_features
        self.layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layer = nnx.Linear(in_features=prev_dim, out_features=dim, rngs=rngs)
            self.layers.append(layer)
            # Register layer as attribute
            setattr(self, f"layer_{i}", layer)
            prev_dim = dim
            
        # Output layer
        self.output_layer = nnx.Linear(in_features=prev_dim, out_features=output_dims, rngs=rngs)
        
    def __call__(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

def render_rays(model, rays_o, rays_d, near, far, num_samples, rand=False):
    """Render rays by sampling along each and querying the model."""
    batch_size = rays_o.shape[0]
    
    
    # Generate sample points along each ray
    t = jnp.broadcast_to(jnp.linspace(near, far, num_samples)[None, :, None], (batch_size, num_samples, 1))
    if rand:
        # Stratified sampling with jittering within each bin
        key = jr.PRNGKey(0)
        t = t + jr.uniform(key, (batch_size, num_samples,1)) * (far - near) / num_samples
    
    
    
    # batch size x num_ray_samples x 3 

    # Points in space to evaluate model
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t
    # Encode points and directions
    pts_flat = pts.reshape(-1, 3)
    if isinstance(model, SimpleMLP):
        pts_encoded = positional_encoding(pts_flat, num_freqs=10)
    
    # Predict RGB and density for each point (in batches if needed)
    raw = model(pts_encoded)
    raw = raw.reshape(batch_size, num_samples, 4)
    
    # Extract RGB and density
    rgb = jax.nn.sigmoid(raw[..., :3])  # (B, S, 3)
    sigma = jax.nn.relu(raw[..., 3])    # (B, S)
    
    # Calculate the distance between adjacent samples
    delta = jnp.concatenate([
        t[:, 1:] - t[:, :-1],
        jnp.broadcast_to(1e10, (batch_size, 1, 1))
    ], axis=1)
    
    # Compute alpha values (opacity)
    alpha = 1.0 - jnp.exp(-sigma[:,:,None] * delta)
    
    # Compute weights for volume rendering
    weights = alpha * jnp.cumprod(jnp.concatenate([
        jnp.ones((batch_size, 1, 1)),
        1.0 - alpha + 1e-10
    ], axis=1), axis=1)[:, :-1]
    
    # Compute color for each ray by integrating along the ray
    rgb_map = jnp.sum(weights * rgb, axis=1)
    
    # Compute depth map as expected distance
    depth_map = jnp.sum(weights * t, axis=1)[:,0]
    
    return rgb_map, depth_map

def save_snapshot(snapshot_file, model_params, val_images, stats, config):
    """Save training snapshot to a file."""
    # Create directory if it doesn't exist
    Path(snapshot_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create snapshot
    snapshot = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_params': model_params,
        'val_images': val_images,
        'stats': stats,
        'config': config
    }
    
    # Save to file
    with open(snapshot_file, 'wb') as f:
        pickle.dump(snapshot, f)
    
    print(f"Snapshot saved to {snapshot_file}")

def load_snapshot(snapshot_file):
    """Load training snapshot from a file."""
    with open(snapshot_file, 'rb') as f:
        snapshot = pickle.load(f)
    
    print(f"Loaded snapshot from {snapshot_file} (created {snapshot['timestamp']})")
    return snapshot

def init_splat_model(input_dim, output_dim=4, num_splats=100, scale=0.1, key=None):
    """
    Initialize splat model parameters for NeRF
    
    Args:
        input_dim: Dimension of input data (after positional encoding)
        output_dim: Dimension of output data (4 for RGB + density)
        num_splats: Number of splat components
        scale: Initial scale for the covariance matrices
        key: JAX random key
        
    Returns:
        Tuple (V, A, B) representing splat parameters
    """
    if key is None:
        key = jr.PRNGKey(0)
        
    key_v, key_a, key_b = jr.split(key, 3)
    
    # Initialize weights (V) to zeros
    V = jnp.zeros((num_splats, output_dim))
    
    # Initialize covariance matrices (A) with diagonal matrices
    A = jnp.tile(jnp.eye(input_dim)[None, :, :] * scale, (num_splats, 1, 1))
    
    # Initialize centers (B) randomly in normalized space
    # For positional encoding, the inputs will be in sin/cos space, so we spread
    # the centers across [-1, 1] for each dimension
    B = jr.uniform(key_b, (num_splats, input_dim), minval=-1.0, maxval=1.0)
    
    return (V, A, B)

# Modify the train_nerf function to support splats
def train_nerf(height, width, focal_length, poses, images, 
               num_iters=5000, batch_size=4096, lr=5e-4, 
               near=2.0, far=6.0, num_samples=64, 
               model_type="mlp", hidden_dims=[256, 256, 256, 256],
               log_every=100, val_pose=None, val_image=None,
               snapshot_file=None, num_splats=100):
    """Train a NeRF model with snapshot saving."""
    # Flatten all rays and pixels for batching
    all_rays_o = []
    all_rays_d = []
    all_pixels = []
    
    for i in range(len(poses)):
        rays_o, rays_d = generate_rays(height, width, focal_length, poses[i])
        all_rays_o.append(rays_o.reshape(-1, 3))
        all_rays_d.append(rays_d.reshape(-1, 3))
        all_pixels.append(images[i].reshape(-1, 3))
    
    all_rays_o = jnp.concatenate(all_rays_o, axis=0)
    all_rays_d = jnp.concatenate(all_rays_d, axis=0)
    all_pixels = jnp.concatenate(all_pixels, axis=0)
    
    # Initialize model
    input_dim = positional_encoding(jnp.zeros((1, 3)), num_freqs=10).shape[1]
    
    if model_type.lower() == "mlp":
        model = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dims=4, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    elif model_type.lower() == "kan":
        model = KAN(
            layer_dims=[input_dim] + hidden_dims + [4],
            layer_type='base',
            required_parameters={'k': 10, 'G': 5},
            seed=jr.PRNGKey(42)
        )
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    elif model_type.lower() == "splat":
        # Initialize splat model
        model = init_splat_model(input_dim=input_dim, output_dim=4, num_splats=num_splats, key=jr.PRNGKey(42))
        # For splats, we'll use optax directly instead of nnx.Optimizer
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Define loss functions and gradient computation based on model type
    if model_type.lower() in ["mlp", "kan"]:
        # Loss function for neural network models
        def loss_fn(model, batch_rays_o, batch_rays_d, batch_pixels):
            def model_fn(x):
                return model(x)
            rgb_pred, _ = render_rays(model_fn, batch_rays_o, batch_rays_d, near, far, num_samples, rand=True)
            return jnp.mean((rgb_pred - batch_pixels) ** 2)
        
        # Use nnx.grad for gradient computation
        grad_fn = nnx.grad(loss_fn)
    else:  # splat model
        # Loss function for splat models
        def loss_fn(splat_params, batch_rays_o, batch_rays_d, batch_pixels):
            def model_fn(x):
                return eval_splat(x, splat_params)
            rgb_pred, _ = render_rays(model_fn, batch_rays_o, batch_rays_d, near, far, num_samples, rand=True)
            return jnp.mean((rgb_pred - batch_pixels) ** 2)
        
        # Use jax.value_and_grad for splat models
        grad_fn = jax.value_and_grad(loss_fn)
    
    # Training loop
    psnrs = []
    losses = []
    val_images = {}  # Store validation renders
    
    # Save configuration
    config = {
        'height': height,
        'width': width,
        'focal_length': focal_length,
        'num_iters': num_iters,
        'batch_size': batch_size,
        'lr': lr,
        'near': near,
        'far': far,
        'num_samples': num_samples,
        'model_type': model_type,
        'hidden_dims': hidden_dims,
        'num_splats': num_splats if model_type.lower() == "splat" else None,
    }
    
    # Create initial snapshot if requested
    if snapshot_file:
        if model_type.lower() in ["mlp", "kan"]:
            model_params = nnx.state(model, nnx.Param)
        else:
            model_params = model  # For splat, the model is already the parameters
        
        save_snapshot(
            snapshot_file, 
            model_params,
            val_images,
            {"losses": losses, "psnrs": psnrs},
            config
        )
    
    start_time = time.time()
    
    # Training loop with progress bar
    for i in tqdm(range(num_iters), desc=f"Training {model_type.upper()} NeRF"):
        # Random batch of rays
        idx = np.random.randint(0, all_pixels.shape[0], size=(batch_size,))
        batch_rays_o = all_rays_o[idx]
        batch_rays_d = all_rays_d[idx]
        batch_pixels = all_pixels[idx]
        
        # Compute gradients and update parameters based on model type
        if model_type.lower() in ["mlp", "kan"]:
            # Compute loss for logging
            batch_loss = loss_fn(model, batch_rays_o, batch_rays_d, batch_pixels)
            losses.append(float(batch_loss))
            
            # Compute gradients and update for neural networks
            grads = grad_fn(model, batch_rays_o, batch_rays_d, batch_pixels)
            optimizer.update(model, grads)
        else:  # splat model
            # Compute loss and gradients for splats
            batch_loss, grads = grad_fn(model, batch_rays_o, batch_rays_d, batch_pixels)
            losses.append(float(batch_loss))
            
            # Update parameters for splats
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
        
        # Log progress
        if i % log_every == 0 or i == num_iters - 1:
            psnr = -10.0 * jnp.log10(batch_loss)
            psnrs.append(float(psnr))
            print(f"Iteration {i}: Loss = {batch_loss:.6f}, PSNR = {psnr:.2f}")
            
            # Render validation image if provided
            if val_pose is not None and val_image is not None:
                val_rays_o, val_rays_d = generate_rays(height, width, focal_length, val_pose)
                val_rays_o = val_rays_o.reshape(-1, 3)
                val_rays_d = val_rays_d.reshape(-1, 3)
                
                # Define model function based on model type
                if model_type.lower() in ["mlp", "kan"]:
                    model_fn = lambda x: model(x)
                else:  # splat model
                    model_fn = lambda x: eval_splat(x, model)
                
                # Render in chunks to avoid OOM
                chunk_size = 4096
                num_chunks = val_rays_o.shape[0] // chunk_size + (val_rays_o.shape[0] % chunk_size != 0)
                val_rgb = []
                
                for j in range(num_chunks):
                    start = j * chunk_size
                    end = min((j + 1) * chunk_size, val_rays_o.shape[0])
                    chunk_rays_o = val_rays_o[start:end]
                    chunk_rays_d = val_rays_d[start:end]
                    chunk_rgb, _ = render_rays(model_fn, chunk_rays_o, chunk_rays_d, near, far, num_samples, rand=False)
                    val_rgb.append(chunk_rgb)
                
                val_rgb = jnp.concatenate(val_rgb, axis=0)
                val_rgb = val_rgb.reshape(height, width, 3)
                
                # Store validation render
                val_images[i] = np.array(val_rgb)
                
                # Display validation render
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(val_image)
                axes[0].set_title("Ground Truth")
                axes[1].imshow(val_rgb)
                axes[1].set_title(f"Iteration {i}")
                plt.tight_layout()
                plt.savefig(f"val_render_{i:04d}.png")
                plt.close()
                
                # Update snapshot
                if snapshot_file and (i % (log_every * 10) == 0 or i == num_iters - 1):
                    # Get model parameters based on model type
                    if model_type.lower() in ["mlp", "kan"]:
                        model_params = nnx.state(model, nnx.Param)
                    else:  # splat model
                        model_params = model
                        
                    save_snapshot(
                        snapshot_file, 
                        model_params,
                        val_images,
                        {"losses": losses, "psnrs": psnrs, 
                         "elapsed_time": time.time() - start_time},
                        config
                    )
    
    # Save final snapshot
    if snapshot_file:
        # Get model parameters based on model type
        if model_type.lower() in ["mlp", "kan"]:
            model_params = nnx.state(model, nnx.Param)
        else:  # splat model
            model_params = model
            
        save_snapshot(
            snapshot_file, 
            model_params,
            val_images,
            {"losses": losses, "psnrs": psnrs, 
             "elapsed_time": time.time() - start_time},
            config
        )
    
    return model, {"losses": losses, "psnrs": psnrs, "val_images": val_images}

def render_test_view(model, height, width, focal_length, test_pose, 
                    near=2.0, far=6.0, num_samples=64):
    """Render a test view from the model."""
    # Generate rays for test view
    rays_o, rays_d = generate_rays(height, width, focal_length, test_pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # Render in chunks to avoid OOM
    chunk_size = 4096
    num_chunks = rays_o.shape[0] // chunk_size + (rays_o.shape[0] % chunk_size != 0)
    rgb = []
    depth = []
    
    for j in range(num_chunks):
        start = j * chunk_size
        end = min((j + 1) * chunk_size, rays_o.shape[0])
        chunk_rays_o = rays_o[start:end]
        chunk_rays_d = rays_d[start:end]
        chunk_rgb, chunk_depth = render_rays(model, chunk_rays_o, chunk_rays_d, near, far, num_samples, rand=False)

        rgb.append(chunk_rgb)
        depth.append(chunk_depth)
    
    rgb = jnp.concatenate(rgb, axis=0).reshape(height, width, 3)
    depth = jnp.concatenate(depth, axis=0).reshape(height, width)
    
    return rgb, depth

def save_snapshot(snapshot_file, model_params, val_images, stats, config):
    """Save training snapshot to a file."""
    # Create directory if it doesn't exist
    Path(snapshot_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create snapshot
    snapshot = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_params': model_params,
        'val_images': val_images,
        'stats': stats,
        'config': config
    }
    
    # Save to file
    with open(snapshot_file, 'wb') as f:
        pickle.dump(snapshot, f)
    
    print(f"Snapshot saved to {snapshot_file}")

def load_snapshot(snapshot_file):
    """Load training snapshot from a file."""
    with open(snapshot_file, 'rb') as f:
        snapshot = pickle.load(f)
    
    print(f"Loaded snapshot from {snapshot_file} (created {snapshot['timestamp']})")
    return snapshot

def train_nerf(height, width, focal_length, poses, images, 
               num_iters=5000, batch_size=4096, lr=5e-4, 
               near=2.0, far=6.0, num_samples=64, 
               model_type="mlp", hidden_dims=[256, 256, 256, 256],
               log_every=100, val_pose=None, val_image=None,
               snapshot_file=None):
    """Train a NeRF model with snapshot saving."""
    # Flatten all rays and pixels for batching
    all_rays_o = []
    all_rays_d = []
    all_pixels = []
    
    for i in range(len(poses)):
        rays_o, rays_d = generate_rays(height, width, focal_length, poses[i])
        all_rays_o.append(rays_o.reshape(-1, 3))
        all_rays_d.append(rays_d.reshape(-1, 3))
        all_pixels.append(images[i].reshape(-1, 3))
    
    all_rays_o = jnp.concatenate(all_rays_o, axis=0)
    all_rays_d = jnp.concatenate(all_rays_d, axis=0)
    all_pixels = jnp.concatenate(all_pixels, axis=0)
    
    # Initialize model with correct dimensions
    input_dim = positional_encoding(jnp.zeros((1, 3)), num_freqs=10).shape[1]
    
    if model_type.lower() == "mlp":
        # CORRECTED: Pass input_dim explicitly
        model = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dims=4, rngs=nnx.Rngs(0))
    elif model_type.lower() == "kan":
        # KAN already handles dimensions correctly
        model = KAN(
            layer_dims=[input_dim] + hidden_dims + [4],
            layer_type='base',
            required_parameters={'k': 10, 'G': 5},
            seed=jr.PRNGKey(42)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize optimizer with nnx.Optimizer
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    
    # Define loss function
    def loss_fn(model, batch_rays_o, batch_rays_d, batch_pixels):
        # Fixed: Call model directly
        rgb_pred, _ = render_rays(model, batch_rays_o, batch_rays_d, near, far, num_samples, rand=True)
        return jnp.mean((rgb_pred - batch_pixels) ** 2)
    
    # Use nnx.grad for gradient computation
    grad_fn = nnx.grad(loss_fn)
    
    # Training loop
    psnrs = []
    losses = []
    val_images = {}  # Store validation renders
    
    # Save configuration
    config = {
        'height': height,
        'width': width,
        'focal_length': focal_length,
        'num_iters': num_iters,
        'batch_size': batch_size,
        'lr': lr,
        'near': near,
        'far': far,
        'num_samples': num_samples,
        'model_type': model_type,
        'hidden_dims': hidden_dims,
    }
    
    # Create initial snapshot if requested
    if snapshot_file:
        save_snapshot(
            snapshot_file, 
            nnx.state(model, nnx.Param),
            val_images,
            {"losses": losses, "psnrs": psnrs},
            config
        )
    
    start_time = time.time()
    
    # Training loop with progress bar
    for i in tqdm(range(num_iters), desc=f"Training {model_type.upper()} NeRF"):
        # Random batch of rays
        idx = np.random.randint(0, all_pixels.shape[0], size=(batch_size,))
        batch_rays_o = all_rays_o[idx]
        batch_rays_d = all_rays_d[idx]
        batch_pixels = all_pixels[idx]
        
        # Compute loss (for logging)
        batch_loss = loss_fn(model, batch_rays_o, batch_rays_d, batch_pixels)
        losses.append(float(batch_loss))
        
        # Compute gradients and update
        grads = grad_fn(model, batch_rays_o, batch_rays_d, batch_pixels)
        optimizer.update(model, grads)
        
        # Log progress
        if i % log_every == 0 or i == num_iters - 1:
            psnr = -10.0 * jnp.log10(batch_loss)
            psnrs.append(float(psnr))
            print(f"Iteration {i}: Loss = {batch_loss:.6f}, PSNR = {psnr:.2f}")
            
            # Render validation image if provided
            if val_pose is not None and val_image is not None:
                val_rays_o, val_rays_d = generate_rays(height, width, focal_length, val_pose)
                val_rays_o = val_rays_o.reshape(-1, 3)
                val_rays_d = val_rays_d.reshape(-1, 3)
                
                # Render in chunks to avoid OOM
                chunk_size = 4096
                num_chunks = val_rays_o.shape[0] // chunk_size + (val_rays_o.shape[0] % chunk_size != 0)
                val_rgb = []
                
                for j in range(num_chunks):
                    start = j * chunk_size
                    end = min((j + 1) * chunk_size, val_rays_o.shape[0])
                    chunk_rays_o = val_rays_o[start:end]
                    chunk_rays_d = val_rays_d[start:end]
                    chunk_rgb, _ = render_rays(model, chunk_rays_o, chunk_rays_d, near, far, num_samples, rand=False)
                    val_rgb.append(chunk_rgb)
                
                val_rgb = jnp.concatenate(val_rgb, axis=0)
                val_rgb = val_rgb.reshape(height, width, 3)
                
                # Store validation render
                val_images[i] = np.array(val_rgb)
                
                # Display validation render
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(val_image)
                axes[0].set_title("Ground Truth")
                axes[1].imshow(val_rgb)
                axes[1].set_title(f"Iteration {i}")
                plt.tight_layout()
                plt.savefig(f"val_render_{i:04d}.png")
                plt.close()
                
                # Update snapshot
                if snapshot_file and (i % (log_every * 10) == 0 or i == num_iters - 1):
                    save_snapshot(
                        snapshot_file, 
                        nnx.state(model, nnx.Param),
                        val_images,
                        {"losses": losses, "psnrs": psnrs, 
                         "elapsed_time": time.time() - start_time},
                        config
                    )
    
    # Save final snapshot
    if snapshot_file:
        save_snapshot(
            snapshot_file, 
            nnx.state(model, nnx.Param),
            val_images,
            {"losses": losses, "psnrs": psnrs, 
             "elapsed_time": time.time() - start_time},
            config
        )
    
    return model, {"losses": losses, "psnrs": psnrs, "val_images": val_images}

def render_test_view(model, height, width, focal_length, test_pose, 
                    near=2.0, far=6.0, num_samples=64):
    """Render a test view from the model."""
    # Generate rays for test view
    rays_o, rays_d = generate_rays(height, width, focal_length, test_pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # Render in chunks to avoid OOM
    chunk_size = 4096
    num_chunks = rays_o.shape[0] // chunk_size + (rays_o.shape[0] % chunk_size != 0)
    rgb = []
    depth = []
    
    for j in range(num_chunks):
        start = j * chunk_size
        end = min((j + 1) * chunk_size, rays_o.shape[0])
        chunk_rays_o = rays_o[start:end]
        chunk_rays_d = rays_d[start:end]
        chunk_rgb, chunk_depth = render_rays(model, chunk_rays_o, chunk_rays_d, near, far, num_samples, rand=False)
        rgb.append(chunk_rgb)
        depth.append(chunk_depth)
    
    rgb = jnp.concatenate(rgb, axis=0).reshape(height, width, 3)
    depth = jnp.concatenate(depth, axis=0).reshape(height, width)
    
    return rgb, depth

def restore_model_from_snapshot(snapshot_file):
    """Restore model and data from a snapshot file."""
    snapshot = load_snapshot(snapshot_file)
    model_params = snapshot['model_params']
    val_images = snapshot['val_images']
    stats = snapshot['stats']
    config = snapshot['config']
    
    # Recreate model architecture
    input_dim = positional_encoding(jnp.zeros((1, 3)), num_freqs=10).shape[1]
    
    if config['model_type'].lower() == "mlp":
        # CORRECTED: Pass input_dim explicitly
        model = SimpleMLP(input_dim=input_dim, hidden_dims=config['hidden_dims'], output_dims=4, rngs=nnx.Rngs(0))
    elif config['model_type'].lower() == "kan":
        model = KAN(
            layer_dims=[input_dim] + config['hidden_dims'] + [4],
            layer_type='base',
            required_parameters={'k': 10, 'G': 5},
            seed=jr.PRNGKey(42)
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Restore model parameters
    nnx.update(model, model_params)
    
    return model, val_images, stats, config

def display_results_from_snapshot(snapshot_file, render_novel_view=True):
    """Display results from a saved snapshot."""
    model, val_images, stats, config = restore_model_from_snapshot(snapshot_file)
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(stats["psnrs"])
    plt.xlabel("Logging Step")
    plt.ylabel("PSNR")
    plt.title(f"Training Progress - {config['model_type'].upper()} Model")
    plt.grid(True)
    plt.savefig(f"{Path(snapshot_file).stem}_training_curve.png")
    plt.show()
    
    # Show validation images progression
    num_images = min(len(val_images), 5)  # Show at most 5 images
    iterations = sorted(val_images.keys())
    selected_iterations = [iterations[i] for i in np.linspace(0, len(iterations)-1, num_images).astype(int)]
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images*4, 4))
    if num_images == 1:
        axes = [axes]  # Make iterable if only one image
        
    for i, iter_num in enumerate(selected_iterations):
        axes[i].imshow(val_images[iter_num])
        axes[i].set_title(f"Iteration {iter_num}")
        axes[i].axis('off')
        
    plt.suptitle(f"Validation Renders - {config['model_type'].upper()} Model")
    plt.tight_layout()
    plt.savefig(f"{Path(snapshot_file).stem}_validation_renders.png")
    plt.show()
    
    # Render novel view if requested
    if render_novel_view:
        # Create a novel viewpoint
        theta = np.pi/4  # 45 degrees
        test_pose = jnp.array([
            [np.cos(theta), -np.sin(theta), 0, 3*np.cos(theta)],
            [np.sin(theta), np.cos(theta), 0, 3*np.sin(theta)],
            [0, 0, 1, 0.5]
        ])
        
        # Render novel view
        rgb, depth = render_test_view(model, 
                                      config['height'], config['width'], 
                                      config['focal_length'], test_pose,
                                      near=config['near'], far=config['far'], 
                                      num_samples=config['num_samples'])
        
        # Visualize results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("Novel View RGB")
        axes[1].imshow(depth, cmap='turbo')
        axes[1].set_title("Novel View Depth")
        plt.suptitle(f"Novel View Render - {config['model_type'].upper()} Model")
        plt.tight_layout()
        plt.savefig(f"{Path(snapshot_file).stem}_novel_view.png")
        plt.show()
    
    return model, val_images, stats, config

def example_usage(load_from=None):
    """Example usage of the TinyNeRF implementation."""
    if load_from:
        print(f"Loading from snapshot: {load_from}")
        return display_results_from_snapshot(load_from)
    
    # Load data (mock example)
    height, width = 100, 100  # small for example
    focal_length = 50.0
    
    # Create dummy poses and images for demonstration
    n_views = 20
    key = jr.PRNGKey(42)
    
    # Generate random camera poses in a circle
    angles = jnp.linspace(0, 2*jnp.pi, n_views)
    poses = []
    for angle in angles:
        # Simple rotation around z-axis with fixed radius
        c2w = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle), 0, 3*jnp.cos(angle)],
            [jnp.sin(angle), jnp.cos(angle), 0, 3*jnp.sin(angle)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        poses.append(c2w[:3, :])
    poses = jnp.stack(poses, axis=0)
    
    # Generate simple colored sphere images (very simplified)
    images = []
    for i in range(n_views):
        # Just a placeholder - real images would come from dataset
        image = jnp.ones((height, width, 3)) * 0.5  # Gray image
        images.append(image)
    images = jnp.stack(images, axis=0)
    
    # Create snapshot filename based on model type and time
    model_type = "mlp"  # or "kan"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = f"nerf_snapshot_{model_type}_{timestamp}.pkl"
    
    # Train the model
    val_pose = poses[0]  # Use first pose as validation
    val_image = images[0]
    
    model, stats = train_nerf(
        height, width, focal_length, poses, images,
        num_iters=1000,  # Small number for example
        batch_size=1024,
        lr=5e-4,
        near=2.0, far=6.0,
        num_samples=32,
        model_type=model_type,
        hidden_dims=[64, 64],  # Small network for example
        log_every=100,
        val_pose=val_pose,
        val_image=val_image,
        snapshot_file=snapshot_file
    )
    
    # Render novel view
    test_pose = jnp.array([
        [1.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ])
    
    rgb, depth = render_test_view(model, height, width, focal_length, test_pose)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Rendered RGB")
    axes[1].imshow(depth, cmap='turbo')
    axes[1].set_title("Rendered Depth")
    plt.tight_layout()
    plt.savefig("test_render.png")
    plt.show()
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(stats["psnrs"])
    plt.xlabel("Logging Step")
    plt.ylabel("PSNR")
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig("training_curve.png")
    plt.show()
    
    return model, stats, snapshot_file

if __name__ == "__main__":
    # Add command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description='TinyNeRF Training')
    parser.add_argument('--load', type=str, default=None, 
                       help='Load from snapshot file')
    args = parser.parse_args()
    
    try:
        import json
        import imageio
        import cv2
        if args.load:
            model, val_images, stats, config = example_usage(load_from=args.load)
        else:
            model, stats, snapshot_file = example_usage()
    except ImportError:
        print("Warning: Some dependencies not found. Full data loading requires json, imageio, and cv2.")
        print("Running simplified example...")
        if args.load:
            model, val_images, stats, config = example_usage(load_from=args.load)
        else:
            model, stats, snapshot_file = example_usage()