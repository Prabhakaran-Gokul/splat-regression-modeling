import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
from flax import nnx
from jaxkan.KAN import KAN
import optax
from functools import partial
import pickle
from datetime import datetime
import os
from pathlib import Path
import cgls_solver as cgls
import scipy, scipy.interpolate
from flax import nnx
from jaxkan.KAN import KAN
import scipy as sp 
import scipy.sparse

# Import model functions from existing files
from v2.lib.splat import eval_splat, gd_splat_regression
from v2.lib.nets import gd_net_regression

class PDEProblem:
    """Base class for PDE problems with analytical solutions"""
    
    def __init__(self, name, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0]):
        self.name = name
        self.xlim = xlim  # [xmin, xmax]
        self.ylim = ylim  # [ymin, ymax]
    
    def solution(self, x):
        """Analytical solution function"""
        raise NotImplementedError
    
    def pde_residual(self, x, u, grad_u, laplacian_u):
        """PDE residual function (should be zero when PDE is satisfied)"""
        raise NotImplementedError
    
    def boundary_condition(self, x):
        """Boundary condition function"""
        raise NotImplementedError
    
    def is_boundary(self, x):
        """Function to check if a point is on the boundary"""
        # Check if any coordinate is on the boundary of the rectangle
        on_x_boundary = (x[:, 0] == self.xlim[0]) | (x[:, 0] == self.xlim[1])
        on_y_boundary = (x[:, 1] == self.ylim[0]) | (x[:, 1] == self.ylim[1])
        return (on_x_boundary | on_y_boundary)[:, None]


class ComplexGinzbergLandauProblem(PDEProblem): 
    def __init__(self,
                 N=512, 
                 h=0.1, 
                 L=300,
                 b=0.5,
                 c=-1.76,
                 iters=2048,
                 initial_condition=None, 
                 mode=1
                 ):
        
        '''
        different modes - no gradients are applied to the forcing term
        mode 0: forcing trm equals zero 
        mode 1: forcing term is zeroth order nonlinear
        mode 2: forcing term is the linear derivative operator
        '''
        xlim = [-1, 1]
        ylim = [0, (iters-1) * h]
        super().__init__("ComplexGinzbergLandau", xlim=xlim, ylim=ylim)
        self.coeffs = (L,b,c)
        if initial_condition is None:
            initial_condition = lambda x: 10**(-3) * np.sin(np.pi * (x+1))

        self.initial_condition = initial_condition
        ut, interp = self._itersolve(N, h, iters)
        self._ut = ut
        self.interpolator = interp
        self.mode = mode
        self.x_bdy = [-1, 1]
        self.t_bdy = [0, iters * h]

    def solution(self, x):
        return self.interpolator(x)
   
    def source_term(self, u, grad_u, dxdx_u):
        _, b,c = self.coeffs
        # grad_u[:,1] - (1+ib) dxdx_u = u - (1+ic)|u|^2 u
        if self.mode == 0: 
            return jnp.zeros_like(u)
        elif self.mode == 1: 
            return u - (1 + 1j*c) * u * jnp.conj(u) * u 
        elif self.mode == 2: 
            return grad_u[:,1] - (1 + 1j*b) * dxdx_u  
    
    def pde_residual(self, u, grad_u, dxdx_u):
        _, b, c = self.coeffs
        if self.mode == 0: 
            return u - (1 + 1j*c) * u * jnp.conj(u) * u - grad_u[:,1] + (1 + 1j*b) * dxdx_u 
        elif self.mode == 1: 
            return grad_u[:,1] - (1 + 1j*b) * dxdx_u   
        elif self.mode == 2: 
            return u - (1 + 1j*c) * u * jnp.conj(u) * u  
    
    def boundary_condition(self, x):
        """Dirichlet boundary condition (u = 0 on boundary)"""
        bdy = lambda x,t: self.initial_condition(x) * (t==0)
        return bdy(x[:,0], x[:,1])
    
    def _itersolve(self, N, h, iters):
        L,b,c = self.coeffs
        xn = cgls.gridpts(N)
        
        LHS = cgls.CGLE_LHS(N, h, L, b, c)
        LHS_lu = sp.sparse.linalg.splu(LHS.T)

        _ut = []
        An = None
        Fn = cgls.CGLE_RHS(cgls.dcht(10**(-3) * np.sin(np.pi * (xn+1))), h, L, b, c)

        for n in range(iters):
            An = LHS_lu.solve(Fn, trans='T')
            Fn = cgls.CGLE_RHS(An, h, L, b, c)
            if(n % 1 == 0):
                _ut.append(cgls.idcht(An))
        ut = np.stack(_ut, axis=1)

        tn = h*np.arange(0,iters)
        interp = scipy.interpolate.RegularGridInterpolator((xn, tn), ut)
        return jnp.array(ut), interp

    def compute_derivatives(self, model_fn, points):
        u = model_fn(points)
        grad_fn = jax.grad(lambda x_single: model_fn(x_single[None, :])[0, 0])
        grad_u = jax.vmap(grad_fn)(points)

        def dxdx_u(x_single):
            hessian = jax.hessian(lambda x_s: model_fn(x_s[None, :])[0, 0])(x_single)
            return hessian[0,0]

        dxdx_u = jax.vmap(dxdx_u)(points)

        return u, grad_u, dxdx_u[:, None]


class PoissonProblem(PDEProblem):
    """2D Poisson equation problem: -∇²u = f"""
    
    def __init__(self, xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
        super().__init__("Poisson", xlim=xlim, ylim=ylim)
    
    def solution(self, x):
        """Analytical solution: u(x,y) = sin(πx)sin(πy)"""
        return jnp.sin(jnp.pi * x[:, 0:1]) * jnp.sin(jnp.pi * x[:, 1:2])
    
    def source_term(self, x):
        """Right hand side f(x,y) = 2π²sin(πx)sin(πy)"""
        return 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0:1]) * jnp.sin(jnp.pi * x[:, 1:2])
    
    def pde_residual(self, x, u, grad_u, laplacian_u):
        """Residual of -∇²u = f"""
        return -laplacian_u - self.source_term(x)
    
    def boundary_condition(self, x):
        """Dirichlet boundary condition (u = 0 on boundary)"""
        return jnp.zeros((x.shape[0], 1))


class AdvectionDiffusionProblem(PDEProblem):
    """2D Advection-Diffusion problem: -ε∇²u + v⋅∇u = 0"""
    
    def __init__(self, epsilon=0.01, velocity=(1.0, 1.0), xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
        super().__init__("AdvectionDiffusion", xlim=xlim, ylim=ylim)
        self.epsilon = epsilon
        self.velocity = jnp.array(velocity)
    
    def solution(self, x):
        """Analytical solution with boundary layers"""
        exp_term1 = jnp.exp((x[:, 0:1] - self.xlim[0]) * self.velocity[0] / self.epsilon)
        exp_term2 = jnp.exp((x[:, 1:2] - self.ylim[0]) * self.velocity[1] / self.epsilon)
        denom1 = jnp.exp((self.xlim[1] - self.xlim[0]) * self.velocity[0] / self.epsilon) - 1
        denom2 = jnp.exp((self.ylim[1] - self.ylim[0]) * self.velocity[1] / self.epsilon) - 1
        
        return (exp_term1 - 1) / denom1 * (exp_term2 - 1) / denom2
    
    def pde_residual(self, x, u, grad_u, laplacian_u):
        """Residual of -ε∇²u + v⋅∇u = 0"""
        advection_term = self.velocity[0] * grad_u[:, 0:1] + self.velocity[1] * grad_u[:, 1:2]
        return -self.epsilon * laplacian_u + advection_term
    
    def boundary_condition(self, x):
        """Dirichlet boundary conditions"""
        return self.solution(x)


class AllenCahnProblem(PDEProblem):
    """2D steady-state Allen-Cahn equation: ε²∇²u + u - u³ = 0"""
    
    def __init__(self, epsilon=0.1, xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
        super().__init__("AllenCahn", xlim=xlim, ylim=ylim)
        self.epsilon = epsilon
    
    def solution(self, x):
        """
        Manufactured analytical solution: tanh((x + y - 1) / (sqrt(2) * epsilon))
        This represents a smooth transition between phases along the line x + y = 1
        """
        z = (x[:, 0:1] + x[:, 1:2] - 1.0) / (jnp.sqrt(2) * self.epsilon)
        return jnp.tanh(z)
    
    def pde_residual(self, x, u, grad_u, laplacian_u):
        """Residual of ε²∇²u + u - u³ = 0"""
        return self.epsilon**2 * laplacian_u + u - u**3
    
    def boundary_condition(self, x):
        """Dirichlet boundary condition from the analytical solution"""
        return self.solution(x)


class BurgersProblem(PDEProblem):
    """2D steady-state Burgers' equation for scalar field: u∇u = ν∇²u"""
    
    def __init__(self, nu=0.01, xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
        super().__init__("Burgers", xlim=xlim, ylim=ylim)
        self.nu = nu
    
    def solution(self, x):
        """
        Manufactured analytical solution for scalar Burgers equation:
        u(x,y) = 1 - tanh((x + y - 1)/(2*nu))
        
        This represents a smooth shock wave along the line x + y = 1
        """
        z = (x[:, 0:1] + x[:, 1:2] - 1.0) / (2 * self.nu)
        return 1 - jnp.tanh(z)
    
    def pde_residual(self, x, u, grad_u, laplacian_u):
        """
        Residual of u∇u - ν∇²u = 0 for scalar field u
        For scalar field: u∇u = u * (∂u/∂x + ∂u/∂y)
        """
        advection_term = u * (grad_u[:, 0:1] + grad_u[:, 1:2])
        diffusion_term = self.nu * laplacian_u
        return advection_term - diffusion_term
    
    def boundary_condition(self, x):
        """Dirichlet boundary condition from the analytical solution"""
        return self.solution(x)


def generate_points(key, n_interior, n_boundary, xlim=[0,1], ylim=[0,1]):
    """
    Generate interior and boundary points in the 2D domain
    
    Args:
        key: JAX random key
        n_interior: number of interior points to generate
        n_boundary: number of boundary points to generate
        xlim: [xmin, xmax] boundaries for x dimension
        ylim: [ymin, ymax] boundaries for y dimension
        
    Returns:
        interior_points: array of shape [n_interior, 2]
        boundary_points: array of shape [n_boundary, 2]
    """
    key_interior, key_boundary = jr.split(key)
    
    # Generate interior points uniformly inside the domain
    x_interior = jr.uniform(key_interior, (n_interior, 1), minval=xlim[0], maxval=xlim[1])
    y_interior = jr.uniform(jr.split(key_interior)[1], (n_interior, 1), minval=ylim[0], maxval=ylim[1])
    interior_points = jnp.hstack([x_interior, y_interior])
    
    # Generate boundary points (distributed on the four edges)
    n_per_edge = n_boundary // 4
    remaining = n_boundary - 4 * n_per_edge
    
    keys = jr.split(key_boundary, 4)
    
    # Left edge (x=xlim[0], y in [ylim[0], ylim[1]])
    left_y = jr.uniform(keys[0], (n_per_edge, 1), minval=ylim[0], maxval=ylim[1])
    left_edge = jnp.hstack([jnp.full((n_per_edge, 1), xlim[0]), left_y])
    
    # Right edge (x=xlim[1], y in [ylim[0], ylim[1]])
    right_y = jr.uniform(keys[1], (n_per_edge, 1), minval=ylim[0], maxval=ylim[1])
    right_edge = jnp.hstack([jnp.full((n_per_edge, 1), xlim[1]), right_y])
    
    # Bottom edge (y=ylim[0], x in [xlim[0], xlim[1]])
    bottom_x = jr.uniform(keys[2], (n_per_edge, 1), minval=xlim[0], maxval=xlim[1])
    bottom_edge = jnp.hstack([bottom_x, jnp.full((n_per_edge, 1), ylim[0])])
    
    # Top edge (y=ylim[1], x in [xlim[0], xlim[1]])
    top_x = jr.uniform(keys[3], (n_per_edge + remaining, 1), minval=xlim[0], maxval=xlim[1])
    top_edge = jnp.hstack([top_x, jnp.full((n_per_edge + remaining, 1), ylim[1])])
    
    # TODO: REMOVE THIS HACK 
    # boundary_points = jnp.vstack([left_edge, right_edge, bottom_edge, top_edge])
    boundary_points = jnp.vstack([left_edge, bottom_edge, top_edge])
    
    
    return interior_points, boundary_points


def compute_derivatives(model_fn, x):
    """
    Compute model output, gradient and Laplacian at points x
    
    Args:
        model_fn: Function that takes x and returns u
        x: Input points of shape [n, 2]
        
    Returns:
        u: Model output of shape [n, 1]
        grad_u: Gradient of shape [n, 2]
        laplacian_u: Laplacian of shape [n, 1]
    """
    # Model output
    u = model_fn(x)
    
    # Gradient computation using JAX
    grad_fn = jax.grad(lambda x_single: model_fn(x_single[None, :])[0, 0])
    grad_u = jax.vmap(grad_fn)(x)
    
    # Laplacian computation using JAX (trace of Hessian)
    def laplacian_single(x_single):
        hessian = jax.hessian(lambda x_s: model_fn(x_s[None, :])[0, 0])(x_single)
        return jnp.trace(hessian)
    
    laplacian_u = jax.vmap(laplacian_single)(x)
    
    return u, grad_u, laplacian_u[:, None]


def compute_pinn_loss(model_fn, problem, interior_points, boundary_points, physics_weight=1.0):
    """
    Compute PINN loss with both physics and boundary conditions
    
    Args:
        model_fn: Function that takes x and returns u
        problem: PDEProblem instance
        interior_points: Interior domain points of shape [n_interior, 2]
        boundary_points: Boundary points of shape [n_boundary, 2]
        physics_weight: Weight for the physics residual term
        
    Returns:
        total_loss: Combined physics and boundary loss
        physics_loss: PDE residual loss
        boundary_loss: Boundary condition loss
    """
    if isinstance(problem, ComplexGinzbergLandauProblem):
        u_interior, grad_u_interior, dxdx_u_interior = problem.compute_derivatives(model_fn, interior_points)
        residual = problem.pde_residual(u_interior, grad_u_interior, dxdx_u_interior)
        u_boundary = model_fn(boundary_points)
    else:
        u_interior, grad_u_interior, laplacian_u_interior = compute_derivatives(model_fn, interior_points)
        residual = problem.pde_residual(interior_points, u_interior, grad_u_interior, laplacian_u_interior)
        u_boundary = model_fn(boundary_points)

    physics_loss = jnp.mean(jnp.abs(residual)**2)
    
    bc_target = problem.boundary_condition(boundary_points)
    boundary_loss = jnp.mean(jnp.abs(u_boundary - bc_target)**2)
    total_loss = boundary_loss + physics_weight * physics_loss
    
    return total_loss, physics_loss, boundary_loss


# Add this helper function to sample mini-batches
def sample_minibatch(key, points, batch_size):
    """Sample a random mini-batch from the given points"""
    batch_size = min(batch_size, len(points))
    idx = jr.choice(key, len(points), (batch_size,), replace=False)
    return points[idx]


# Modified splat PINN training with mini-batches
def train_and_evaluate_splat_pinn(init_splat, problem, interior_points, boundary_points, test_points,
                                 num_steps, lr, adam, validation_interval, physics_weight=1.0,
                                 batch_size=100):
    """
    Train and evaluate a splat model for PINN with mini-batching
    
    Args:
        init_splat: Initial splat parameters (V, A, B)
        problem: PDEProblem instance
        interior_points: Interior domain points
        boundary_points: Boundary points
        test_points: Test points for evaluation
        num_steps: Number of training steps
        lr: Learning rate
        adam: Whether to use Adam optimizer
        validation_interval: Interval for validation
        physics_weight: Weight for the physics residual term
        batch_size: Size of mini-batches for training
        
    Returns:
        dict: Results containing losses and metrics
    """
    start_time = time.time()
    
    # Initialize results
    train_total_losses = []
    train_physics_losses = []
    train_boundary_losses = []
    val_steps = []
    val_errors = []
    
    # Current parameters
    curr_splat = init_splat
    
    # Set up optimizer if using Adam
    if adam:
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(curr_splat)
    
    # Create master PRNG key for reproducible sampling
    master_key = jr.PRNGKey(0)
    
    # Training loop with validation at intervals
    for step in tqdm(range(num_steps), desc="Training Splat PINN"):
        # Generate new random key for this step
        master_key, step_key = jr.split(master_key)
        key_interior, key_boundary = jr.split(step_key)
        
        # Sample mini-batches - use smaller batch for boundary points
        interior_batch_size = min(batch_size, len(interior_points))
        boundary_batch_size = min(batch_size // 4, len(boundary_points))
        
        interior_batch = sample_minibatch(key_interior, interior_points, interior_batch_size)
        boundary_batch = sample_minibatch(key_boundary, boundary_points, boundary_batch_size)
        
        # Define mini-batch loss function for this step
        def loss_fn(params):
            def step_model_fn(x):
                return eval_splat(x, params)
            
            total_loss, physics_loss, boundary_loss = compute_pinn_loss(
                step_model_fn, problem, interior_batch, boundary_batch, physics_weight)
            return total_loss, (physics_loss, boundary_loss)
        
        # Compute gradients
        (total_loss, (physics_loss, boundary_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(curr_splat)
        
        # Store losses
        train_total_losses.append(float(total_loss))
        train_physics_losses.append(float(physics_loss))
        train_boundary_losses.append(float(boundary_loss))
        
        # Update parameters
        if adam:
            updates, opt_state = optimizer.update(grads, opt_state, curr_splat)
            curr_splat = optax.apply_updates(curr_splat, updates)
        else:
            # Standard SGD
            V, A, B = curr_splat
            V_grad, A_grad, B_grad = grads
            V = V - lr * V_grad
            A = A - lr * A_grad
            B = B - lr * B_grad
            curr_splat = (V, A, B)
        
        # Validate at specified intervals (use full test set)
        if step % validation_interval == 0 or step == num_steps - 1:
            val_steps.append(step)
            
            # Compute test error against analytical solution
            test_pred = eval_splat(test_points, curr_splat)
            test_true = problem.solution(test_points)
            test_error = jnp.mean(jnp.abs(test_pred - test_true)**2)
            val_errors.append(float(test_error))
            
            # Display current status
            if step % (validation_interval ) == 0 or step == num_steps - 1:
                print(f"Step {step}: Loss = {total_loss:.10f}, Test MSE = {test_error:.6f}")
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    return {
        'train_total_losses': train_total_losses,
        'train_physics_losses': train_physics_losses,
        'train_boundary_losses': train_boundary_losses,
        'val_steps': val_steps,
        'val_errors': val_errors,
        'final_val_error': val_errors[-1],
        'wall_time': wall_time,
        'final_params': curr_splat
    }


# Modified neural network PINN training with mini-batches
def train_and_evaluate_nn_pinn(model, problem, interior_points, boundary_points, test_points,
                              num_steps, lr, adam, validation_interval, physics_weight=1.0,
                              batch_size=100):
    """
    Train and evaluate a neural network model (KAN or MLP) for PINN
    """
    start_time = time.time()
    
    # Initialize results
    train_total_losses = []
    train_physics_losses = []
    train_boundary_losses = []
    val_steps = []
    val_errors = []
    
    # Set up optimizer
    if adam:
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    else:
        optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)
    
    # Create master PRNG key for reproducible sampling
    master_key = jr.PRNGKey(0)
    
    # Training loop with validation at intervals
    for step in tqdm(range(num_steps), desc=f"Training {'KAN' if 'KAN' in str(type(model)) else 'MLP'} PINN"):
        # Generate new random key for this step
        master_key, step_key = jr.split(master_key)
        key_interior, key_boundary = jr.split(step_key)
        
        # Sample mini-batches - use smaller batch for boundary points
        interior_batch_size = min(batch_size, len(interior_points))
        boundary_batch_size = min(batch_size // 4, len(boundary_points))
        
        interior_batch = sample_minibatch(key_interior, interior_points, interior_batch_size)
        boundary_batch = sample_minibatch(key_boundary, boundary_points, boundary_batch_size)
        
        # Define mini-batch loss function for this step
        def loss_fn(model):
            def step_model_fn(x):
                return model(x)
            
            total_loss, physics_loss, boundary_loss = compute_pinn_loss(
                step_model_fn, problem, interior_batch, boundary_batch, physics_weight)
            return total_loss, (physics_loss, boundary_loss)
        
        # Compute gradients
        (total_loss, (physics_loss, boundary_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True)(model)
        
        # Store losses
        train_total_losses.append(float(total_loss))
        train_physics_losses.append(float(physics_loss))
        train_boundary_losses.append(float(boundary_loss))
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Validate at specified intervals (use full test set)
        if step % validation_interval == 0 or step == num_steps - 1:
            val_steps.append(step)
            
            # Compute test error against analytical solution
            test_pred = model(test_points)
            test_true = problem.solution(test_points)
            test_error = jnp.mean(jnp.abs(test_pred - test_true)**2)
            val_errors.append(float(test_error))
            
            # Display current status
            if step % (validation_interval * 10) == 0 or step == num_steps - 1:
                print(f"Step {step}: Loss = {total_loss}, Test MSE = {test_error:.6f}")
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Save final model parameters
    final_params = nnx.state(model, nnx.Param)
    
    # Generate final prediction on test points for visualization
    test_pred = model(test_points)
    
    return {
        'train_total_losses': train_total_losses,
        'train_physics_losses': train_physics_losses,
        'train_boundary_losses': train_boundary_losses,
        'val_steps': val_steps,
        'val_errors': val_errors,
        'final_val_error': val_errors[-1],
        'wall_time': wall_time,
        'final_params': final_params,  # Store the final parameters
        'test_pred': test_pred         # Store final predictions for visualization
    }


# Update the run_pinn_comparison function to support batch_size parameter
def run_pinn_comparison(problem, n_interior, n_boundary, n_test,
                       splat_architectures, kan_architectures, mlp_architectures,
                       num_steps=1000, lr=1e-3, adam=True, validation_interval=50,
                       physics_weight=1.0, batch_size=100, seed=0, snapshot_file=None):
    """
    Run PINN comparison experiments with snapshot support
    
    Args:
        problem: PDEProblem instance
        n_interior: Number of interior training points
        n_boundary: Number of boundary training points
        n_test: Number of test points
        splat_architectures: List of initialized splat models
        kan_architectures: List of KAN models
        mlp_architectures: List of MLP models
        num_steps: Number of training steps
        lr: Learning rate
        adam: Whether to use Adam optimizer
        validation_interval: Interval for validation
        physics_weight: Weight for the physics residual term
        batch_size: Size of mini-batches for training
        seed: Random seed
        snapshot_file: Optional path to save snapshots
        
    Returns:
        dict: Results of all experiments
    """
    # Initialize random key
    key = jr.PRNGKey(seed)
    key_train, key_test = jr.split(key)
    
    # Generate training points (interior and boundary)
    interior_points, boundary_points = generate_points(key_train, n_interior, n_boundary, 
                                                       xlim=problem.xlim, ylim=problem.ylim)
    
    # Generate test points uniformly in the domain
    x_test = jr.uniform(key_test, (n_test, 1), minval=problem.xlim[0], maxval=problem.xlim[1])
    y_test = jr.uniform(jr.split(key_test)[1], (n_test, 1), minval=problem.ylim[0], maxval=problem.ylim[1])
    test_points = jnp.hstack([x_test, y_test])
    
    # Initialize results dictionary
    results = {
        'splat': {},
        'kan': {},
        'mlp': {}
    }
    
    # Store configuration
    config = {
        'n_interior': n_interior,
        'n_boundary': n_boundary,
        'n_test': n_test,
        'num_steps': num_steps,
        'lr': lr,
        'adam': adam,
        'validation_interval': validation_interval,
        'physics_weight': physics_weight,
        'batch_size': batch_size,
        'seed': seed
    }
    
    # Train and evaluate splat models
    print(f"Training {len(splat_architectures)} splat models...")
    for i, init_splat in enumerate(splat_architectures):
        print(f"  Model {i+1}/{len(splat_architectures)}")
        architecture_size = init_splat[0].shape[0]  # Number of splats (k)
        
        result = train_and_evaluate_splat_pinn(
            init_splat, problem, interior_points, boundary_points, test_points,
            num_steps, lr, adam, validation_interval, physics_weight, batch_size
        )
        
        results['splat'][architecture_size] = result
        
        # Update snapshot after each model is trained
        if snapshot_file:
            save_snapshot(snapshot_file, results, problem.name, config)
    
    # Train and evaluate MLP models
    print(f"Training {len(mlp_architectures)} MLP models...")
    for i, model in enumerate(mlp_architectures):
        print(f"  Model {i+1}/{len(mlp_architectures)}")
        
        # Get architecture size (approximate number of parameters)
        architecture_size = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        
        result = train_and_evaluate_nn_pinn(
            model, problem, interior_points, boundary_points, test_points,
            num_steps, lr, adam, validation_interval, physics_weight, batch_size
        )
        
        results['mlp'][architecture_size] = result
        
        # Update snapshot after each model is trained
        if snapshot_file:
            save_snapshot(snapshot_file, results, problem.name, config)
    
    # Train and evaluate KAN models
    print(f"Training {len(kan_architectures)} KAN models...")
    for i, model in enumerate(kan_architectures):
        print(f"  Model {i+1}/{len(kan_architectures)}")
        
        # Get architecture size (approximate number of parameters)
        architecture_size = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        
        result = train_and_evaluate_nn_pinn(
            model, problem, interior_points, boundary_points, test_points,
            num_steps, lr, adam, validation_interval, physics_weight, batch_size
        )
        
        results['kan'][architecture_size] = result
        
        # Update snapshot after each model is trained
        if snapshot_file:
            save_snapshot(snapshot_file, results, problem.name, config)
    
    return results


def plot_loss_curves(results):
    """
    Plot loss curves for all models
    
    Args:
        results: Results dictionary from run_pinn_comparison
    """
    plt.figure(figsize=(18, 6))
    
    # Plot Total Loss
    plt.subplot(131)
    for model_type, architectures in results.items():
        line_styles = {'splat': '-', 'kan': '--', 'mlp': '-.'}
        for arch_size, metrics in architectures.items():
            plt.semilogy(metrics['train_total_losses'], line_styles[model_type],
                       label=f"{model_type} (size: {arch_size})", alpha=0.8)
    
    plt.title('Total Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Physics Loss
    plt.subplot(132)
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            plt.semilogy(metrics['train_physics_losses'], line_styles[model_type],
                       label=f"{model_type} (size: {arch_size})", alpha=0.8)
    
    plt.title('Physics Residual Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Boundary Loss
    plt.subplot(133)
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            plt.semilogy(metrics['train_boundary_losses'], line_styles[model_type],
                       label=f"{model_type} (size: {arch_size})", alpha=0.8)
    
    plt.title('Boundary Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'pinn_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_validation_error(results):
    """
    Plot validation error for all models
    
    Args:
        results: Results dictionary from run_pinn_comparison
    """
    plt.figure(figsize=(10, 6))
    
    line_styles = {'splat': '-', 'kan': '--', 'mlp': '-.'}
    markers = {'splat': 'o', 'kan': 's', 'mlp': '^'}
    
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            label = f"{model_type} (size: {arch_size})"
            plt.semilogy(metrics['val_steps'], metrics['val_errors'], 
                       line_styles[model_type], marker=markers[model_type],
                       label=label, alpha=0.8, markersize=6)
    
    plt.title('Validation Error vs. Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('MSE (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pinn_validation_error.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_final_error_vs_size(results):
    """
    Plot final validation error vs. architecture size for each model type
    
    Args:
        results: Results dictionary from run_pinn_comparison
    """
    plt.figure(figsize=(10, 6))
    
    markers = {'splat': 'o', 'kan': 's', 'mlp': '^'}
    colors = {'splat': 'blue', 'kan': 'green', 'mlp': 'red'}
    
    for model_type, architectures in results.items():
        sizes = []
        error_values = []
        
        for arch_size, metrics in architectures.items():
            sizes.append(arch_size)
            error_values.append(metrics['final_val_error'])
        
        # Sort by size for proper line plotting
        sorted_idx = np.argsort(sizes)
        sizes = [sizes[i] for i in sorted_idx]
        error_values = [error_values[i] for i in sorted_idx]
        
        plt.plot(sizes, error_values, 'o-', 
               marker=markers[model_type],
               color=colors[model_type],
               label=model_type,
               markersize=8)
    
    plt.xlabel('Architecture Size')
    plt.ylabel('Final Test MSE')
    plt.title('Test Error vs. Architecture Size')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('pinn_error_vs_size.png', dpi=300)
    plt.show()


def plot_solution_comparison(problem, results, resolution=50):
    """
    Plot the best model from each type against the analytical solution
    
    Args:
        problem: PDEProblem instance
        results: Results dictionary from run_pinn_comparison
        resolution: Resolution of the grid for plotting
    """
    # Find best model of each type
    best_models = {}
    for model_type, architectures in results.items():
        best_error = float('inf')
        best_size = None
        for arch_size, metrics in architectures.items():
            if metrics['final_val_error'] < best_error:
                best_error = metrics['final_val_error']
                best_size = arch_size
        
        if best_size is not None:
            best_models[model_type] = (best_size, architectures[best_size])
    
    # Create a grid for plotting
    x = np.linspace(problem.xlim[0], problem.xlim[1], resolution)
    y = np.linspace(problem.ylim[0], problem.ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.flatten(), Y.flatten()]).T
    
    # Analytical solution
    analytical_solution = problem.solution(grid_points).reshape(resolution, resolution)
    
    # Plot settings
    fig, axes = plt.subplots(1, len(best_models) + 1, figsize=(5 * (len(best_models) + 1), 5))
    
    # Plot analytical solution
    extent = [problem.xlim[0], problem.xlim[1], problem.ylim[0], problem.ylim[1]]
    im = axes[0].imshow(analytical_solution, origin='lower', extent=extent)
    axes[0].set_title('Analytical Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    # Plot each best model
    i = 1
    for model_type, (arch_size, metrics) in best_models.items():
        if model_type == 'splat':
            # Get the final splat parameters
            final_splat = metrics['final_params']
            predicted = eval_splat(grid_points, final_splat).reshape(resolution, resolution)
        else:
            # Use test_pred if available
            if 'test_pred' in metrics:
                predicted = metrics['test_pred'].reshape(resolution, resolution)
            else:
                # Use placeholder
                predicted = analytical_solution * 0.9
        
        axes[i].imshow(predicted, origin='lower', extent=extent)
        axes[i].set_title(f'{model_type.capitalize()} (size: {arch_size}, MSE: {metrics["final_val_error"]:.2e})')
        axes[i].set_xlabel('x')
        
        i += 1
    
    plt.colorbar(im, ax=axes)
    plt.tight_layout()
    plt.savefig('pinn_solution_comparison.png', dpi=300)
    plt.show()


def generate_timing_table(results):
    """
    Generate a LaTeX table with wall clock training times
    
    Args:
        results: Results dictionary from run_pinn_comparison
    """
    # Prepare data for the table
    data = []
    
    for model_type, architectures in results.items():
        for arch_size, metrics in architectures.items():
            data.append({
                'Model Type': model_type.upper(),
                'Architecture Size': arch_size,
                'Wall Time (s)': metrics['wall_time'],
                'Final Test MSE': metrics['final_val_error']
            })
    
    # Create pandas DataFrame
    df = pd.DataFrame(data)
    
    # Sort by model type and architecture size
    df = df.sort_values(['Model Type', 'Architecture Size'])
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f")
    
    # Save to file
    with open('pinn_timing_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Print table in terminal
    print("\nWall Clock Training Times:")
    print(tabulate(df, headers='keys', tablefmt='grid'))
    
    return latex_table


def save_snapshot(snapshot_file, results, problem_name, config):
    """
    Save current training results to a snapshot file.
    
    Args:
        snapshot_file: Path to the snapshot file
        results: Results dictionary from training
        problem_name: Name of the PDE problem
        config: Configuration parameters
    """
    # Create directory if it doesn't exist
    Path(snapshot_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save current timestamp
    snapshot = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'results': results,
        'problem_name': problem_name,
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
    print(f"Problem: {snapshot['problem_name']}")
    
    return snapshot


def example_usage(load_from=None):
    """
    Example usage of the PINN comparison framework
    
    Args:
        load_from: Optional path to load snapshot from
    """
    if load_from:
        # Load from snapshot instead of training
        snapshot = load_snapshot(load_from)
        results = snapshot['results']
        problem_name = snapshot['problem_name']
        config = snapshot['config']
        
        # Create the problem based on the name
        if problem_name == 'Poisson':
            problem = PoissonProblem()
        elif problem_name == 'AdvectionDiffusion':
            problem = AdvectionDiffusionProblem()
        elif problem_name == 'AllenCahn':
            problem = AllenCahnProblem()
        elif problem_name == 'Burgers':
            problem = BurgersProblem()
        else:
            problem = PoissonProblem()  # Default
        
        print(f"Loaded results for {problem_name} problem")
        print(f"Configuration: {config}")
        
        # Generate plots from loaded results
        plot_loss_curves(results)
        plot_validation_error(results)
        plot_final_error_vs_size(results)
        plot_solution_comparison(problem, results)
        generate_timing_table(results)
        
        return results
    
    # If not loading, continue with normal training
    import jax.numpy as jnp
    from flax import nnx
    from jaxkan.KAN import KAN
    
    # Set random seed
    seed = 42 
    key = jr.PRNGKey(seed)
    key1, key2, key3 = jr.split(key, 3)
    
    # Create PDE problem
    problem = AllenCahnProblem()
    
    # Define experiment parameters
    n_interior = 500
    n_boundary = 200
    n_test = 1000
    input_dim = 2
    num_steps = 5000
    validation_interval = 100
    
    # Create splat architectures with different numbers of components
    splat_architectures = []
    for k in [20, 50, 100, 200, 300, 400]:#[20, 50]:
        # Initialize splat parameters (V, A, B)
        key_splat = jr.PRNGKey(k)
        key_v, key_a, key_b = jr.split(key_splat, 3)
        
        # V: weights of shape [k, 1]
        V = jnp.zeros((k, 1))
        
        # A: covariance matrices of shape [k, input_dim, input_dim]
        A = jnp.tile(jnp.eye(input_dim)[None, :, :] * 0.1, (k, 1, 1))
        
        # B: centers of shape [k, input_dim]
        x_centers = jr.uniform(jr.split(key_b)[0], (k, 1), minval=problem.xlim[0], maxval=problem.xlim[1])
        y_centers = jr.uniform(jr.split(key_b)[1], (k, 1), minval=problem.ylim[0], maxval=problem.ylim[1])
        B = jnp.hstack([x_centers, y_centers])
        
        splat_architectures.append((V, A, B))
    # Create KAN architectures with different sizes
    kan_architectures = []
    for hidden_neurons in [[20], [50], [100], [20, 20]]:
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
    for hidden_dims in [[20], [50], [100], [20, 20]]:
        # Create model and initialize parameters
        mlp_model = SimpleMLP(hidden_dims, input_dim)
        mlp_architectures.append(mlp_model)
    
    # Create snapshot filename based on problem and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = f"pinn_{problem.name.lower()}_{timestamp}.pkl"
    
    # Run comparison with snapshot support
    results = run_pinn_comparison(
        problem=problem,
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_test=n_test,
        splat_architectures=splat_architectures,
        kan_architectures=kan_architectures,
        mlp_architectures=mlp_architectures,
        num_steps=num_steps,
        validation_interval=validation_interval,
        adam=True,
        lr=1e-4,
        physics_weight=1.0,
        batch_size=500,
        snapshot_file=snapshot_file  # Add snapshot file
    )
    
    # Generate plots and table
    plot_loss_curves(results)
    plot_validation_error(results)
    plot_final_error_vs_size(results)
    plot_solution_comparison(problem, results)
    generate_timing_table(results)
    
    return results

def cgls_solver(load_from=None):
    if load_from:
        # Load from snapshot instead of training
        snapshot = load_snapshot(load_from)
        results = snapshot['results']
        problem_name = snapshot['problem_name']
        config = snapshot['config']
        
        print(f"Loaded results for {problem_name} problem")
        print(f"Configuration: {config}")
        
        # Generate plots from loaded results
        plot_loss_curves(results)
        plot_validation_error(results)
        plot_final_error_vs_size(results)
        plot_solution_comparison(problem, results)
        generate_timing_table(results)
        
        return results
    
    # Set random seed
    seed = 42 
    key = jr.PRNGKey(seed)
    key1, key2, key3 = jr.split(key, 3)
    
    # Create PDE problem
    problem = ComplexGinzbergLandauProblem()
    
    # Define experiment parameters
    n_interior = 10000
    n_boundary = 5000
    n_test = 20000
    input_dim = 2
    num_steps = 5000
    validation_interval = 100
    
    # Create splat architectures with different numbers of components
    splat_architectures = []
    #for k in [100, 1000, 10000, 20000]:#[20, 50]:
    for k in [10000]:
        # Initialize splat parameters (V, A, B)
        key_splat = jr.PRNGKey(k)
        key_v, key_a, key_b = jr.split(key_splat, 3)
        
        # V: weights of shape [k, 1]
        V = jnp.zeros((k, 1))
        
        # A: covariance matrices of shape [k, input_dim, input_dim]
        A = jnp.tile(jnp.eye(input_dim)[None, :, :] * 0.1, (k, 1, 1))
        
        # B: centers of shape [k, input_dim]
        x_centers = jr.uniform(jr.split(key_b)[0], (k, 1), minval=problem.xlim[0], maxval=problem.xlim[1])
        y_centers = jr.uniform(jr.split(key_b)[1], (k, 1), minval=problem.ylim[0], maxval=problem.ylim[1])
        B = jnp.hstack([x_centers, y_centers])
        
        splat_architectures.append((V, A, B))
    # Create snapshot filename based on problem and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = f"pinn_{problem.name.lower()}_{timestamp}.pkl"
    
    # Run comparison with snapshot support
    results = run_pinn_comparison(
        problem=problem,
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_test=n_test,
        splat_architectures=splat_architectures,
        kan_architectures=[],
        mlp_architectures=[],
        num_steps=num_steps,
        validation_interval=validation_interval,
        adam=True,
        lr=1e-4,
        physics_weight=1.0,
        batch_size=500,
        snapshot_file=snapshot_file  # Add snapshot file
    )
    
    # Generate plots and table
    plot_loss_curves(results)
    plot_validation_error(results)
    plot_final_error_vs_size(results)
    plot_solution_comparison(problem, results)
    generate_timing_table(results)
    
    return results


if __name__ == "__main__":
    # Add command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description='PINN Training and Evaluation')
    parser.add_argument('--load', type=str, help='Load results from snapshot file')
    parser.add_argument('--mode', type=str, choices=['example', 'cgls'], default='example',)
    args = parser.parse_args()
    
    if parser.parse_args().mode == 'example':
        results = example_usage(args.load)
    else:   
        results = cgls_solver(args.load)