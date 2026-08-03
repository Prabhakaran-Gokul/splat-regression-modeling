from v2.lib.splat import gd_splat_regression, splat_anim_1d, eval_splat
import jax.numpy as jnp
import jax.random as jr 
import matplotlib.pyplot as plt 
from utils import gridpts
import numpy as np
from scipy.linalg import lstsq
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

a = lambda x: jnp.array(x)
f = lambda x: jnp.flatten(x)

def cheb_poly(n, x):
    """Evaluate Chebyshev polynomial of degree n at points x"""
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return x
    else:
        return 2 * x * cheb_poly(n-1, x) - cheb_poly(n-2, x)

def cheb_poly_all(n, x):
    """
    Efficiently evaluate all Chebyshev polynomials up to degree n at points x
    
    Args:
        n: Maximum polynomial degree
        x: Points at which to evaluate polynomials
        
    Returns:
        Array of shape (n+1, len(x)) with T_i(x) values for i=0...n
    """
    # Pre-allocate result array
    T = jnp.zeros((n + 1, x.shape[0]))
    
    # Set T_0(x) = 1
    T = T.at[0].set(jnp.ones_like(x))
    
    if n >= 1:
        # Set T_1(x) = x
        T = T.at[1].set(x)
        
        # Use recurrence relation to compute higher-order polynomials
        for i in range(2, n + 1):
            T = T.at[i].set(2 * x * T[i-1] - T[i-2])
    
    return T

def cheb_ls_fit(train_X, train_Y, num_points):
    """
    Perform least squares fitting with Chebyshev polynomials
    
    Args:
        train_X: Training inputs, shape [n, 1]
        train_Y: Training outputs, shape [n, 1]
        num_points: Number of Chebyshev polynomials to use (polynomial degree + 1)
        
    Returns:
        mse: Mean squared error on training data
    """
    # Rescale X to [-1, 1] for Chebyshev polynomials
    x_min, x_max = jnp.min(train_X), jnp.max(train_X)
    x_scaled = 2 * (train_X - x_min) / (x_max - x_min) - 1
    
    # Build design matrix
    X_design = jnp.zeros((len(train_X), num_points))
    for i in range(num_points):
        X_design = X_design.at[:, i].set(cheb_poly(i, x_scaled)[:,0])
    
    # Solve least squares problem
    coeffs, residuals, rank, s = lstsq(X_design, train_Y)
    
    # Compute predicted values
    y_pred = X_design @ coeffs
    
    # Compute MSE
    mse = jnp.mean((y_pred - train_Y)**2)
    
    return float(mse)

def cheb_monte_carlo_fit(train_X, train_Y, num_points, num_samples=1000):
    """
    Perform Monte Carlo estimation of Chebyshev coefficients
    
    Args:
        train_X: Training inputs, shape [n, 1]
        train_Y: Training outputs, shape [n, 1]
        num_points: Number of Chebyshev polynomials to use (polynomial degree + 1)
        num_samples: Number of Monte Carlo samples to use
        
    Returns:
        mse: Mean squared error on training data
    """
    # Rescale X to [-1, 1] for Chebyshev polynomials
    x_min, x_max = jnp.min(train_X), jnp.max(train_X)
    x_scaled = 2 * (train_X - x_min) / (x_max - x_min) - 1
    
    # Generate Monte Carlo samples following the Chebyshev distribution
    # by sampling theta uniformly and setting x = cos(theta)
    key = jr.PRNGKey(0)
    theta = jr.uniform(key, (num_samples,), minval=0, maxval=jnp.pi)
    x_mc = jnp.cos(theta)
    
    # Scale back to original domain
    x_mc_orig = (x_mc + 1) / 2 * (x_max - x_min) + x_min
    
    # For each Monte Carlo point, find the nearest training point
    # We need to compute pairwise distances
    distances = jnp.abs(x_mc_orig.reshape(-1, 1) - train_X.reshape(1, -1))
    nearest_indices = jnp.argmin(distances, axis=1)
    y_mc = train_Y[nearest_indices, 0]
    
    # Initialize coefficients
    coeffs = jnp.zeros(num_points)
    
    # Estimate coefficients using Monte Carlo
    for n in range(num_points):
        if n == 0:
            coeffs = coeffs.at[n].set(jnp.mean(y_mc))
        else:
            coeffs = coeffs.at[n].set(2 * jnp.mean(y_mc * jnp.cos(n * theta)))
    
    # Evaluate the Chebyshev approximation at training points
    y_pred = jnp.zeros_like(train_Y[:, 0])
    for n in range(num_points):
        y_pred = y_pred + coeffs[n] * cheb_poly(n, x_scaled[:, 0])
    
    # Compute MSE
    mse = jnp.mean((y_pred - train_Y[:, 0])**2)
    
    return float(mse)

def cheb_monte_carlo_fit(y_func, num_points, num_samples=10000, eval_points=None):
    """
    Perform Monte Carlo estimation of Chebyshev coefficients directly using a function
    
    Args:
        y_func: Function to approximate, defined on [0, 1]
        num_points: Number of Chebyshev polynomials to use (polynomial degree + 1)
        num_samples: Number of Monte Carlo samples to use
        eval_points: Points to evaluate MSE (if None, generates 1000 uniform points)
        
    Returns:
        mse: Mean squared error on evaluation points
    """
    # Generate Monte Carlo samples following the Chebyshev distribution
    key = jr.PRNGKey(0)
    theta = jr.uniform(key, (num_samples,), minval=0, maxval=jnp.pi)
    x_mc = jnp.cos(theta)
    
    # Transform from [-1, 1] to [0, 1] for our domain
    x_mc_scaled = (x_mc + 1) / 2
    
    # Evaluate the function at Monte Carlo points
    y_mc = y_func(x_mc_scaled.reshape(-1, 1)).flatten()
    
    # Compute cos(n*theta) for all n at once for faster coefficient calculation
    cos_n_theta = jnp.vstack([jnp.cos(n * theta) for n in range(num_points)])
    
    # Efficiently compute all coefficients at once
    coeffs = jnp.zeros(num_points)
    # First coefficient (n=0)
    coeffs = coeffs.at[0].set(jnp.mean(y_mc))
    # Remaining coefficients (n>0) 
    coeffs = coeffs.at[1:].set(2 * jnp.mean(y_mc * cos_n_theta[1:], axis=1))
    
    # If no evaluation points provided, generate uniform grid
    if eval_points is None:
        eval_key = jr.PRNGKey(1)
        eval_points = jr.uniform(eval_key, (1000, 1), minval=0, maxval=1)
    elif len(eval_points.shape) == 1:
        eval_points = eval_points.reshape(-1, 1)
    
    # Transform evaluation points to [-1, 1] for Chebyshev polynomials
    x_eval_scaled = 2 * eval_points - 1
    
    # Compute all Chebyshev polynomials at evaluation points at once
    T = cheb_poly_all(num_points - 1, x_eval_scaled.flatten())
    
    # Efficiently compute approximation using matrix multiplication
    y_pred = jnp.dot(coeffs, T)
    
    # Compute MSE against true function values
    y_true = y_func(eval_points).flatten()
    mse = jnp.mean((y_pred - y_true)**2)
    
    return float(mse)

def haar_wavelet(j, k, x):
    """
    Evaluate the Haar wavelet at points x
    j: scale parameter (j >= 0)
    k: translation parameter (0 <= k < 2^j)
    x: points to evaluate (in [0, 1])
    """
    if j == -1:  # Scaling function (constant)
        return jnp.ones_like(x)
    
    # Compute support interval
    a = k / (2**j)
    b = (k + 0.5) / (2**j)
    c = (k + 1) / (2**j)
    
    # Piecewise definition
    result = jnp.zeros_like(x)
    result = jnp.where((x >= a) & (x < b), 2**(j/2), result)
    result = jnp.where((x >= b) & (x < c), -2**(j/2), result)
    
    return result

def haar_monte_carlo_fit(y_func, max_level, num_samples=10000, eval_points=None):
    """
    Compute Haar wavelet approximations at multiple levels using Monte Carlo
    
    Args:
        y_func: Function to approximate, defined on [0, 1]
        max_level: Maximum wavelet level L
        num_samples: Number of Monte Carlo samples to use
        eval_points: Points to evaluate MSE (if None, generates 1000 uniform points)
    
    Returns:
        List of (mse, basis_count) tuples for each level
    """
    # Generate random samples uniformly in [0, 1] for coefficient estimation
    key = jr.PRNGKey(42)
    x_samples = jr.uniform(key, (num_samples, 1), minval=0, maxval=1)
    y_samples = y_func(x_samples)
    
    # If no evaluation points provided, generate uniform grid
    if eval_points is None:
        eval_key = jr.PRNGKey(43)
        eval_points = jr.uniform(eval_key, (1000, 1), minval=0, maxval=1)
    
    # True function values at evaluation points
    y_true = y_func(eval_points)
    
    results = []
    
    # For each level
    for level in range(max_level + 1):
        # Number of basis functions for this level: scaling function + all wavelets
        basis_count = 2**(level+1) - 1
        
        # Initialize approximation at eval points to zero
        y_approx = jnp.zeros_like(eval_points[:, 0])
        
        # Start with scaling function (mean)
        scaling_coef = jnp.mean(y_samples)
        y_approx += scaling_coef
        
        # Add contributions from each wavelet level
        for j in range(level):
            for k in range(2**j):
                # Evaluate wavelet on MC samples
                psi_samples = haar_wavelet(j, k, x_samples)
                
                # Estimate coefficient using Monte Carlo
                coef = jnp.mean(y_samples[:, 0] * psi_samples[:, 0])
                
                # Add contribution to approximation
                psi_eval = haar_wavelet(j, k, eval_points)
                y_approx += coef * psi_eval[:, 0]
        
        # Compute MSE
        mse = jnp.mean((y_approx - y_true[:, 0])**2)
        results.append((float(mse), basis_count))
    
    return results

# More efficient implementation using vectorization
def haar_monte_carlo_fit_efficient(y_func, max_level, num_samples=10000, eval_points=None):
    """
    Compute Haar wavelet approximations at multiple levels using Monte Carlo with vectorization
    """
    # Generate random samples uniformly in [0, 1] for coefficient estimation
    key = jr.PRNGKey(42)
    x_samples = jr.uniform(key, (num_samples, 1), minval=0, maxval=1)
    y_samples = y_func(x_samples).flatten()
    
    # If no evaluation points provided, generate uniform grid
    if eval_points is None:
        eval_key = jr.PRNGKey(43)
        eval_points = jr.uniform(eval_key, (1000, 1), minval=0, maxval=1)
    
    # True function values at evaluation points
    y_true = y_func(eval_points).flatten()
    
    # Compute scaling function coefficient (mean of function)
    scaling_coef = jnp.mean(y_samples)
    
    results = []
    
    # Start with just the scaling function
    y_approx_base = jnp.ones_like(eval_points[:, 0]) * scaling_coef
    
    # Track accumulated approximation
    y_approx = y_approx_base.copy()
    
    # For each level
    for level in range(max_level + 1):
        if level > 0:
            # Add wavelet contributions for the current level j=level-1
            j = level - 1
            for k in range(2**j):
                # Evaluate wavelet on samples
                psi_samples = haar_wavelet(j, k, x_samples)
                
                # Estimate coefficient using Monte Carlo
                coef = jnp.mean(y_samples * psi_samples[:, 0])
                
                # Add contribution to approximation
                psi_eval = haar_wavelet(j, k, eval_points)
                y_approx += coef * psi_eval[:, 0]
        
        # Compute MSE for current level
        basis_count = 2**(level+1) - 1
        mse = jnp.mean((y_approx - y_true)**2)
        results.append((float(mse), basis_count))
    
    return results

def splat_anim_1d(splatnns, f, train_X, train_Y, xlim=[-1,1], Ngrid=50, interval=100, fskip=1, db=False, baselines=[]): 
    '''
    splatnns: list of tuples of (V,A,B) where V is a [k,1] real tensor, A is a [k,1,1] real tensor, B is a [k,1,1] real tensor
    f: a real function with real input. f(X) automatically broadcasts elementwise.
    train_X, train_Y: training data to be plotted.
    xlim: (optional) domain for plotting
    baselines: (optional) list of (value, label) tuples to plot as horizontal lines on MSE plot
    '''
    splatnns = splatnns[::fskip]
    V_stack = jnp.stack([splatnn[0] for splatnn in splatnns], axis=0) # [N,k,1]
    A_stack = jnp.stack([splatnn[1] for splatnn in splatnns], axis=0)
    B_stack = jnp.stack([splatnn[2] for splatnn in splatnns], axis=0) # [N,k,1]
    assert V_stack.ndim == 3 and A_stack.ndim == 4 and B_stack.ndim == 3
    assert V_stack.shape[0] == A_stack.shape[0] == B_stack.shape[0]
    assert V_stack.shape[1] == A_stack.shape[1] == B_stack.shape[1]
    
    n_steps,k,d = V_stack.shape 
    # Ensure inputs are JAX arrays
    V = jnp.asarray(V_stack).reshape((n_steps,k))
    A = jnp.asarray(A_stack).reshape((n_steps,k))
    B = jnp.asarray(B_stack).reshape((n_steps,k))
    N = n_steps

    x = jnp.linspace(xlim[0], xlim[1], Ngrid)
    f_x = f(x)

    # Precompute MSE for all frames
    mse_vals = []
    for i in range(N):
        y_pred = eval_splat(train_X, (splatnns[i][0], splatnns[i][1], splatnns[i][2]))
        mse = jnp.mean((y_pred - train_Y)**2)
        if db:
            mse_vals.append(jnp.log10(mse))
        else:
            mse_vals.append(mse)
        

    mse_vals = jnp.array(mse_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Setup for plot 1 (function fit)
    line_splat, = ax1.plot([], [], lw=2, label='Splat')
    line_f, = ax1.plot([], [], lw=2, label='f(x)')
    ax1.plot(train_X, train_Y, 'x', color='orange', label='Training Data')
    
    # Artists for splat components
    v_lines = ax1.vlines([], [], [], color="k", linestyle="dashed", lw=0.5)
    tick_size = 0.05 * (np.max(np.asarray(f_x)) - np.min(np.asarray(f_x)))
    a_ticks, = ax1.plot([], [], '|', color='k', lw=0.5)

    ax1.set_xlim(xlim)
    ax1.set_ylim(np.min(np.asarray(f_x))-0.5, np.max(np.asarray(f_x))+0.5)
    ax1.legend()
    ax1.set_title('Splat Regression')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Setup for plot 2 (MSE)
    line_mse, = ax2.plot([], [], lw=2, label='MSE', color="#594B48")
    ax2.set_xlim(0, N)
    if db:
        min_val = jnp.min(mse_vals) if len(mse_vals) > 0 else -0.5
        min_baseline = jnp.min(jnp.log10(jnp.array([baseline[0] for baseline in baselines]))) if len(baselines) > 0 else min_val
        ax2.set_ylim(1.1*jnp.minimum(min_val, min_baseline), 1)
    else: 
        ax2.set_ylim(0, jnp.max(mse_vals) * 1.1 if len(mse_vals) > 0 else 1)
    
    # Add baseline horizontal lines with type-based coloring and inline labels
    baseline_lines = []
    for value, label in baselines:
        if db:
            value = jnp.log10(value)
            
        # Set color based on baseline type
        if 'cheb' in label:
            color = '#6A8AD9'
        elif 'wav' in label:
            color = '#DB694F'
        else:
            # Use default colors for other baselines
            color = None
            
        # Add horizontal line (without label in legend)
        line = ax2.axhline(y=value, linestyle='--', alpha=0.8, color=color, lw=0.5)
        baseline_lines.append(line)
        
        # Add text label at the left side of the plot with small margin
        # Position text at x=0 (left edge) with small offset
        margin = N * 0.02  # Small margin (2% of x-axis width)
        ax2.text(margin, value, label, va='bottom', ha='left', fontsize=8, 
                 color=color, alpha=0.9, backgroundcolor='white', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Only show MSE in the legend now
    ax2.legend([line_mse], ['MSE'])
    
    ax2.set_title('log(MSE)' if db else 'Mean Squared Error')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('log(MSE)' if db else 'MSE')

    def splat_func(i):
        # x shape: (Ngrid,)
        # V[i], A[i], B[i] shapes: (k,)
        # Reshape for broadcasting:
        # x -> (Ngrid, 1)
        # V[i], A[i], B[i] -> (1, k)
        x_reshaped = x[:, jnp.newaxis]
        
        # p will have shape (Ngrid, k) due to broadcasting
        p = norm.pdf(x_reshaped, loc=B[i, :], scale=A[i, :])
        
        # V[i,:] is broadcast to (Ngrid, k), element-wise multiply, then sum over k (axis=1)
        splat = jnp.sum(V[i, :] * p, axis=1)
        return splat

    def init():
        line_splat.set_data([], [])
        line_f.set_data([], [])
        line_mse.set_data([], [])
        v_lines.set_segments([])
        a_ticks.set_data([], [])
        return line_splat, line_f, line_mse, v_lines, a_ticks

    def animate(i):
        ax1.set_title(f"Step {i+1}/{N}")
        y_splat = splat_func(i)
        # Convert JAX arrays to numpy for matplotlib
        line_splat.set_data(np.asarray(x), np.asarray(y_splat))
        line_f.set_data(np.asarray(x), np.asarray(f_x))
        
        # Update splat component visualizations
        b_i, a_i, v_i = B[i], A[i], V[i]
        
        # Update vertical lines for V
        segments = [[[b, 0], [b, v]] for b, v in zip(b_i, v_i)]
        v_lines.set_segments(segments)
        
        # Update ticks for A
        tick_xs = np.concatenate([(b_i - a_i), (b_i + a_i)])
        tick_ys = np.zeros_like(tick_xs)
        a_ticks.set_data(tick_xs, tick_ys)

        # Update MSE plot
        iterations = jnp.arange(i + 1)
        line_mse.set_data(np.asarray(iterations), np.asarray(mse_vals[:i+1]))
        
        return line_splat, line_f, line_mse, v_lines, a_ticks

    anim = FuncAnimation(fig, animate, init_func=init, frames=list(range(N))[::fskip], interval=interval)
    plt.tight_layout()
    return anim


# Add this function after the main code to create a static multi-frame summary
def create_summary_plot(splatnns, f, train_X, train_Y, mse_vals, baselines, xlim=[-1,1], Ngrid=400, fskip=10, db=True):
    """
    Create a static summary plot with frames at different iterations and the MSE curve
    
    Args:
        splatnns: List of splat parameters at each iteration
        f: Target function
        train_X, train_Y: Training data
        mse_vals: MSE values for each iteration
        baselines: Baseline values and labels
        xlim: Domain for plotting
        Ngrid: Number of grid points
        fskip: Frame skip factor
        db: Whether to use log scale for MSE
    """
    # Increase default font sizes
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })
    
    # Select frames at specific iterations
    frame_indices = jnp.arange(10) * (len(splatnns) // 10)
    frame_indices = [min(i, len(splatnns)-1) for i in frame_indices]  # Ensure valid indices
    
    # Create figure with specified width ratio
    fig = plt.figure(figsize=(22, 10))  # Slightly larger figure
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    
    # Create left subplot area (for frames grid)
    frames_area = fig.add_subplot(gs[0])
    frames_area.axis('off')  # Hide the main axes
    
    # Create grid for frames within the left area
    frames_grid = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0], 
                                                wspace=0.2, hspace=0.4)  # More space between subplots
    
    # Setup data for plotting
    V_stack = jnp.stack([splatnns[i][0] for i in range(len(splatnns))], axis=0)
    A_stack = jnp.stack([splatnns[i][1] for i in range(len(splatnns))], axis=0)
    B_stack = jnp.stack([splatnns[i][2] for i in range(len(splatnns))], axis=0)
    
    n_steps, k, d = V_stack.shape
    V = jnp.asarray(V_stack).reshape((n_steps, k))
    A = jnp.asarray(A_stack).reshape((n_steps, k))
    B = jnp.asarray(B_stack).reshape((n_steps, k))
    
    x = jnp.linspace(xlim[0], xlim[1], Ngrid)
    f_x = f(x)
    
    # Function to compute splat values
    def splat_func(i):
        x_reshaped = x[:, jnp.newaxis]
        p = norm.pdf(x_reshaped, loc=B[i, :], scale=A[i, :])
        splat = jnp.sum(V[i, :] * p, axis=1)
        return splat
    
    # Plot frames
    for idx, frame_idx in enumerate(frame_indices):
        frame_idx = min(frame_idx, n_steps-1)  # Ensure valid index
        
        # Compute subplot position (row, col)
        row = idx // 5
        col = idx % 5
        
        # Create subplot
        ax = fig.add_subplot(frames_grid[row, col])
        
        # Plot function and splat
        y_splat = splat_func(frame_idx)
        ax.plot(np.asarray(x), np.asarray(f_x), 'g-', lw=1.5, alpha=0.7, label='f(x)')
        ax.plot(np.asarray(x), np.asarray(y_splat), 'b-', lw=1.5, label='Splat')
        
        # Plot training data (fewer points for clarity)
        if len(train_X) > 1000:
            # Sample subset of points for cleaner visualization
            idx_subset = np.random.choice(len(train_X), 50, replace=False)
            ax.plot(train_X[idx_subset], train_Y[idx_subset], 'x', color='orange', 
                   markersize=3, alpha=0.7)
        else:
            ax.plot(train_X, train_Y, 'x', color='orange', markersize=3, alpha=0.7)
        
        # Plot splat components
        b_i, a_i, v_i = B[frame_idx], A[frame_idx], V[frame_idx]
        segments = [[[b, 0], [b, v]] for b, v in zip(b_i, v_i)]
        ax.vlines(x=np.array([s[0][0] for s in segments]), 
                 ymin=np.array([s[0][1] for s in segments]), 
                 ymax=np.array([s[1][1] for s in segments]), 
                 color="k", linestyle="dashed", lw=0.8)
        
        # Set limits and title
        ax.set_xlim(xlim)
        ax.set_ylim(np.min(np.asarray(f_x))-0.5, np.max(np.asarray(f_x))+0.5)
        ax.set_title(f"Step {frame_idx*fskip}", fontsize=14)
        
        # Only add x and y labels for edge subplots
        if row == 1:
            ax.set_xlabel('x', fontsize=13)
        if col == 0:
            ax.set_ylabel('y', fontsize=13)
            
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=12)
        
        # Use fewer ticks for cleaner appearance
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Create MSE plot on the right
    ax_mse = fig.add_subplot(gs[1])
    
    # Plot MSE curve
    iterations = np.arange(len(mse_vals))
    ax_mse.plot(iterations, np.asarray(mse_vals), lw=2.5, color="#594B48", label='MSE')
    
    # Add baselines with inline labels
    for value, label in baselines:
        if db:
            value = jnp.log10(value)
            
        # Set color based on baseline type
        if 'cheb' in label:
            color = '#6A8AD9'  # Blue for Chebyshev
        elif 'wav' in label:
            color = '#DB694F'  # Red for wavelets
        else:
            color = 'gray'
            
        # Add horizontal line
        ax_mse.axhline(y=value, linestyle='--', alpha=0.8, color=color, lw=1.2)
        
        # Add text label at the right side of plot for better visibility in static plot
        ax_mse.text(len(mse_vals) * 0.02, value, label, va='bottom', ha='left', 
                   fontsize=16, color=color, alpha=0.9, backgroundcolor='white',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Set MSE plot settings
    ax_mse.set_xlabel('Iteration', fontsize=16)
    ax_mse.set_ylabel('log(MSE)' if db else 'MSE', fontsize=16)
    ax_mse.set_title('Training Progress', fontsize=16)
    ax_mse.tick_params(axis='both', which='major', labelsize=12)
    ax_mse.legend(loc='upper right', fontsize=13)
    
    # Add overall super title
    fig.suptitle('Splat Function Training Progress', fontsize=18, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Make room for suptitle
    return fig

if __name__ == "__main__": 
    key = jr.PRNGKey(3621)
    k = 30
    
    # Initialize splat parameters
    init_B = (0.5 + 0.5*gridpts(k)).reshape((k, 1))
    #init_B = jnp.arange(k).reshape((k,1))/k #jr.uniform(key, (k,1), minval=0, maxval=1)
    neighborhoods = a([
        jnp.minimum(jnp.abs(left - center), jnp.abs(right - center))
        for right, center, left 
        in zip([2] + list(init_B.flatten())[:-1], list(init_B.flatten()), list(init_B.flatten())[1:] + [-1])
    ])
    init_A = jnp.ones((k,1,1)) * neighborhoods.reshape((k,1,1)) * 0.5
    init_V = jnp.zeros((k,1))
    init_splat = (init_V, init_A, init_B)

    # Generate training data
    train_X = jr.uniform(key, (200, 1), minval=0, maxval=1)
    
    y_func = lambda x: jnp.sin(20 * jnp.pi * x * (2-x))
    
    train_Y = y_func(train_X)
    
    # Compute Chebyshev fit MSEs
    print('Chebyshev Monte Carlo fit MSEs:')
    cheb_k_mse = cheb_monte_carlo_fit(y_func, k)
    cheb_kk_mse = cheb_monte_carlo_fit(y_func, (3*k)//2)

    print(f"Chebyshev MC fit log10(MSE) (k={k}): {jnp.log10(cheb_k_mse):.6f}")
    print(f"Chebyshev MC fit log10(MSE) (k={3*k//2}): {jnp.log10(cheb_kk_mse):.6f}")
    
    # Compute Haar wavelet approximation MSEs
    # Maximum wavelet level L = ceil(log2(3*k))
    import math
    L = math.ceil(math.log2(3*k))
    print(f"\nComputing Haar wavelet approximations up to level {L}...")
    
    wavelet_results = haar_monte_carlo_fit_efficient(y_func, L)
    
    print("Haar wavelet approximation MSEs:")
    for level, (mse, basis_count) in enumerate(wavelet_results):
        print(f"Level {level} (m={basis_count}): log10(MSE) = {jnp.log10(mse):.6f}")

    # Define baselines for the plot
    baselines = [
        (cheb_k_mse, f"cheb (m={k})"),
        (cheb_kk_mse, f"cheb (m={3*k//2})")
    ]
    
    # Add wavelet baselines
    for level, (mse, basis_count) in enumerate(wavelet_results):
        if basis_count > 1:  # Skip the very basic level if desired
            baselines.append((mse, f"wav (m={basis_count})"))
    
    # Train the splat model
    splat_regression_trajectory = gd_splat_regression(init_splat, train_X, train_Y, 
                                                    train_mask=(1,1,1), 
                                                    num_steps=20000, 
                                                    lr=0.00001, 
                                                    verbose=True,
                                                    selective_noise=None)
    
    # Create animation with baselines
    R1 = splat_anim_1d(splat_regression_trajectory, y_func, train_X, train_Y, 
                      xlim=[-0.2, 1.2], Ngrid=400, interval=20, fskip=10, 
                      db=True, baselines=baselines)
    
    R1.save('fit_sin_animation.mp4', writer='ffmpeg', fps=30)
    plt.show()
    
    # After creating and saving animation, create the summary plot
    from matplotlib import gridspec
    
    # Generate static summary plot
    summary_fig = create_summary_plot(splat_regression_trajectory, y_func, train_X, train_Y, 
                                    mse_vals=jnp.log10(jnp.array([jnp.mean((eval_splat(train_X, splat) - train_Y)**2) 
                                                               for splat in splat_regression_trajectory])), 
                                    baselines=baselines, xlim=[-0.2, 1.2], Ngrid=400, 
                                    fskip=10, db=True)
    
    # Save the summary plot
    summary_fig.savefig('splat_training_summary.pdf', format='pdf', bbox_inches='tight')