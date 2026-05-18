from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jax.scipy.stats import norm
from jax.scipy.linalg import solve
from tqdm import trange 
import optax
from flax import linen as nn
from jaxkan.models.KAN import KAN
from tqdm import tqdm

@partial(jax.jit, static_argnames=('rho',))
def eval_splat(X, splatnn, rho=None, eps=1e-6):
    '''
    X: [n,d] real tensor
    splatnn: tuple of (V,A,B) where V is a [k,p] real tensor, A is a [k,d,d] real tensor, B is a [k,d] real tensor
    rho: a real valued function with real tensor [n,d] input

    splatnn is a splat neural network parameters and rho is a mother splat function.  
    Returns: a [n,p] real tensor Y where Y[i,:] = sum_{j=1...k} V[j,:] * rho_{A[j,:,:], B[j,:]}(X[i,:])
    where rho_{A,B}(x) = det(A)^(-1) * rho(A^(-1) * (x-B))
    '''
    V, A, B = splatnn
    assert X.ndim == 2, f"X must be a [n,d] tensor, but has shape {X.shape}"
    assert V.ndim == 2, f"V must be a [k,p] tensor, but has shape {V.shape}"
    assert A.ndim == 3, f"A must be a [k,d,d] tensor, but has shape {A.shape}"
    assert B.ndim == 2, f"B must be a [k,d] tensor, but has shape {B.shape}"
    n, d = X.shape
    k, p = V.shape
    assert A.shape[0] == k, f"A's first dimension must be k={k}, but is {A.shape[0]}"
    assert A.shape[2] == d, f"A's third dimension must be d={d}, but is {A.shape[2]}"
    assert B.shape[0] == k, f"B's first dimension must be k={k}, but is {B.shape[0]}"
    assert B.shape[1] == d, f"B's second dimension must be d={d}, but is {B.shape[1]}"

    
    if rho is None:
        def gaussian_rho(x):
            # x has shape [..., d]
            # Standard multivariate normal density
            norm_sq = jnp.sum(x**2, axis=-1)
            return jnp.exp(-0.5 * norm_sq) / jnp.power(2 * jnp.pi, d / 2.0)
        rho = gaussian_rho

    # Reshape for broadcasting
    # X: [n, d] -> [n, 1, d]
    # B: [k, d] -> [1, k, d]
    X_reshaped = X[:, jnp.newaxis, :]
    B_reshaped = B[jnp.newaxis, :, :]

    # X_minus_B will have shape [n, k, d]
    X_minus_B = X_reshaped - B_reshaped

    # We want to compute A_inv @ (X - B) for each n and k.
    # A has shape [k, d, d]. X_minus_B has shape [n, k, d].
    # We need to solve k systems of linear equations for each of the n data points.
    # jax.linalg.solve(a, b) can broadcast over leading dimensions.
    # Let's make 'a' be A of shape [k, d, d] and 'b' be X_minus_B of shape [n, k, d].
    # To use solve, we need b to be of shape [..., M, K]
    # Our A is [k, d, d]. Our X_minus_B is [n, k, d]. We want output [n, k, d].
    # Let's make b have shape [n, k, d, 1].
    X_minus_B_for_solve = X_minus_B[..., jnp.newaxis] # Shape: [n, k, d, 1]
    # A is [k, d, d]. It will be broadcast to [n, k, d, d]
    A_inv_X_minus_B_solved = solve(A, X_minus_B_for_solve) # Shape: [n, k, d, 1]
    A_inv_X_minus_B = jnp.squeeze(A_inv_X_minus_B_solved, axis=-1) # Shape: [n, k, d]

    # rho_input is A_inv_X_minus_B, shape [n, k, d]
    # rho_output will have shape [n, k]
    rho_vals = rho(A_inv_X_minus_B)

    # det_A will have shape [k]
    det_A = jnp.linalg.det(A)

    # We need to handle the case where det(A) is close to zero.
    # The docstring says det(A)^(-1), let's use it.
    # Using jnp.where to avoid division by zero or very small numbers.
    inv_det_A = jnp.where(jnp.abs(det_A) > eps, 1.0 / det_A, 0.0)

    # inv_det_A has shape [k]. rho_vals has shape [n, k].
    # Broadcasting will make this [n, k]
    rho_transformed = inv_det_A * rho_vals

    # V has shape [k, p]. rho_transformed has shape [n, k].
    # We want Y of shape [n, p].
    # Y[i,:] = sum_{j=1...k} V[j,:] * rho_transformed[i,j]
    # This is a matrix multiplication: rho_transformed @ V
    Y = jnp.dot(rho_transformed, V)

    return Y

@partial(jax.jit, static_argnums=(3,))
def eval_splat_grad(splatnn, X, Y, variation, rho=None, eps=1e-9, sgd=False):
    '''
    

    rho: a real valued function with real tensor [n,d] input
    eps: tikhonov regularization parameter for matrix inversion
    sgd: if True, randomly subsample data points to compute stochastic gradient estimate

    splatnn is a splat neural network parameters and rho is a mother splat function.  
    Returns: a [n,p] real tensor Y where Y[i,:] = sum_{j=1...k} V[j,:] * rho_{A[j,:,:], B[j,:]}(X[i,:])
    where rho_{A,B}(x) = det(A)^(-1) * rho(A^(-1) * (x-B))
    '''
    V, A, B = splatnn
    assert X.ndim == 2, f"X must be a [n,d] tensor, but has shape {X.shape}"
    assert V.ndim == 2, f"V must be a [k,p] tensor, but has shape {V.shape}"
    assert A.ndim == 3, f"A must be a [k,d,d] tensor, but has shape {A.shape}"
    assert B.ndim == 2, f"B must be a [k,d] tensor, but has shape {B.shape}"
    n, d = X.shape
    k, p = V.shape
    assert A.shape[0] == k, f"A's first dimension must be k={k}, but is {A.shape[0]}"
    assert A.shape[1] == d, f"A's second dimension must be d={d}, but is {A.shape[1]}"
    assert A.shape[2] == d, f"A's third dimension must be d={d}, but is {A.shape[2]}"
    assert B.shape[0] == k, f"B's first dimension must be k={k}, but is {B.shape[0]}"
    assert B.shape[1] == d, f"B's second dimension must be d={d}, but is {B.shape[1]}"
    
    if rho is None:
        def gaussian_rho(x):
            # x has shape [..., d]
            # Standard multivariate normal density
            norm_sq = jnp.sum(x**2, axis=-1)
            return jnp.exp(-0.5 * norm_sq) / jnp.power(2 * jnp.pi, d / 2.0)
        rho = gaussian_rho

    if sgd == True:
        raise NotImplementedError("SGD not implemented yet")

    # Reshape for broadcasting
    # X: [n, d] -> [n, 1, d]
    # B: [k, d] -> [1, k, d]
    X_reshaped = X[:, jnp.newaxis, :]
    B_reshaped = B[jnp.newaxis, :, :]

    # X_minus_B will have shape [n, k, d]
    X_minus_B = X_reshaped - B_reshaped
    A_tikhonov = A + jnp.stack(k*[eps * np.eye(d)], axis=0)

    # Our A is [k, d, d]. Our X_minus_B is [n, k, d]. We want output [n, k, d].
    X_minus_B_for_solve = X_minus_B[..., jnp.newaxis] # Shape: [n, k, d, 1]
    # A is [k, d, d]. It will be broadcast to [n, k, d, d]
    A_inv_X_minus_B_solved = solve(A_tikhonov, X_minus_B_for_solve) # Shape: [n, k, d, 1]
    A_inv_X_minus_B = jnp.squeeze(A_inv_X_minus_B_solved, axis=-1) # Shape: [n, k, d]

    rho_vals = rho(A_inv_X_minus_B)
    inv_det_A = 1.0/jnp.linalg.det(A_tikhonov)
    rho_transformed = inv_det_A * rho_vals

    active_pts = (rho_transformed > 1e-9)[:,:,jnp.newaxis]
    rho_transformed = rho_transformed * active_pts[:,:,0] # delete it

    dFx = variation(X, Y) # [n,p]
    vdFx = dFx @ V.T # [n,k]
    
    # calculate grad log s(x), whose shape is [n,k,d]
    #grad_log_rho = jax.vmap(jax.grad(lambda x: rho(x)))(A_inv_X_minus_B.reshape((-1, d))).reshape((n,k,d)) 
    grad_log_rho_fn = jax.grad(lambda x: jnp.log(rho(x)))
    grad_log_rho =  jax.vmap(jax.vmap(grad_log_rho_fn))(A_inv_X_minus_B)
    grad_log_s = solve(A_tikhonov.transpose((0,2,1)), grad_log_rho[..., jnp.newaxis])[:,:,:,0]
    
    grad_log_s = jnp.nan_to_num(grad_log_s)*active_pts

    Ainv = jnp.linalg.inv(A_tikhonov)

    grad_V = rho_transformed.T @ dFx
    # still not quite right! it has a negative bias
    grad_A = -jnp.einsum('jkl,ij,ij->jkl', Ainv.transpose((0, 2, 1)), vdFx, rho_transformed) + \
        -jnp.einsum('ij,ijk,ijl,ij->jkl', rho_transformed, grad_log_s, A_inv_X_minus_B, vdFx)
    grad_B = -jnp.einsum('ij,ij,ijk->jk', rho_transformed, vdFx, grad_log_s)
 
    return grad_V, grad_A, grad_B

def splat_anim_1d(splatnns, f, train_X, train_Y, xlim=[-1,1], Ngrid=50, interval=100, fskip=1, db=False): 
    '''
    splatnns: list of tuples of (V,A,B) where V is a [k,1] real tensor, A is a [k,1,1] real tensor, B is a [k,1,1] real tensor
    f: a real function with real input. f(X) automatically broadcasts elementwise.
    train_X, train_Y: training data to be plotted.
    xlim: (optional) domain for plotting

    Call inside a jupyter notebook to generate an animation of training a k-splat regression model with N training steps. Frame i=1...N of the animation corresponds is a plot of the values of the splat function with parameters V[i], A[i], B[i] evalued over the grid with Ngrid points and xlim boundaries. The input f(.) is also plotted at the same parameters. 

    The value of the splat function with parameters V[i], A[i], B[i] is equal to the formula 
    splat(x) = sum_{j=1...k} V[ij] * p_{A[ij], B[ij]}(x)

    where p_{A[ij], B[ij]}(x) is the density of a 1D Gaussian with mean B[ij] and standard deviation A[ij]. 
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
    line_mse, = ax2.plot([], [], lw=2, label='MSE')
    ax2.set_xlim(0, N)
    if db:
        ax2.set_ylim(jnp.min(mse_vals) if len(mse_vals) > 0 else -0.5, 1)
    else: 
        ax2.set_ylim(0, jnp.max(mse_vals) * 1.1 if len(mse_vals) > 0 else 1)
    ax2.legend()
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

def splat_anim_2d(splatnns, f, xlim=[-1,1], ylim=[-1,1], Ngrid=50, interval=100, fskip=1):
    '''
    splatnns: list of tuples of (V,A,B) where V is a [k,2] real tensor, A is a [k,2,2] real tensor, B is a [k,2] real tensor
    f: a real function with real input. f(X) where X is [Ngrid*Ngrid, 2] returns [Ngrid*Ngrid, 1].
    xlim, ylim: (optional) domain for plotting
    Ngrid: (optional) number of grid points per dimension

    Call inside a jupyter notebook to generate an animation of training a k-splat regression model in 2D.
    '''
    V = jnp.stack([splatnn[0] for splatnn in splatnns], axis=0)
    A = jnp.stack([splatnn[1] for splatnn in splatnns], axis=0)
    B = jnp.stack([splatnn[2] for splatnn in splatnns], axis=0)
    
    assert V.ndim == 3 and V.shape[2] == 1, f"V should have shape [N,k,1], but has {V.shape}"
    assert A.ndim == 4 and A.shape[2] == 2 and A.shape[3] == 2, f"A should have shape [N,k,2,2], but has {A.shape}"
    assert B.ndim == 3 and B.shape[2] == 2, f"B should have shape [N,k,2], but has {B.shape}"
    N, k, _ = V.shape
    assert A.shape[0] == N and A.shape[1] == k, f"A has shape {A.shape} but should have N={N}, k={k}"
    assert B.shape[0] == N and B.shape[1] == k, f"B has shape {B.shape} but should have N={N}, k={k}"

    V = jnp.asarray(V)
    A = jnp.asarray(A)
    B = jnp.asarray(B)

    x = jnp.linspace(xlim[0], xlim[1], Ngrid)
    y = jnp.linspace(ylim[0], ylim[1], Ngrid)
    X_grid, Y_grid = jnp.meshgrid(x, y)
    grid_points = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

    f_vals = f(grid_points).reshape(Ngrid, Ngrid)

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    # Initial plot setup
    # We will update the data of this object in the animation function
    quad = ax.pcolormesh(X_grid, Y_grid, np.zeros((Ngrid, Ngrid)), shading='auto', cmap='viridis')
    fig.colorbar(quad, ax=ax)
    contour = ax.contour(X_grid, Y_grid, f_vals, colors='white', linestyles='dashed')

    def splat_func(i):
        # V_i is [k, 1], A_i is [k, 2, 2], B_i is [k, 2]
        splatnn_i = (V[i], A[i], B[i])
        # grid_points is [Ngrid*Ngrid, 2]
        # eval_splat returns [Ngrid*Ngrid, 1]
        splat_vals = eval_splat(grid_points, splatnn_i)
        return splat_vals.reshape(Ngrid, Ngrid)

    def animate(i):
        y_splat = splat_func(i)
        quad.set_array(y_splat.ravel())
        ax.set_title(f"Step {i+1}/{N}")
        # Set color limits based on current frame's data
        quad.set_clim(vmin=jnp.min(y_splat), vmax=jnp.max(y_splat))
        return quad,

    anim = FuncAnimation(fig, animate, frames=list(range(N))[::fskip], interval=interval, blit=False)
    return anim


def gd_splat_regression(init_splat, train_X, train_Y, lr=1e-4, num_steps=1000, train_mask=(1.0,1.0,1.0), verbose=False, adam=False, adam_params=(0.9,0.999,1e-8), selective_noise=None): 
    splats = []
    train_mask = jnp.array(train_mask).astype(float)

    cur_splat = init_splat

    if adam:
        b1, b2, eps = adam_params
        optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2, eps=eps)
        opt_state = optimizer.init(init_splat)

    if verbose: 
        R = tqdm(range(num_steps), desc=f"Training SRM")
    else:
        R = range(num_steps)
    
    for _ in R:
        V, A, B = cur_splat

        '''
        V = jnp.ones((10,2))
        A = jnp.array([jnp.eye(3)]*10)
        B = jnp.ones((10,3))
        train_X = jnp.ones((5,3))
        dFx_vjp = jax.vjp(lambda X: eval_splat(X, (V, A, B)), train_X)[1]
        dFx_vjp = jax.vjp(lambda X: eval_splat(X, cur_splat), train_X)[1]
        vDdFx = jax.vmap(lambda V_: dFx_vjp(jnp.tile(V_, (len(train_X), 1)))[0])(V).transpose((1, 0, 2))
        
        vDdFx = jax.vmap(lambda x: 
                    jax.vmap(lambda v: 
                        jax.vjp(lambda x_: eval_splat(x_, (V, A, B)))[0](v) )(V))(X)
                        '''
        # for debugging: vDdFx.block_until_ready() will un-lazy load it 
        variation = lambda x, y: 2*(eval_splat(x, cur_splat) - y) / len(y)
        variation = jax.jit(variation)
       
        grad_V, grad_A, grad_B = eval_splat_grad(cur_splat, train_X, train_Y, variation)

        if adam:
            grads = (
                grad_V * train_mask[0],
                grad_A * train_mask[1],
                grad_B * train_mask[2]
            )
            updates, opt_state = optimizer.update(grads, opt_state, cur_splat)
            V_, A_, B_ = optax.apply_updates(cur_splat, updates)
        else:
            V_ = V - lr * train_mask[0] * grad_V
            A_ = A - lr * train_mask[1] * grad_A
            B_ = B - lr * train_mask[2] * grad_B
       
        if selective_noise is not None:
            key = jax.random.key(0)
            noise = jnp.einsum('ijk,ik->ij', A_, jax.random.normal(key, B_.shape) * selective_noise[1] * (V_ < selective_noise[0]))
            B_ = B_ + jnp.sqrt(lr) * noise

        splats.append((V_, A_, B_))
        cur_splat = (V_, A_, B_)
        loss = jnp.mean((eval_splat(train_X, cur_splat) - train_Y)**2)
        R.set_description(f"Training SRM – log(MSE) = {jnp.log10(loss):.4f}")
    return splats

# Note: gd_splat_regression cannot be fully JIT compiled due to Python control flow
# The JIT decoration is removed to avoid compilation errors

if __name__ == "__main__":
    """
    Demo script showing splat regression animations in 1D and 2D
    """
    import jax.random as jr
    
    print("Running Splat Regression Demos...")
    
    # Demo 1: 1D Splat Animation
    print("\n=== 1D Splat Regression Demo ===")
    
    # Define target function
    def target_1d(x):
        return jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.cos(4 * jnp.pi * x)
    
    # Generate training data
    key = jr.PRNGKey(42)
    n_train = 100
    train_X_1d = jr.uniform(key, (n_train, 1), minval=-1, maxval=1)
    train_Y_1d = target_1d(train_X_1d) + 0.1 * jr.normal(jr.split(key)[0], (n_train, 1))
    
    # Initialize splat parameters
    k = 200  # number of splats
    key_init = jr.PRNGKey(123)
    keys = jr.split(key_init, 3)
    
    V_init = jr.normal(keys[0], (k, 1)) * 0.1
    A_init = jnp.abs(jr.normal(keys[1], (k, 1, 1))) * 0.3 + 0.1
    B_init = jr.uniform(keys[2], (k, 1), minval=-1, maxval=1)
    
    init_splat_1d = (V_init, A_init, B_init)
    
    # Train the model
    print("Training 1D splat model...")
    splat_history_1d = gd_splat_regression(
        init_splat_1d, train_X_1d, train_Y_1d, 
        lr=1e-3, num_steps=200, verbose=True
    )
    
    # Create animation
    print("Creating 1D animation...")
    anim_1d = splat_anim_1d(
        splat_history_1d, target_1d, train_X_1d, train_Y_1d,
        xlim=[-1.5, 1.5], interval=50, fskip=2
    )
    
    # Save animation
    try:
        anim_1d.save('splat_1d_demo.gif', writer='pillow', fps=10)
        print("Saved 1D animation as 'splat_1d_demo.gif'")
    except Exception as e:
        print(f"Could not save 1D animation: {e}")
    
    plt.show()
    
    # Demo 2: 2D Splat Animation
    print("\n=== 2D Splat Regression Demo ===")
    
    # Define target function
    def target_2d(X):
        x, y = X[:, 0:1], X[:, 1:2]
        return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) + 0.3 * jnp.exp(-(x**2 + y**2))
    
    # Generate training data
    key_2d = jr.PRNGKey(456)
    n_train_2d = 300
    train_X_2d = jr.uniform(key_2d, (n_train_2d, 2), minval=-1, maxval=1)
    train_Y_2d = target_2d(train_X_2d) + 0.05 * jr.normal(jr.split(key_2d)[0], (n_train_2d, 1))
    
    # Initialize 2D splat parameters
    k_2d = 300  # number of splats
    key_init_2d = jr.PRNGKey(789)
    keys_2d = jr.split(key_init_2d, 3)
    
    V_init_2d = jr.normal(keys_2d[0], (k_2d, 1)) * 0.1
    # Initialize A as diagonal matrices with small random values
    A_init_2d = jnp.array([jnp.diag(jnp.abs(jr.normal(keys_2d[1], (2,))) * 0.2 + 0.1) 
                          for _ in range(k_2d)])
    B_init_2d = jr.uniform(keys_2d[2], (k_2d, 2), minval=-1, maxval=1)
    
    init_splat_2d = (V_init_2d, A_init_2d, B_init_2d)
    
    # Train the 2D model
    print("Training 2D splat model...")
    splat_history_2d = gd_splat_regression(
        init_splat_2d, train_X_2d, train_Y_2d,
        lr=5e-3, num_steps=200, verbose=True
    )
    
    # Create 2D animation
    print("Creating 2D animation...")
    anim_2d = splat_anim_2d(
        splat_history_2d, target_2d,
        xlim=[-1.2, 1.2], ylim=[-1.2, 1.2], 
        Ngrid=40, interval=20, fskip=2
    )
    
    # Save animation
    try:
        anim_2d.save('splat_2d_demo.gif', writer='pillow', fps=8)
        print("Saved 2D animation as 'splat_2d_demo.gif'")
    except Exception as e:
        print(f"Could not save 2D animation: {e}")
    
    plt.show()
    
    print("\nDemo completed! Check the generated animations.")
    print("1D Demo: Shows splat components as vertical lines with error bars")
    print("2D Demo: Shows heatmap evolution with target function contours")
