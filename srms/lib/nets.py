import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jax.scipy.stats import norm
from jax.scipy.linalg import solve
from tqdm import trange 
import optax
from flax import nnx
from jaxkan.KAN import KAN


def eval_kan(X, kan_model, kan_params):
    """
    Evaluates a KAN model for a given input X and parameters.
    
    Args:
        X (jnp.ndarray): Input data of shape [n, d].
        kan_model (KAN): An instance of the jaxkan.KAN model.
        kan_params (nnx.State): The parameters of the KAN model.
        
    Returns:
        jnp.ndarray: The model's output.
    """
    return kan_model(X, params=kan_params)


def gd_net_regression(model, train_X, train_Y, lr=1e-3, num_steps=1000, verbose=False, adam=False, adam_params=(0.9, 0.999, 1e-8)):
    """
    Performs gradient descent to train an nnx-based model (works for both MLP and KAN).
    
    Args:
        model (nnx.Module): An nnx model instance.
        train_X (jnp.ndarray): Training input data.
        train_Y (jnp.ndarray): Training target data.
        lr (float): Learning rate.
        num_steps (int): Number of gradient descent steps.
        verbose (bool): If True, displays a progress bar.
        adam (bool): If True, uses the Adam optimizer. Otherwise, uses standard SGD.
        adam_params (tuple): Parameters for the Adam optimizer.
        
    Returns:
        list: A list containing the model parameters at each step of the training.
    """
    params_trajectory = []

    if adam:
        b1, b2, eps = adam_params
        optimizer = nnx.Optimizer(model, optax.adam(lr, b1, b2, eps=eps), wrt=nnx.Param)
    else:
        optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)

    @jax.jit
    def loss_fn(kan_model):
        y_pred = kan_model(train_X)
        return jnp.mean((y_pred - train_Y)**2)

    grad_fn = nnx.grad(loss_fn)
    R = trange(num_steps) if verbose else range(num_steps)
    
    for _ in R:
        grads = grad_fn(model)
        
        optimizer.update(model, grads)
        params_trajectory.append(nnx.state(model))
        print(loss_fn(model))

    return params_trajectory


'''
def gd_kan_regression(init_kan_model, train_X, train_Y, lr=1e-3, num_steps=1000, verbose=False, adam=False, adam_params=(0.9, 0.999, 1e-8)):
    """
    Performs gradient descent to train a KAN model.

    Args:
        kan_model (KAN): A jaxkan.KAN model instance.
        init_params (pytree): The initial parameters for the model.
        train_X (jnp.ndarray): Training input data.
        train_Y (jnp.ndarray): Training target data.
        lr (float): Learning rate.
        num_steps (int): Number of gradient descent steps.
        verbose (bool): If True, displays a progress bar.
        adam (bool): If True, uses the Adam optimizer. Otherwise, uses standard SGD.
        adam_params (tuple): Parameters for the Adam optimizer.

    Returns:
        list: A list containing the model parameters at each step of the training.
    """
    kan_trajectory = [nnx.state(init_kan_model)]

    if adam:
        b1, b2, eps = adam_params
        optimizer = nnx.Optimizer(init_kan_model, optax.adam(lr, b1, b2, eps=eps), wrt=nnx.Param)
    else:
        optimizer = nnx.Optimizer(init_kan_model, optax.sgd(lr), wrt=nnx.Param)

    # Define the least squares loss function for KAN
    @jax.jit
    def loss_fn(kan_model):
        y_pred = kan_model(train_X)
        return jnp.mean((y_pred - train_Y)**2)

    grad_fn = nnx.grad(loss_fn)
    R = trange(num_steps) if verbose else range(num_steps)
    
    for _ in R:
        grads = grad_fn(init_kan_model)
        
        optimizer.update(init_kan_model, grads)
        kan_trajectory.append(nnx.state(init_kan_model))
        print(loss_fn(init_kan_model))

    return kan_trajectory


def eval_mlp(X, mlp_model, mlp_params):
    """
    Evaluates an MLP model for a given input X and parameters.
    
    Args:
        X (jnp.ndarray): Input data of shape [n, d].
        mlp_model (nnx.Module): An instance of an nnx MLP model.
        mlp_params (nnx.State): The parameters of the MLP model.
        
    Returns:
        jnp.ndarray: The model's output.
    """
    return mlp_model(X, params=mlp_params)

'''