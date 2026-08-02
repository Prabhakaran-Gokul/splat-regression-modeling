#!/usr/bin/env python3
"""
Simple GPU Stress Test - Easy to Monitor

This script runs a simple but intensive computation that should max out GPU usage.
Perfect for testing if your JAX setup can achieve high GPU utilization.

Usage:
    python simple_gpu_test.py

Monitor with:
    nvidia-smi -l 1
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time

print("Simple JAX GPU Stress Test")
print("=" * 40)

# Show device info
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Configure for GPU
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platform_name', 'gpu')

@jax.jit
def gpu_intensive_loop(key, size, iterations):
    """Simple but intensive GPU computation"""
    def body_fn(i, carry):
        A, key_carry = carry
        key_carry, subkey = jr.split(key_carry)
        B = jr.normal(subkey, (size, size))
        C = A @ B @ A.T  # Matrix chain multiplication
        D = jnp.sin(C) + jnp.cos(C)  # Element-wise operations
        return D, key_carry
    
    A = jr.normal(key, (size, size))
    result, _ = jax.lax.fori_loop(0, iterations, body_fn, (A, key))
    return jnp.mean(result)

def main():
    print("\nStarting GPU stress test...")
    print("You should see high GPU utilization in nvidia-smi")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    key = jr.PRNGKey(42)
    size = 2048  # Large matrices
    iterations_per_batch = 50
    
    try:
        batch = 0
        start_time = time.time()
        
        while True:
            batch_start = time.time()
            
            # Run intensive computation
            key, subkey = jr.split(key)
            result = gpu_intensive_loop(subkey, size, iterations_per_batch)
            result.block_until_ready()  # Force completion
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            total_time = batch_end - start_time
            
            batch += 1
            ops_per_sec = iterations_per_batch / batch_time
            
            print(f"Batch {batch:3d} | Time: {batch_time:5.2f}s | "
                  f"Ops/sec: {ops_per_sec:6.1f} | Total: {total_time:6.1f}s | "
                  f"Result: {float(result):8.4f}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
        total_time = time.time() - start_time
        print(f"Ran for {total_time:.1f} seconds total")

if __name__ == "__main__":
    main()