from v2.lib.nets import gd_splat_regression, splat_anim_1d, eval_splat
import jax.numpy as jnp
import jax.random as jr 
import matplotlib.pyplot as plt 

if __name__ == "__main__": 
    key = jr.key(3621)
    k = 1
    init_V = jnp.ones((k, 1)) 
    init_A = jnp.ones((k,1,1)) * 0.5
    init_B = jnp.linspace(-0.3, 0.3, k).reshape((k, 1))
    init_splat = (init_V, init_A, init_B)

    test_V = 0.5 * jnp.ones((k, 1)) 
    test_A = 0.1 * jnp.ones((k,1,1)) 
    test_B = jnp.linspace(-0.3, 0.3, k).reshape((k, 1)) - 0.3
    test_splat = (test_V, test_A, test_B)

    train_X = jr.uniform(key, (50, 1), minval=-1, maxval=1)

    #test_splat = init_splat
    #train_Y = eval_splat(train_X, (jnp.array([[1]]), jnp.array([[[0.1]]]), jnp.array([[0.5]])))
    train_Y = eval_splat(train_X, test_splat)
    splat_regression_trajectory = gd_splat_regression(init_splat, train_X, train_Y, train_mask=(1,1,1), num_steps=5000, lr=0.0001)
    R1 = splat_anim_1d(splat_regression_trajectory, lambda x: eval_splat(x.reshape((-1, 1)), test_splat), train_X, train_Y, xlim=[-1.5, 1.5], Ngrid=400, interval=5, fskip=5)
    R1.save('fit_animation.gif', writer='pillow', fps=30)
    plt.show()