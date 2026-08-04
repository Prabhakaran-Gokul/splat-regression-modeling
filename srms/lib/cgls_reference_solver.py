import numpy as np
import scipy as sp
import scipy.special, scipy.sparse, scipy.sparse.linalg, scipy.fft
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import trange


np.set_printoptions(linewidth=150, edgeitems=4)

# discrete chebyshev transform with N gridpoints on [-1, 1]
def dcht(u):
    N = len(u)
    scale = np.ones((N,)) + np.concatenate(([1], np.zeros(N-2), [1]))
    return sp.fft.dct(u/(N-1), norm="backward", type=1)/scale

# inverse discrete chebyshev transform
def idcht(u):
    N = len(u)
    scale = np.ones((N,)) + np.concatenate(([1], np.zeros(N-2), [1]))
    return (N-1) * sp.fft.idct(u*scale, norm="backward", type=1)  

# compute chebyshev extremal nodes
def gridpts(N):
    return np.cos(np.pi * np.arange(N) / (N-1))

# sparse basis transformation between ultraspherical polynomials
def S(k, N):
    '''
    k: order of the input ultraspherical basis
    N: dimension (number of basis coefficients) of input
    '''
    if(k==0):
        d0 = 0.5 * np.ones(N)
        d2 = -0.5 * np.ones(N)
        d0[0] = 1
        
    else:
        idx = np.arange(N)
        d0 = k/(k+idx)
        d2 = -k/(k+idx+2)
    return sp.sparse.diags(diagonals=[d0, d2], offsets=[0, 2], format="csc")


# sparse differentiation operator
def D(k, N):
    '''
    k: order of the 1D differential operator (input is assumed to be in chebyshev/ultraspherical-0 basis)
    N: dimension (number of basis coefficients) of input
    '''
    d = 2**(k-1) * sp.special.factorial(k-1) * (k + np.arange(N-k))
    return sp.sparse.diags(diagonals=d, offsets=k, format="csc")


def iultspht(x_hat, k):
    N = len(x_hat)
    conversion = reduce(lambda Skp1, Sk: Sk @ Skp1, [S(r, N) for r in range(k)], sp.sparse.eye(N))
    chebyshev_coeffs = sp.sparse.linalg.spsolve_triangular(conversion, x_hat, lower=False)
    return idcht(chebyshev_coeffs)

def ultspht(x, k):
    N = len(x)
    chebyshev_coeffs = dcht(x)
    conversion = reduce(lambda Skp1, Sk: Sk @ Skp1, [S(r, N) for r in range(k)], sp.sparse.eye(N))
    return conversion @ chebyshev_coeffs


def CGLE_LHS(N, h, L=300, b=0.5, c=-1.76):
    LHS = (h**(-1) - 1) *  S(1, N) @ S(0, N) - (1 + 1j*b) * (2/L)**2 * D(2, N)
    LHS[-1] = (-1)**np.arange(N)
    LHS[-2] = np.ones(N,)
    return LHS
    

def CGLE_RHS(An_hat, h, L=300, b=0.5, c=-1.76, conversion=None):
    N = len(An_hat)
    An = idcht(An_hat)
    RHS_spatial = (h**(-1) - (1+c*1j) * np.abs(An)**2)*An
    if(conversion is None):
        conversion = S(1, N) @ S(0, N)
    RHS_spectral = conversion @ dcht(RHS_spatial)
    RHS_spectral[-1] = 0
    RHS_spectral[-2] = 0
    return RHS_spectral


if __name__ == "__main__":
    N = 512
    h = 0.1
    L = 300
    b = 0.5
    c = -1.76
    iters = 2048

    xn = gridpts(N)
    LHS = CGLE_LHS(N, h)
    LHS_lu = sp.sparse.linalg.splu(LHS.T)

    solutions = []

    An = None
    Fn = CGLE_RHS(dcht(10**(-3) * np.sin(np.pi * (xn+1))), h)

    for n in trange(iters):
        An = LHS_lu.solve(Fn, trans='T')
        Fn = CGLE_RHS(An, h)
        if(n % 1 == 0):
            solutions.append(idcht(An))


    Ant = np.stack(solutions, axis=1)

    plt.subplot(3, 1, 1)
    plt.title("Amplitude")
    plt.imshow(np.abs(Ant))
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.title("log10(Amplitude)")
    plt.imshow(np.log10(np.abs(Ant)))
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.title("Phase")
    plt.imshow(np.arctan2(np.imag(Ant), np.real(Ant)))
    plt.colorbar()

    plt.gcf().set_size_inches(10, 10)
    plt.show()