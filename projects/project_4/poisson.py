import numpy as np

def create_laplacian_1d(nx, lx, pbc=True):
    """Ceates a discretized Laplacian in 1D

    Arguments:
        nx (int): number of grid points; needs more than one
        lx (float): box lenght along x; must be positive
        pbc (boolean): use periodic boundary conditions
    """
    if nx < 2:
        raise ValueError('We need at least two grid points')
    if lx <= 0.0:
        raise ValueError('We need a positive length')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')
    laplacian = np.zeros((nx, nx))
    mx = (nx / lx)**2
    for x in range(nx):
        laplacian[x, x] -= 2.0 * mx
        laplacian[x, (x + 1) % nx] += mx
        laplacian[(x + 1) % nx, x] += mx
    if not pbc:
        laplacian[0, -1] = 0
        laplacian[-1, 0] = 0
    return laplacian
