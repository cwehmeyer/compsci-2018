import numpy as np
from .poisson import create_laplacian_1d

def create_laplacian_2d(nx, lx, ny, ly, pbc=True):
    """Ceates a discretized Laplacian in 2D using Kronecker-products

    Arguments:
        nx (int): number of grid points in x-direction; needs more than one
        ny (int): number of grid points in y-direction; needs more than one
        lx (float): box lenght along x; must be positive
        ly (float): box lenght along y; must be positive
        pbc (boolean): use periodic boundary conditions
    """
    laplacian_x = create_laplacian_1d(nx, lx, pbc)
    laplacian_y = create_laplacian_1d(ny, ly, pbc)
    ey, ex = np.identity(ny), np.identity(nx)
    
    laplacian = np.kron(laplacian_x, ey) + np.kron(ex, laplacian_y)

    return laplacian