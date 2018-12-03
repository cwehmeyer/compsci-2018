import numpy as np

def _find_index(x, y, nx, ny):
    """Finds the actually index in the 2-D Laplacian based on x, y coordinates.
    """
    x, y = x % nx, y % ny
    # Be careful of the order to reshape. Commented is Fortran/Matlab-style,
    # but numpy uses C-style by default
    # return x + y * nx
    return x * ny + y


def create_laplacian_2d(nx, ny, lx, ly, pbc = True):
    """Creates a discretized Laplacian in 2D

    Arguments:
        nx, ny (int > 1): number of grid points of x and y directions
        lx, ly (float > 0): box lenght along x and y
        pbc (boolean): use periodic boundary conditions
    """
    if(nx < 2 or ny < 2):
        raise ValueError('We need at least two grid points for each direction')
    if(lx <= 0.0 or ly <= 0.0):
        raise ValueError('We need a positive length')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')
    laplacian = np.zeros((nx * ny, nx * ny))
    mx = (nx / lx)**2
    my = (ny / ly)**2
    mxy = -2 * (mx + my)
    find_index = lambda x, y: _find_index(x, y, nx, ny)
    for x in range(nx):
        for y in range(ny):
            laplacian[find_index(x, y), find_index(x, y)] += mxy
            laplacian[find_index(x, y), find_index(x - 1, y)] += mx
            laplacian[find_index(x, y), find_index(x + 1, y)] += mx
            laplacian[find_index(x, y), find_index(x, y - 1)] += my
            laplacian[find_index(x, y), find_index(x, y + 1)] += my
    if not pbc:
        # modified for boundary condition calc. 
        # Should not just be set to 0, or it will be incorrect for 2x2 cases
        for y in range(ny):
            laplacian[find_index(0, y), find_index(-1, y)] -= mx
            laplacian[find_index(nx - 1, y), find_index(0, y)] -= mx
        for x in range(nx):
            laplacian[find_index(x, 0), find_index(x, -1)] -= my
            laplacian[find_index(x, ny - 1), find_index(x, 0)] -= my
    return laplacian