import numpy as np
"""
Creates a Laplacian in 2D

Arguments:
    nx, ny (int): number of grid points of x and y directions
    lx, ly (float): box lenght along x and y
    pbc (boolean): use periodic boundary condition

"""


def create_laplacian_2d(nx, ny, lx, ly, pbc=True):
    if(nx < 2 or ny < 2):
        raise ValueError('We need at least two grid points')
    if(lx <= 0 or ly <= 0):
        raise ValueError('We need a positive length')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')
    max_matrix_entry = ny * nx
    mx = (nx / lx)**2
    my = (ny / ly)**2
    laplacian = np.zeros([max_matrix_entry, max_matrix_entry])
    if pbc:
        for x in range(nx):
            for y in range(ny):
                i = x * ny + y
                laplacian[i, i] -= 2 * (mx + my)
                laplacian[i, (i+ny) % max_matrix_entry] += 1 * mx
                laplacian[i, (i-ny) % max_matrix_entry] += 1 * mx
                laplacian[i, (i+1) % max_matrix_entry] += 1 * my
                laplacian[i, (i-1) % max_matrix_entry] += 1 * my
    if not pbc:
        for x in range(nx):
            for y in range(ny):
                i = x * ny + y
                laplacian[i, i] -= 2 * (mx + my)
                if i+ny < max_matrix_entry:
                    laplacian[i, i+ny] += 1 * mx
                if i-ny >= 0:
                    laplacian[i, i-ny] += 1 * mx
                if i+1 < max_matrix_entry:
                    laplacian[i, i+1] += 1 * my
                if i-1 >= max_matrix_entry:
                    laplacian[i, i-1] += 1 * my

    return laplacian




