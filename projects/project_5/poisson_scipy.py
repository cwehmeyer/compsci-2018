import numpy as np
from scipy.sparse import diags


def create_laplacian_2d(nx, ny, lx, ly, pbc=True):
    """ Computes discrete Laplacian for a 2d
        charge density matrix, ordered row-wise
        Args:
            nx(int  >= 2): number of grid points along x axis
            ny(int  >= 2): number of grid points along y axis
            lx(float > 0): length of grid along x axis
            ly(float > 0): length of grid along y axis
            pbc(boolean): periodic boundry conditions
        output:
            Laplacian as nx * ny by nx * ny np.array
    """
    if type(nx) != int or type(ny) != int:
        raise TypeError('We need an integer')
    if type(lx) != int and type(lx) != float:
        raise TypeError('We need a number')
    if type(ly) != int and type(ly) != float:
        raise TypeError('We need a number')
    if nx < 2 or ny < 2:
        raise ValueError('We need at least two grid points')
    if lx <= 0 or ly <= 0:
        raise ValueError('We need a positive length')
    if type(pbc) != bool:
        raise TypeError('We need a boolean as pbc')

    hx = (nx / lx) ** 2
    hy = (ny / ly) ** 2
    diag0 = (-2 * hx - 2 * hy) * np.ones(nx * ny)
    diag1 = hx * np.ones(nx * ny - 1)
    diag1[nx-1::nx] = 0
    diag2 = hy * np.ones(nx * ny - nx)
    diagonals = [diag0, diag1, diag1, diag2, diag2]
    offsets   = [0, 1, -1, nx, -nx]

    if pbc:
        diag4 = hy * np.ones(nx)
        diag5 = hx * np.zeros(nx * ny - nx + 1)
        diag5[::nx] = hx
        diagonals.extend([diag4, diag4, diag5, diag5])
        offsets.extend([nx * ny - nx, nx - nx * ny, nx - 1, 1 - nx])

    return diags(diagonals, offsets).todense()
