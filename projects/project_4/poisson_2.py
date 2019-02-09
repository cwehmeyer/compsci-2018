import numpy as np

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
    a1 =  np.diag((-2 * hx - 2 * hy) * np.ones(nx * ny))
    diag1 = hx * np.ones(nx * ny - 1)
    diag1[nx-1::nx] = 0
    a2 = np.diag(diag1 , 1)
    a3 = np.diag(diag1 , -1)
    a4 = np.diag(hy * np.ones(nx * ny - nx), nx)
    a5 = np.diag(hy * np.ones(nx * ny - nx), -nx)
    laplacian = a1 + a2 + a3 + a4 + a5

    if pbc:
        a6 = np.diag(hy * np.ones(nx), nx * ny - nx)
        a7 = np.diag(hy * np.ones(nx), -nx * ny + nx)
        diag2 = hx * np.zeros(nx * ny - nx + 1)
        diag2[::nx] = hx
        a8 = np.diag(diag2, nx - 1)
        a9 = np.diag(diag2, -nx + 1)
        laplacian += a6 + a7 + a8 + a9

    return laplacian