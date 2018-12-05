import numpy as np


def poisson_solver_sor(rho, boxsize, omega=None, tol=1e-6, maxiter=2000, epsilon0=1.):
    '''Solve the Poisson equation on 2D grid with periodic boundary conditions by SOR method
    within a given tolerance. (Red-Black-Grid version)

    Inputs:
    rho (float matrix, shape (m, n)): charge distribution; m, n > 1
    boxsize (float vector, > 0, len = 2): box lengths for x and y direction
    omega (None or float between 0 and 2): relaxation factor, leave as None for auto selection
    tol (float >= 0): tolerance between nearest two iteration to stop; 0 for reaching maxiter
    maxiter (int > 0): maximum iteration number
    epsilon0 (float > 0): electric permittivity

    Outputs:
    phi (float matrix, shape (m, n)): electrostatic potential on grid
    error (float): final error
    '''

    # check the inputs
    rho = np.array(rho)
    if rho.ndim != 2 or np.shape(rho)[0] < 2 or np.shape(rho)[1] < 2:
        raise ValueError('We need at least two grid points on each direction')
    m, n = np.shape(rho)
    if np.shape(boxsize) != (2,) or boxsize[0] <= 0 or boxsize[1] <= 0:
        raise ValueError('We need correct boxsize')
    if omega is not None and (omega <= 0 or omega >= 2):
        raise ValueError('We need correct relaxation factor')
    if tol < 0:
        raise ValueError('We need correct tolerance')
    if maxiter <= 0:
        raise ValueError('We need correct iteration number')
    if epsilon0 <= 0:
        raise ValueError('We need correct electric permittivity')

    # optimal omega
    # ref: http://www.math.umbc.edu/~kogan/technical_papers/2007/Yang_Gobbert.pdf
    if omega is None:
        omega = 2 / (1 + np.sin(np.pi / max(m, n)))

    # preparing the red and black grids
    row_even, row_odd = np.arange(0, m, 2), np.arange(1, m, 2)
    col_even, col_odd = np.arange(0, n, 2), np.arange(1, n, 2)
    red_row = np.concatenate((np.tile(row_even, len(col_even)), np.tile(row_odd, len(col_odd))))
    red_col = np.concatenate((np.repeat(col_even, len(row_even)), np.repeat(col_odd, len(row_odd))))
    blk_row = np.concatenate((np.tile(row_even, len(col_odd)), np.tile(row_odd, len(col_even))))
    blk_col = np.concatenate((np.repeat(col_odd, len(row_even)), np.repeat(col_even, len(row_odd))))

    # calculate the coefficients
    one_minus_omega = 1. - omega
    hx2 = (boxsize[0] / m) ** 2
    hy2 = (boxsize[1] / n) ** 2
    coeff_x = hy2 / (2 * (hx2 + hy2))
    coeff_y = hx2 / (2 * (hx2 + hy2))
    coeff_rho = hx2 * hy2 / (2 * epsilon0 * (hx2 + hy2))
    rho = rho * coeff_rho

    # initiate variables for iteration
    phi = np.zeros((m, n))
    errors = np.zeros((m, n))
    error = 1000.
    newvalue = 0.
    iter_num = 0

    # define update function (vectorized)
    def _update(i, j):
        newvalue = one_minus_omega * phi[i, j] + \
                    omega * (coeff_x * (phi[(i - 1) % m, j] + phi[(i + 1) % m, j]) + \
                        coeff_y * (phi[i, (j - 1) % n] + phi[i, (j + 1) % n]) + \
                            rho[i, j])
        errors[i, j] = newvalue - phi[i, j]
        phi[i, j] = newvalue
    update = np.vectorize(_update, signature='(n),(m)->()')

    # iteration
    while error > tol and iter_num < maxiter:
        # errors.fill(0.)
        update(red_row, red_col)
        update(blk_row, blk_col)
        error = np.linalg.norm(errors)
        # print("one iter finished.")
        iter_num += 1

    return phi, error
