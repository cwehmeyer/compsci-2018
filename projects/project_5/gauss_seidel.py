import numpy as np


def compute_phi_2d(rho, l, max_it=1E4, f_tol=1E-7, epsilon_0=1):
    """
        Computes potential from charge distribution using
        Gauss Seidel Method. Does not check for correct input.
        Arguments:
            rho (np.array): Charge distribution. axis=0 is y, axis=1 is x
                Ensure that sum(rho) = 0
            l (list): list containing lx, ly. min(l) > 0
            f_tol(float): Error tolerance
                Iteration is terminated if change of norm of potenital < f_tol
            max_it(int): maximum number of iterations
            epsilon_0(float): vacuum permittivity
    """
    ### define relevant variables ###
    nx = rho.shape[1]
    ny = rho.shape[0]
    n_min = min([nx, ny])
    hx = (nx / l[0]) ** 2
    hy = (ny / l[1]) ** 2
    hxhy = 1 / (2 * hx + 2 * hy)
    phi = np.zeros((ny, nx))
    norm = 1
    n_it = 0

    ### define indexing lists for loop ###
    index_y  = [ny - i if ny - i > 0 else 0 for i in range(1, nx + ny)]
    index_x  = [i - ny if i - ny > 0 else 0 for i in range(1, nx + ny)]
    indices  = [(a,b) for a,b in zip(index_y,index_x)]
    diag_len = [n_min for i in np.ones(nx + ny - 1, dtype=np.int8)]
    for i in range(n_min):
        diag_len[i] = i + 1
        diag_len[-i-1] = i + 1

    ### run main loop ###
    while norm > f_tol and n_it < max_it:
        n_it += 1
        norm_old = np.linalg.norm(phi)
        for k, l in enumerate(indices):
            for m in range(diag_len[k]):
                i = l[0] + m
                j = l[1] + m
                phi_up = phi[i-1,j]
                phi_left = phi[i,j-1]
                if i < ny - 1:
                    phi_down = phi[i+1,j]
                else:
                    phi_down = phi[0,j]
                if j < nx - 1:
                    phi_right = phi[i,j+1]
                else:
                    phi_right = phi[i,0]
                phi[i,j] = (hxhy * ( hx * (phi_right + phi_left)
                                   + hy * (phi_up + phi_down)
                                   + 1 / epsilon_0 * rho[i,j]
                                   ))
        norm_new = np.linalg.norm(phi)
        norm = abs(norm_old - norm_new)

    return phi
