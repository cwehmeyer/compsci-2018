import numpy as np

def jacobi_2d(charge_dist, boxsize, maxiter=100, tol=10**(-5), epsilon0=1.0):
    hx, hy = boxsize
    rho = np.asarray(charge_dist)
    nx, ny = rho.shape
    pot = rho
    dif = np.inf
    hxy = (hx**2+hy**2)/(2*hx**2*hy**2)
    while dif < tol and counter<maxiter:
        pot_next = np.zeros(shape=(nx,ny))
        for i in range(ny):
            for j in range(nx):
                pot_next[i, j] = hxy*( (pot[(i+1)%ny,j] + pot[(i-1)%ny, j])/hy**2 + (pot[i, (j+1)%nx] + pot[i, (j-1)%nx])/hx**2 + rho[i,j]/epsilon0)
        pot = pot_next[::]
        counter+=1
    return pot              
