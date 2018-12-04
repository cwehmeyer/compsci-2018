import pytest
import numpy as np
from itertools import product
from project_5.jacobi import jacobi
from project_4.poisson_2 import create_laplacian_2d

@pytest.mark.parametrize('rho, boxsize, exception', [
    ([[1,2,-3],[4,5,'yes']], (1,3), TypeError),
    ([1,-1], None, TypeError),
    ([1,2,-3], 'hello', TypeError),
    ([1,2,-3], [[2, 1]], TypeError),
    ([1,-1], 0, ValueError),
    ([1,2,-3], [0,0.1], ValueError),
    ([[1,-1],[-1,1]], (0.1,0.1,0.1), ValueError),
    (-1, (2), ValueError),
    ([[]], (0.1,0.1), ValueError),
    ([[1,2,-3]],(0.1,0.1), ValueError),
    ([1, 1], (0.1), ValueError)])
def test_jacobi_exceptions(rho, boxsize, exception):
    with pytest.raises(exception):
        jacobi(rho, boxsize)
        
@pytest.mark.parametrize('nx, ny, lx, ly', [
    (nx, ny, lx, ly) for nx, ny, lx, ly in product(
        [5, 10, 20],
        [5, 10, 20],
        [1.0, 3.0],
        [1.0, 3.0])])
def test_consisteny(nx, ny, lx, ly):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    hx, hy = lx/nx, ly/ny
    boxsize = (hx, hy)
  
    rho = np.random.normal(size=(ny,nx))
    rho -= np.mean(rho)
    laplacian = create_laplacian_2d(nx, ny, lx, ly)
    pot = jacobi(rho, boxsize, maxiter=5000)
    
    assert pot.ndim == 2, \
    f'Potentials have wrong dimensions: {pot.ndim}'
    
    assert pot.shape[0] == ny, \
    f'Potentials have wrong first shape: {pot.shape[0]}'
    
    assert pot.shape[1] == nx, \
    f'Potentials have wrong second shape: {pot.shape[1]}'
    
    laplacian = create_laplacian_2d(nx, ny, lx, ly)
    pot_laplacian = np.linalg.solve(laplacian, -rho.reshape(nx*ny))
    pot_laplacian = pot_laplacian.reshape((ny,nx))
    pot_laplacian = pot_laplacian + (pot[0, 0] - pot_laplacian[0, 0]) #Equivalent potentials can differ by a constant.
    
    np.testing.assert_allclose(pot_laplacian, pot, rtol=1e-2, atol=1e-2)
    
    
    
  
    
      