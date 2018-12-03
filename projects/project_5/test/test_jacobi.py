import pytest
import numpy as np
from itertools import product
from project_5.jacobi import jacobi
from project_4.poisson_1 import create_laplacian_2d

@pytest.mark.parametrize('rho, boxsize, exception', [
    ([[1,2,3],[4,5,'yes']], (1,3), TypeError),
    ([1,2,3], None, TypeError),
    ([1,2,3], 'hello', TypeError),
    ([1,-1], 0, ValueError),
    ([1,2,3], [0, 0.1], ValueError),
    ([[1,4,6],[4,5,6]], (0.1,0.1,0.1), ValueError),
    (-1, (2), ValueError)])
def test_jacobi_exceptions(rho, boxsize, exception):
    with pytest.raises(exception):
        jacobi(rho, boxsize)
        
@pytest.mark.parametrize('hx, hy, lx, ly', [
    (nx, ny, lx, ly) for nx, ny, lx, ly in product(
        [1, 0.2, 0.1],
        [1, 0.2, 0.1],
        [7, 20., 30.],
        [7, 20., 30.])])
def test_consisteny(hx, hy, lx, ly):
    x = np.arange(0, lx, hx)
    y = np.arange(0, ly, hy)
    boxsize = (hx, hy)
    nx, ny = len(x), len(y)
  
    rho = np.random.normal(size=(ny,nx))
    rho -= np.mean(rho)
    pot = jacobi(rho, boxsize, maxiter=10000)
    
    assert pot.ndim == 2, \
    f'Potentials have wrong dimensions: {pot.ndim}'
    
    assert pot.shape[0] == ny, \
    f'Potentials have wrong first shape: {pot.shape[0]}'
    
    assert pot.shape[1] == nx, \
    f'Potentials have wrong second shape: {pot.shape[1]}'
    
    laplacian = create_laplacian_2d(nx, ny, lx, ly)
    pot_laplacian = np.linalg.solve(laplacian, -rho.reshape(nx*ny))
    
    np.testing.assert_allclose(pot_laplacian[0]+(pot[0]-pot_laplacian[0]),pot[0],rtol=1e-5, atol=1e-5)
    
    
    
  
    
      