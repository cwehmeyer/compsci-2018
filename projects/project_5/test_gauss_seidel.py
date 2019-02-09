import numpy as np
import pytest
from poisson_scipy import create_laplacian_2d
from gauss_seidel import compute_phi_2d


### test gauss seidel vs laplacian ###
@pytest.mark.parametrize('nx, ny', [
    (5, 5),
    (10, 5),
    (5, 10),
    (30, 50)
    ])

def test_compute_phi_2d(nx, ny):
    n = (nx * ny - 1) / 2
    rho = np.linspace(-n , n, nx * ny).reshape(ny , nx)
    laplacian = create_laplacian_2d(nx, ny, 1, 1)
    phi = compute_phi_2d(rho, [1, 1])
    rho_calc = -np.dot(laplacian, phi.reshape(-1)).reshape(ny, nx)
    np.testing.assert_almost_equal(rho, rho_calc, decimal=4)
