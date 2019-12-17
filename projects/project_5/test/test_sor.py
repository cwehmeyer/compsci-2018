import pytest
import numpy as np
from itertools import product
from project_5.sor import poisson_solver_sor


@pytest.mark.parametrize('rho, boxsize, omega, exception', [
    ([[1, 2], [3, 4]], (1), 1.2, ValueError),
    ([[-1, 2, 3], [1]], (1, 2), 0.5, ValueError),
    ([[1]], (1, 2), 1.2, ValueError),
    ([[-1, "hello"], [1, 2]], (1, 2), 0.5, TypeError),
    ([[2], [3, 4]], (1, 2), 3, ValueError),
    ([[-1, 2, 3], [1], [4, 5, 6]], (1, 2), 0.5, ValueError)])
def test_poisson_solver_sor_exceptions(rho, boxsize, omega, exception):
    with pytest.raises(exception):
        poisson_solver_sor(rho, boxsize, omega)


@pytest.mark.parametrize(
    'nx,ny', [(nx, ny) for nx, ny in product(
        [50, 100],
        [50, 100])])
def test_cossin(nx, ny):
    """Test a known solution of Poisson's equation.
    """
    gx = np.linspace(-np.pi, np.pi, nx, endpoint=False)
    gy = np.linspace(-np.pi, np.pi, ny, endpoint=False)
    x, y = np.meshgrid(gx, gy, indexing='ij')
    phi = np.cos(x) + np.sin(y)
    np.testing.assert_allclose(
        poisson_solver_sor(phi, (2 * np.pi, 2 * np.pi))[0],
        phi,
        rtol=1e-2, atol=5e-2)
