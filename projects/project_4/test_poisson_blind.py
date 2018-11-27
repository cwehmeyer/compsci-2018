import numpy as np
import pytest
from importlib import import_module
from itertools import product
try:
    from .poisson_1 import create_laplacian_2d as create_laplacian_2d_1
except ImportError:
    print('Cannot import poisson_1.create_laplacian_2d')
try:
    from .poisson_2 import create_laplacian_2d as create_laplacian_2d_2
except ImportError:
    print('Cannot import poisson_2.create_laplacian_2d')
try:
    from .poisson_3 import create_laplacian_2d as create_laplacian_2d_3
except ImportError:
    print('Cannot import poisson_3.create_laplacian_2d')


@pytest.mark.parametrize(
    'module,function', [
        ('.poisson','create_laplacian_1d'),
        ('.poisson_1','create_laplacian_2d'),
        ('.poisson_2','create_laplacian_2d'),
        ('.poisson_3','create_laplacian_2d')])
def test_module_and_interface(module, function):
    """Test that the module can be imported and that it
    provides the desired function.
    """
    imported = import_module(module, package='project_4')
    assert function in dir(imported)


@pytest.mark.parametrize(
    'create_laplacian_2d,nx,ny,lx,ly', [
    (func, nx, ny, lx, ly)
    for func, nx, ny, lx, ly in product(
        [
            'create_laplacian_2d_1',
            'create_laplacian_2d_2',
            'create_laplacian_2d_3'],
        [5, 10, 20],
        [5, 10, 20],
        [1.0, 3.0],
        [1.0, 3.0])])
def test_consistency(create_laplacian_2d, nx, ny, lx, ly):
    laplacian = eval(
        f'{create_laplacian_2d}({nx}, {ny}, {lx}, {ly}, pbc=True)')
    assert laplacian.ndim == 2, \
        f'laplacian has wrong dimension: {laplacian.ndim}'
    assert laplacian.shape[0] == nx * ny, \
        f'laplacian has wrong first shape: {laplacian.shape[0]}'
    assert laplacian.shape[1] == nx * ny, \
        f'laplacian has wrong second shape: {laplacian.shape[1]}'
    rho = np.random.normal(size=(nx, ny))
    rho -= np.mean(rho)
    phi = np.linalg.solve(laplacian, -rho.reshape(-1))
    np.testing.assert_allclose(
        -np.dot(laplacian, phi).reshape(nx, ny),
        rho,
        rtol=1e-5, atol=1e-5)
