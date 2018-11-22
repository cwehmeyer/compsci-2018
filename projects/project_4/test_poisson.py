import pytest
import numpy as np
from .poisson import create_laplacian_1d


def test_laplacian_1d():
    laplacian = create_laplacian_1d(3, 1, pbc=True)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-2.0, 1.0, 1.0],
                  [1.0, -2.0, 1.0],
                  [1.0, 1.0, -2.0]]) * (3 / 1)**2)
    laplacian = create_laplacian_1d(3, 1, pbc=False)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-2.0, 1.0, 0.0],
                  [1.0, -2.0, 1.0],
                  [0.0, 1.0, -2.0]]) * (3 / 1)**2)


@pytest.mark.parametrize('nx,lx,pbc,exception', [
    (1, 1, True, ValueError),
    (3, -1, True, ValueError),
    ('hello', 1, True, TypeError),
    (3, None, True, TypeError),
    (3, 1, 'hello', TypeError)])
def test_laplacian_1d_exceptions(nx, lx, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_1d(nx, lx, pbc=pbc)
