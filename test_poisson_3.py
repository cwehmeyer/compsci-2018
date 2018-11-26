import pytest
import numpy as np
from .poisson_3 import create_laplacian_2d

def test_laplacian_2d():
    laplacian = create_laplacian_2d(3, 3, 1, 1, pbc=True)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-4.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [1.0, -4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                  [1.0, 1.0, -4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, -4.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0, 1.0, -4.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, 1.0, 1.0],
                  [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -4.0, 1.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -4.0]]) * (3 / 1)**2)
    laplacian = create_laplacian_2d(3, 3, 1, 1, pbc=False)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, -4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, -4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, -4.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 1.0, -4.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -4.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -4.0]]) * (3 / 1)**2)

@pytest.mark.parametrize('nx,ny,lx,ly,pbc,exception', [
    (1, 1, 1, 1, True, ValueError),
    (3, 3, -1, -1, True, ValueError),
    ('hello', 'hello', 1, 1, True, TypeError),
    (3, 3, None, None, True, TypeError),
    (3, 3, 1, 1, 'hello', TypeError)])
def test_laplacian_2d_exceptions(nx, ny, lx, ly, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_2d(nx, ny, lx, ly, pbc=pbc)




