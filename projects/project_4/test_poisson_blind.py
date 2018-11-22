import pytest
from importlib import import_module


@pytest.mark.parametrize('module,function', [
    ('.poisson','create_laplacian_1d'),
    ('.poisson_1','create_laplacian_2d'),
    ('.poisson_2','create_laplacian_2d'),
    ('.poisson_3','create_laplacian_2d'),])
def test_module_and_interface(module, function):
    """Test that the module can be imported and that it
    provides the desired function.
    """
    imported = import_module(module, package='project_4')
    assert function in dir(imported)
