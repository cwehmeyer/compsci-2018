import pytest
from regression import mean, scalar_product, linear_regression


@pytest.mark.parametrize('sequence,mu', [
    ([0, 0], 0),
    ([1, 1], 1),
    ([float(i + 1) for i in range(100)], 50.5),
    ([], 0)])
def test_mean(sequence, mu):
    assert mean(sequence) == mu


@pytest.mark.parametrize('argument,exception', [
    (None, TypeError),
    (1, TypeError),
    ('hello, world', TypeError)])
def test_mean_exceptions(argument, exception):
    with pytest.raises(exception):
        mean(argument)


@pytest.mark.parametrize('a,b', [
    ([1, 1], [0, 0]),
    ([0, 1], [1, 0]),
    ([1, 1], [1, -1])])
def test_scalar_product_orthogonal(a, b):
    assert scalar_product(a, b) == 0


@pytest.mark.parametrize('x,expected', [
    ([], 0),
    ([1, 1], 2),
    ([3, 4], 25)])
def test_scalar_product_squared_norm(x, expected):
    assert scalar_product(x, x) == expected


@pytest.mark.parametrize('arguments,exception', [
    (None, TypeError),
    ([1, 1], TypeError),
    ([[1], 1], TypeError),
    ([1, [1]], TypeError),
    ([[1, 1], [1]], ValueError),
    ([[1], [1, 1]], ValueError),
    (['hello', 'world'], TypeError)])
def test_scalar_product_exceptions(arguments, exception):
    with pytest.raises(exception):
        scalar_product(*arguments)


@pytest.mark.parametrize('x,y,slope,const', [
    ([], [], 0, 0),
    ([0, 1], [0, 0], 0, 0),
    ([0, 1], [1, 1], 0, 1),
    ([0, 1], [0, 1], 1, 0),
    ([0, 1], [1, 0], -1, 1)])
def test_linear_regression(x, y, slope, const):
    slope_, const_ = linear_regression(x, y)
    assert slope == slope_
    assert const == const_


@pytest.mark.parametrize('arguments,exception', [
    (None, TypeError),
    ([1, 1], TypeError),
    ([[1], 1], TypeError),
    ([1, [1]], TypeError),
    ([[1, 1], [1]], ValueError),
    ([[1], [1, 1]], ValueError),
    (['hello', 'world'], TypeError)])
def test_linear_regression_exceptions(arguments, exception):
    with pytest.raises(exception):
        linear_regression(*arguments)
