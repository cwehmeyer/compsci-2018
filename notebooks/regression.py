"""An example module which provides a linear regression function"""


def mean(a):
    """Compute the arithmetic mean over an iterable a"""
    try:
        return sum(a) / len(a)
    except ZeroDivisionError:
        return 0


def scalar_product(a, b):
    """Compute the scalar product for two vectors a and b"""
    if len(a) != len(b):
        raise ValueError(
            f'Cannot compute a scalar product for vectors with'
            f' lengths {len(a)} and {len(b)}')
    return sum(a_ * b_ for a_, b_ in zip(a, b))


def linear_regression(x, y):
    """Perform a linear regression

    Estimate a model y_ = slope * x + const such
    that y_ approximates y.

    Arguments:
        x (iterable of float): x values
        y (iterable of float): y values

    Returns:
        slope (float): the slope parameter of the regression model
        const (float): the constant parameter of the regression model
    """
    x_mean = mean(x)
    y_mean = mean(y)
    x_meanfree = [x_ - x_mean for x_ in x]
    y_meanfree = [y_ - y_mean for y_ in y]
    xy = scalar_product(x_meanfree, y_meanfree)
    xx = scalar_product(x_meanfree, x_meanfree)
    try:
        slope = xy / xx
    except ZeroDivisionError:
        slope = 0
    const = y_mean - slope * x_mean
    return slope, const
