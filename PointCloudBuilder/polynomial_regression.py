import numpy as np


def func(data, a, b, c, d, e):
    # unpacking the multi-dim. array column-wise, that's why the transpose
    x, y = data.T

    return a + (b * x) + (c * y) + (d * x ** 2) + (e * x * y)


def func_array(data, a, b, c, d, e):
    # unpacking the multi-dim. array column-wise, that's why the transpose
    result = np.zeros(shape=(data.shape[0]))
    for row in range(0, data.shape[0]):
        x, y = data[row][0]
        result[row] = a + (b*x) + (c*y) + (d*x**2) + (e*x*y)
    return result
