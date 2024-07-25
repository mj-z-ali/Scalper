import numpy as np
from functools import reduce, partial
from typing import Callable
from numpy.typing import NDArray


def fit_polynomial(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> NDArray[np.float64]:

    '''
    Finds the best-fit polynomial for a set of datapoints (X,Y) by solving
    a least-squares problem (X^T X)a = X^T y
    
    Parameters:
    * x, an NDArray(n,) of n number of x-points.
    * y, an NDarray(n,) of n number of y-points.
    * degree, an int to denote the degree of the polynomial.

    Output:
    The best-fit coeffients of type NDArray(degree+1,) such that NDArray(i)
    is coefficient of x^i for all i in NDArray.
    '''

    # Designe matrix
    X = np.vander(x, degree+1, increasing=True)

    X_T = X.T

    # Compute coefficients "a" using the formula:
    # (X^T X)a = X^T y -> a = (X^T X)^-1 X^T y
    coefficients = np.linalg.inv(X_T @ X) @ X_T @ y

    return coefficients