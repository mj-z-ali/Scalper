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
    print(f"x: {x} \n y: {y} ")
    # Designe matrix
    X = np.vander(x, degree+1, increasing=True)

    X_T = X.T

    # Compute coefficients "a" using the formula:
    # (X^T X)a = X^T y -> a = (X^T X)^-1 X^T y
    coefficients = np.linalg.inv(X_T @ X) @ X_T @ y
    print(f"f''(x) = {2*coefficients[2]}")
    return coefficients

def parabola_area(coefficients: NDArray[np.float64], x: NDArray[np.int64]) -> NDArray[np.float64]:
    '''
    Computes the absolute area of a parabola using the formula
    A = a/3(x_1^3 - x_0^3) + b/2(x_1^2 - x_0^2) + c(x_1 - x_0), 
    which is the formula derived from the definite integral[x_0,x_1] 
    of ax^2 + bx + c.

    Parameters: 
    * coefficients, an NDArray(n,3) of coefficients [c,b,a] for n parabolas.
    * x, an NDArray(n,2) of x ranges [x_0, x_1] for n parabolas.

    Output: 
    An NDArray(n,) of floating points denoting the areas of each parabola.
    '''
    
    x_x_sq_x_cb = np.array([np.diff(x), np.diff(x**2), np.diff(x**3)]).transpose(1,0,2)

    areas = np.einsum('ij,ijk->i', coefficients, x_x_sq_x_cb)

    return areas

