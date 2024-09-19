import numpy as np
from functools import reduce, partial
from typing import Callable
from numpy.typing import NDArray


def fit_polynomial(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    '''
    Finds the best-fit polynomial for a set of datapoints (X,Y) by solving
    a least-squares problem (X^T X)a = X^T y
    
    Parameters:
    * x, an NDArray(r,n) of n number of x-points for r data sets.
    * y, an NDarray(r,n) of n number of y-points forr r data sets.
    * degree, an int to denote the degree of the polynomial.

    Output:
    The best-fit coeffients of type NDArray(r,degree+1) such that NDArray(r,i)
    is coefficient of x^i for all i in NDArray for data set r.
    '''
    # Design matrix
    Xs = np.array([np.vander(x_row, degree + 1, increasing=True) for x_row in x])

    # Compute coefficients "a" using the formula:
    # (X^T X)a = X^T y -> a = (X^T X)^-1 X^T y   
    return np.array([np.linalg.inv(Xi.T @ Xi) @ Xi.T @ yi for Xi, yi in zip(Xs, y)])

def parabolic_area(coefficients: NDArray[np.float64], x: NDArray[np.int64]) -> NDArray[np.float64]:
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
    c = coefficients * np.array([1,1/2,1/3])

    x_x_sq_x_cb = np.array([np.diff(x), np.diff(x**2), np.diff(x**3)]).transpose(1,0,2)

    areas = np.einsum('ij,ijk->i', c, np.abs(x_x_sq_x_cb))

    return areas

