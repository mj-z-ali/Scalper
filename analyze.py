import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import polynomial as poly


def barrier_mask(columns: NDArray[np.uint64], l: NDArray[np.uint64], i: NDArray[np.uint64], r: NDArray[np.uint64]) -> tuple[Callable, Callable, Callable]:

    m = (columns <= l[:,None]) | (columns >= r[:,None])

    return lambda : m, \
           lambda : m | (columns == i[:,None]), \
           lambda n, j : n | (columns == j[:,None])

def upper_vertices(q: NDArray[np.float64], f: Callable, g: Callable) -> tuple[Callable, Callable]:
    
    m = f()

    return lambda : np.argmax(np.where(m, -np.inf, q), axis=1), \
           lambda v : np.argmax(np.where(g(m, v), -np.inf, q), axis=1)

def lower_vertices(q: NDArray[np.float64], f: Callable, g: Callable) -> tuple[Callable, Callable]:
    
    m = f()

    return lambda : np.argmin(np.where(m, np.inf, q), axis=1), \
           lambda v : np.argmin(np.where(g(m, v), np.inf, q), axis=1)

def vertices(columns: NDArray[np.uint64], l: NDArray[np.uint64], i: NDArray[np.uint64], r: NDArray[np.uint64]) -> tuple[Callable, Callable]:
    
    t = barrier_mask(columns, l, i, r)

    return lambda q: vertices_pair(upper_vertices(q, t[0], t[2])), \
           lambda q: vertices_pair(lower_vertices(q, t[1], t[2]))

def vertices_pair(t: [Callable, Callable]) -> tuple[NDArray[np.uint64]]:

    first_vertices = t[0]()
    second_vertices = t[1](first_vertices)

    return first_vertices, second_vertices

def inner_points_matrix(f: Callable) -> NDArray[np.float64]:

    m = f()

    return lambda q: np.where(m, 0, q)

def k_rmsd(k: np.float64, q: NDArray[np.float64], l: NDArray[np.uint64], r: NDArray[np.uint64]) -> NDArray[np.float64]:

    return np.power(np.sqrt(np.divide(np.sum(q**2, axis=1), r-l)), 1/k)

def parabolic_area_enclosed(l_x: NDArray[np.uint64], v_x: NDArray[np.uint64], r_x: NDArray[np.uint64], i_p: NDArray[np.float64], v_p: NDArray[np.float64]) -> NDArray[np.float64]:

    coefficients = poly.fit_polynomial(np.column_stack((l_x, v_x, r_x)), np.column_stack((i_p, v_p, i_p)), 2)

    return ((i_p * r_x) - (i_p * l_x))  - poly.parabolic_area(coefficients, np.column_stack((l_x, r_x)))

def sort_cubic_coordinates(v_x_0: NDArray[np.uint64], v_x_1: NDArray[np.uint64]) -> tuple[Callable, Callable]:

    v_x = np.column_stack((v_x_0, v_x_1))

    v_x_s = np.argsort(v_x, axis=1)

    return lambda : np.take_along_axis(v_x, v_x_s, axis=1), \
           lambda v_0, v_1: np.take_along_axis(np.column_stack((v_0, v_1)), v_x_s, axis=1)

def cubic_area_enclosed(l_x: NDArray[np.uint64], v_x_0: NDArray[np.uint64], v_x_1: NDArray[np.uint64], r_x: NDArray[np.uint64], i_p: NDArray[np.float64], v_p_0: NDArray[np.float64], v_p_1: NDArray[np.float64]) -> NDArray[np.float64]:

    v = sort_cubic_coordinates(v_x_0, v_x_1)

    coefficients = poly.fit_polynomial(np.column_stack((l_x, v[0](), r_x)), np.column_stack((i_p, v[1](v_p_0, v_p_1), i_p)), 3)

    return ((i_p * r_x) - (i_p * l_x))  - poly.parabolic_area(coefficients, np.column_stack((l_x, r_x)))

def upper_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.triu(np.ones(s, dtype=bool), k=1)

def lower_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.tril(np.ones(s, dtype=bool), k=-1)

def left_mask(m: NDArray[np.bool_]) -> NDArray[np.bool_]:

    return m & lower_tri_mask(m.shape)

def right_mask(m: NDArray[np.bool_]) -> NDArray[np.bool_]:

    return m & upper_tri_mask(m.shape)

def last_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return m.shape[1] - np.argmax(m[:,::-1], axis=1) - 1

def first_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return np.argmax(m, axis=1)

def right_barrier_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_points = lambda p: np.where(p == 0, len(p), p)

    return default_points(first_points(right_mask(m)))

def left_barrier_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_points = lambda p: np.where(p == len(p)-1, -1, p)

    return default_points(last_points(left_mask(m)))

def barrier_points(q: NDArray[np.float64]) -> Callable[[Callable], NDArray[np.uint64]]:

    m = (q > 0)

    return lambda f: f(m)

def initial_right_points_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return right_mask(q > 0)

def next_right_points_mask(columns: NDArray[np.uint64], m: NDArray[np.bool_], r: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return m & (columns != r[:,None])
    
def right_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return first_points(m)

def initial_validate(i: np.uint64, l: NDArray[np.uint64], r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b) & ((r - l) > i)

def next_validate(r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b)

def recursive_fun():

    m = right_points_mask()

    r = first_right_points(m)

    l = first_left_points()

    b = barrier_points()

    v = validate(r,l,b)

    d = data(r[v], l[v], b[v])

    n = new_mask(m[v], r[v])