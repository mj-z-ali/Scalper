import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import polynomial as poly

def set_mask_coordinates(columns: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return lambda l,r: (columns <= l[:,None]) | (columns >= r[:,None]), \
           lambda m,c: m | (columns == c[:,None]), \
           lambda m,c: m & (columns != c[:,None])

def preliminary_data(q_t: NDArray[np.float64], q_b: NDArray[np.float64], l: NDArray[np.uint64], i: NDArray[np.uint64], r: NDArray[np.uint64], f: Callable, g: Callable) -> tuple[Callable, Callable, Callable]:

    m = f(l,r)
    m_i = g(m,i) 

    return lambda : upper_vertices(q_t, m_i, g), \
           lambda : lower_vertices(q_b, m, g), \
           lambda : inner_points_matrix(q_t, m)

def upper_vertices(q: NDArray[np.float64], m: NDArray[np.bool_], f: Callable) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    
    first_vertices = np.argmax(np.where(m, -np.inf, q), axis=1)

    second_vertices = np.argmax(np.where(f(m, first_vertices), -np.inf, q), axis=1)

    return first_vertices, second_vertices

def lower_vertices(q: NDArray[np.float64], m: NDArray[np.bool_], f: Callable) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    
    first_vertices = np.argmin(np.where(m, np.inf, q), axis=1)

    second_vertices = np.argmin(np.where(f(m, first_vertices), np.inf, q), axis=1)

    return first_vertices, second_vertices

def inner_points_matrix(q: NDArray[np.float64], m: NDArray[np.bool_]) -> NDArray[np.float64]:

    return np.where(m, 0, q)

def k_rmsd(k: np.float64, q: NDArray[np.float64], l: NDArray[np.uint64], r: NDArray[np.uint64]) -> NDArray[np.float64]:

    return np.power(np.sqrt(np.divide(np.sum(q**2, axis=1), r-l)), 1/k)

def parabolic_area_enclosed(l_x: NDArray[np.uint64], v_x: NDArray[np.uint64], r_x: NDArray[np.uint64], i_y: NDArray[np.float64], v_y: NDArray[np.float64]) -> NDArray[np.float64]:

    coefficients = poly.fit_polynomial(np.column_stack((l_x, v_x, r_x)), np.column_stack((i_y, v_y, i_y)), 2)

    return ((i_y * r_x) - (i_y * l_x))  - poly.parabolic_area(coefficients, np.column_stack((l_x, r_x)))

def sort_cubic_coordinates(v_x_0: NDArray[np.uint64], v_x_1: NDArray[np.uint64]) -> tuple[Callable, Callable]:

    v_x = np.column_stack((v_x_0, v_x_1))

    v_x_s = np.argsort(v_x, axis=1)

    return lambda : np.take_along_axis(v_x, v_x_s, axis=1), \
           lambda v_0, v_1: np.take_along_axis(np.column_stack((v_0, v_1)), v_x_s, axis=1)

def cubic_area_enclosed(l_x: NDArray[np.uint64], v_x_0: NDArray[np.uint64], v_x_1: NDArray[np.uint64], r_x: NDArray[np.uint64], i_y: NDArray[np.float64], v_y_0: NDArray[np.float64], v_y_1: NDArray[np.float64]) -> NDArray[np.float64]:

    v = sort_cubic_coordinates(v_x_0, v_x_1)

    coefficients = poly.fit_polynomial(np.column_stack((l_x, v[0](), r_x)), np.column_stack((i_y, v[1](v_y_0, v_y_1), i_y)), 3)

    return ((i_y * r_x) - (i_y * l_x))  - poly.cubic_area(coefficients, np.column_stack((l_x, r_x)))

def euclidean_distance(i_x: NDArray[np.uint64], v_x: NDArray[np.uint64], i_y: NDArray[np.float64], v_y: NDArray[np.float64]) -> NDArray[np.float64]:

    return np.sqrt(((v_x - i_x)**2) + ((v_y - i_y)**2))

def slope(i_x: NDArray[np.uint64], v_x: NDArray[np.uint64], i_y: NDArray[np.float64], v_y: NDArray[np.float64]) -> NDArray[np.float64]:

    return np.abs(np.divide((v_y - i_y), (v_x - i_x)))

def percentage_diff(i_x: NDArray[np.uint64], v_x: NDArray[np.uint64]) -> NDArray[np.float64]:

    return np.abs(100*np.divide((v_x - i_x), i_x))

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

def barrier_points(q: NDArray[np.float64]) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:

    m = (q > 0)

    return left_barrier_points(m), right_barrier_points(m)

def initial_right_points_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return right_mask(q > 0)

def next_right_points_mask(columns: NDArray[np.uint64], m: NDArray[np.bool_], r: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return m & (columns != r[:,None])
    
def right_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return first_points(m)

def initial_validate(c: np.uint64, l: NDArray[np.uint64], r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b) & ((r - l) > c)

def next_validate(r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b)


df 
df[top]
df[bottom]
r_mask_0 = 

def recursive_fun():

    