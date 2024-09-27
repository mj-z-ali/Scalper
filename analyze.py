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

def percentage_diff(i_y: NDArray[np.uint64], v_y: NDArray[np.uint64]) -> NDArray[np.float64]:

    return 100*np.divide(np.abs(v_y - i_y), i_y)

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

def next_right_points_mask(columns: NDArray[np.uint64], m: NDArray[np.bool_], r: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return m & (columns != r[:,None])

def right_points_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return right_mask(q > 0)

def right_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return first_points(m)

def initial_validate(c: np.uint64, lb: NDArray[np.uint64], rb: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (rb - lb) > c

def next_validate(r: NDArray[np.uint64], rb: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= rb)

def diff_matrix(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:

    return a - b.reshape(-1,1)

def initial_points(df: pd.DataFrame):

    top = df['top'].values
    bottom = df['bottom'].values
    high = df['high'].values
    
    q_t = diff_matrix(top, top)
    q_b = diff_matrix(top, bottom)
    q_h = diff_matrix(top, high)
    i = np.arange(len(top))

    lb,rb = barrier_points(q_t)
    r_m = right_points_mask(q_h)
    r = right_points(r_m)

    vld = initial_validate(3, lb, rb)

    return lb[vld], r[vld], rb[vld], r_m[vld], i[vld], q_t[vld], q_b[vld]


def recursive_fun(columns,lb,r,rb,r_m,i,q_t,q_b):

    vld = next_validate(r,rb)

    mask_funs = set_mask_coordinates(columns)

    p_data = preliminary_data(q_t[vld],q_b[vld],lb[vld],i[vld],r[vld],mask_funs[0],mask_funs[1])

    uv_x_0, uv_x_1 = p_data[0]()

    lv_x_0, lv_x_1 = p_data[1]()

    uv_y_0, uv_y_1 = df['top'].values[uv_x_0], df['top'].values[uv_x_1]

    lv_y_0, lv_y_1 = df['bottom'].values[lv_x_0], df['bottom'].values[lv_y_1]

    i_y = df['top'].values[i[vld]]

    k_rmsd(2, p_data[2](),lb,r)

    parabolic_area_enclosed(lb,uv_x_0,r,i_y,uv_y_0)

    parabolic_area_enclosed(lb,uv_x_1,r,i_y,uv_y_1)

    parabolic_area_enclosed(lb,lv_x_0,r,i_y,lv_y_0)

    parabolic_area_enclosed(lb,lv_x_1,r,i_y,lv_y_1)

    cubic_area_enclosed(lb, uv_x_0, uv_x_1, r, i_y, uv_y_0, uv_y_1) 

    euclidean_distance(i, uv_x_0, i_y, uv_y_0)

    euclidean_distance(i, uv_x_1, i_y, uv_y_1)

    slope(i, uv_x_0, i_y, uv_y_0)

    slope(i, uv_x_1, i_y, uv_y_1)

    percentage_diff(i_y, uv_y_0)

    percentage_diff(i_y, uv_y_1)

    new_r_m = mask_funs[2](r_m[vld], r[vld])

    new_r = right_points(new_r_m)

    recursive_fun(columns, lb[vld], new_r, rb[vld], new_r_m, i[vld], q_t[vld], q_b[vld])