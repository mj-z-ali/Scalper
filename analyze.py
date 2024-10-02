import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import polynomial as poly



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

    lb,rb = boundary_points(q_t)
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


def data_frame(df: pd.DataFrame) -> Callable:

    data = {
        'range_x': df.index.values,
        'top_p': df['top'].values,
        'bottom_p' : df['bottom'].values,
        'high_p' : df['high'].values
    }

    return lambda s: data[s]

def relational_matrices(f: Callable) -> Callable:

    data = {
        'q_t': diff_matrix(f('top_p'), f('top_p')),
        'q_b': diff_matrix(f('top_p'), f('bottom_p')),
        'q_h': diff_matrix(f('top_p'), f('high_p'))
    }

    return lambda s: data[s]


def point_masks(f: Callable) -> Callable:
    
    q_t_mask, q_h_mask = f('q_t') > 0, f('q_h') > 0

    data = {
        'lb_mask': left_mask(q_t_mask),
        'rb_mask': right_mask(q_t_mask),
        'r_mask': right_mask(q_h_mask)
    }

    return lambda s: data[s]

def points(f: Callable, g: Callable) -> Callable:

    data = {
        'lb_x': left_boundary_points(f('lb_mask')),
        'rb_x': right_boundary_points(f('rb_mask')),
        'r_x': right_points(f('r_mask')),
        'i_x': g('range_x')
    }
    
    return lambda s: data[s]


def initial_validated_data(c: np.uint64, p: Callable, q: Callable, r: Callable) -> Callable:

    vld = (p('rb_x') - p('lb_x')) > c

    data = {
        'lb_x': p('lb_x')[vld],
        'rb_x': p('rb_x')[vld],
        'r_x': p('r_x')[vld],
        'i_x': p('range_x')[vld],                      
        'columns': p('range_x'),
        'q_t': q('q_t')[vld],
        'q_b': q('q_b')[vld],
        'r_mask': r('r_mask')[vld]
    }

    return lambda s: data[s]


def next_validated_data(f: Callable, g: Callable) -> Callable:

    vld = g('r_x') <= f('rb_x')

    data = {
        'lb_x': f('lb_x')[vld],
        'rb_x': f('rb_x')[vld],
        'r_x': g('r_x')[vld],
        'i_x': f('i_x')[vld],
        'columns': f('columns'),
        'q_t': f('q_t')[vld],
        'q_b': f('q_b')[vld],
        'r_mask': g('r_mask')[vld]
    }

    return lambda s: data[s]


def variable_data(f: Callable) -> Callable:

    r_mask = f('r_mask') & (f('columns') != f('r_x'))

    data = {
        'r_x': right_points(r_mask),
        'r_mask': r_mask
    }

    return lambda s: data[s]


def boundary_mask(f: Callable) -> Callable:

    b_mask = (f('columns') <= f('lb_x')[:,None]) | (f('columns') >= f('r_x')[:,None])

    data = {
        'b_mask': b_mask,
        'bi_mask': b_mask | (f('columns') == f('i_x')[:,None]),
        'b_op': lambda c: data['b_mask'] | (f('columns') == c[:,None]),
        'bi_op': lambda c: data['bi_mask'] | (f('columns') == c[:,None])
    }

    return lambda s: data[s]

def preliminary_data_x(f: Callable, g: Callable) -> Callable:

    uv_x_0 = np.argmax(np.where(g('bi_mask'), -np.inf, f('q_t')), axis=1)
    lv_x_0 = np.argmin(np.where(g('b_mask'), np.inf, f('q_b')), axis=1)

    data = { 
        'uv_x_0': uv_x_0,
        'uv_x_1': np.argmax(np.where(g('bi_op')(uv_x_0), -np.inf, f('q_t')), axis=1),
        'lv_x_0': lv_x_0,
        'lv_x_1': np.argmin(np.where(g('b_op')(lv_x_0), np.inf, f('q_b')), axis=1),
        'ipm': np.where(g('b_mask'), 0, f('q_t')),
        'i_x': f('i_x'),
        'lb_x': f('lb_x'),
        'r_x': f('r_x')
    }

    return lambda s: data[s]

def preliminary_data_y(f: Callable, g: Callable) -> Callable:

    data = {
        'uv_y_0': f('top_p')[g('uv_x_0')],
        'uv_y_1': f('top_p')[g('uv_x_1')],
        'lv_y_0': f('bottom_p')[g('lv_x_0')],
        'lv_y_1': f('bottom_p')[g('lv_x_1')],
        'i_y': f('top_p')[g('i_x')]
    }

    return lambda s: data[s]

def resistance_data(f: Callable, g: Callable) -> Callable:

    parab = lambda d,i: parabolic_area_enclosed(f('lb_x'), f(f'{d}v_x_{i}'), f('r_x'), g('i_y'), g(f'{d}v_y_{i}'))
    eucl_slop = lambda fun,i: fun(f('i_x'), f(f'uv_x_{i}'), g('i_y'), g(f'uv_y_{i}'))
    perc_diff = lambda i: percentage_diff(g('i_y'), g(f'uv_y_{i}'))
    
    data = {
        'k_rmsd': lambda k: k_rmsd(k, f('ipm'), f('lb_x'), f('r_x')),
        'upper_parabolic_area_enclosed_0': lambda : parab('u', 0),
        'upper_parabolic_area_enclosed_1': lambda : parab('u', 1),
        'lower_parabolic_area_enclosed_0': lambda : parab('l', 0),
        'lower_parabolic_area_enclosed_1': lambda : parab('l', 1),
        'cubic_area_enclosed': lambda : cubic_area_enclosed(f('lb_x'), f('uv_x_0'), f('uv_x_1'), f('r_x'), g('i_y'), g('uv_y_0'), g('uv_y_1')),
        'euclidean_distance_0': lambda : eucl_slop(euclidean_distance,0),
        'euclidean_distance_1': lambda : eucl_slop(euclidean_distance,1),
        'slope_0': lambda : eucl_slop(slope,0),
        'slope_1': lambda : eucl_slop(slope,1),
        'percentage_diff_0': lambda : perc_diff(0),
        'percentage_diff_1': lambda : perc_diff(1)
    }
    
    return lambda s: data[s]


def last_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return m.shape[1] - np.argmax(m[:,::-1], axis=1) - 1

def first_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return np.argmax(m, axis=1)

def right_boundary_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_points = lambda p: np.where(p == 0, len(p), p)

    return default_points(first_points(m))

def left_boundary_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_points = lambda p: np.where(p == len(p)-1, -1, p)

    return default_points(last_points(m))

def right_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return first_points(m)


def operations_dependent_on_dataframe(df: pd.DataFrame) -> Callable:

    f_df = data_frame(df)

    data = {
        'relational_matrices': relational_matrices(f_df),
        'points': lambda f_pm: points(f_pm, f_df),
        'preliminary_data_y': lambda f_px: preliminary_data_y(f_df, f_px)
    }
    
    return lambda s: data[s]