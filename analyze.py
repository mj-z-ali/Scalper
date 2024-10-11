import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import polynomial as poly
from functools import reduce


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

    vld = (p('r_x') - p('lb_x')) > c

    data = {
        'lb_x': p('lb_x')[vld],
        'rb_x': p('rb_x')[vld],
        'r_x': p('r_x')[vld],
        'i_x': p('i_x')[vld],                      
        'columns': p('i_x'),
        'q_t': q('q_t')[vld],
        'q_b': q('q_b')[vld],
        'r_mask': r('r_mask')[vld]
    }

    return lambda s: data[s]


def next_validated_data(f: Callable, g: Callable) -> Callable:

    vld = (g('r_x') <= f('rb_x')) & (g('r_x') != -1)
    print(vld)
    data = {
        'empty': np.all(~vld),
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

    r_mask = f('r_mask') & (f('columns') != f('r_x')[:,None])

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
        'i_y': f('top_p')[g('i_x')],
        'start_time_y': f('time')[g('r_x')],
        'end_time_y': f('time')[g('r_x')+1]
    }

    return lambda s: data[s]

def parameter(f: Callable, g: Callable) -> Callable:

    data = {
        'x_parameter_package': lambda op: op(f),
        'x_y_parameter_package': lambda op: op(f,g),
        'y_parameter_package': lambda op: op(g)
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
    
    default_points = lambda p: np.where(p == 0, -1, p)

    return default_points(first_points(m))


def ivd_params(f_df: Callable) -> Callable:

    f_rm = relational_matrices(f_df)

    f_pm = point_masks(f_rm)

    f_p = points(f_pm, f_df)

    return lambda f_ivd: f_ivd(f_rm, f_pm, f_p)

def init_validated_data(c: np.uint64, f_df: Callable) -> Callable:

    f_ivd = ivd_params(f_df)(lambda f_rm, f_pm, f_p: initial_validated_data(c, f_p, f_rm, f_pm))

    return next_validated_data(f_ivd, f_ivd)


def preliminary_data(f_bf: Callable) -> Callable:

    f_pd = lambda f_px: (f_px, preliminary_data_y(f_bf, f_px))

    return lambda f_vd: f_pd(preliminary_data_x(f_vd, boundary_mask(f_vd)))

def operations(*args: Callable) -> Callable:

    return lambda f, l: reduce(lambda acc, x: np.column_stack((acc, x)), map(lambda g: g(f), *args), np.empty((l,0)))

def build_resistance_data(f_vd: Callable, f_pd: Callable, f_op: Callable, data: NDArray[np.float64]):

    if f_vd('empty'):
        return data
    
    f_px, f_py = f_pd(f_vd)

    new_data = np.concatenate((data, f_op(parameter(f_px, f_py), f_vd('i_x').shape[0])))

    return build_resistance_data(next_validated_data(f_vd, variable_data(f_vd)), f_pd, f_op, new_data)

def resistance_data(day: np.uint64, min_line_width: np.uint64, f_bf: Callable, *args: Callable):

    data = build_resistance_data(init_validated_data(min_line_width, f_bf), preliminary_data(f_bf), operations(args), np.empty((0, len(args))))

    return np.column_stack((np.array([day]*data.shape[0]), data))

def f_vt(f_tf: Callable, g: Callable, r: range):

    # First breakout trade indices
    bt_x = first_points((f_tf('time') >= g('start_time_y')[:,None]) & (f_tf('time') < g('end_time_y')[:,None]) & (f_tf('price') > g('i_y')[:,None]))
    
    si = reduce(lambda acc, x: acc + [acc[0] - np.timedelta64(x,'s')], r, [f_tf('time')[bt_x]])

    return lambda f_vy: poly.fit_polynomial(np.array([[0,1,2]]*len(g('i_y'))), f_vy(si), 2)

def volume_time(f_tf: Callable, g: Callable, r: range):

    f_vi = lambda si, i: np.sum(((f_tf('time') >= si[i][:,None]) & (f_tf('time') < si[i-1][:,None])) * f_tf('size'), axis=1)
    
    f_vy = lambda si: np.column_stack((list(map(lambda i: f_vi(si, i), r))))

    return f_vt(f_tf, g, r)(f_vy)

def acceleration(f_tf: Callable) -> Callable:

    f_ddvt = lambda vt: vt[:, 2]*2

    f_a = lambda g: f_ddvt(volume_time(f_tf, g, range(3,0,-1)))

    return lambda f: f('y_parameter_package')(f_a)

def k_rmsd(k: np.float64) -> Callable:

    f_krmsd = lambda f: np.power(np.sqrt(np.divide(np.sum(f('ipm')**2, axis=1), f('r_x')-f('lb_x'))), 1/k)

    return lambda f: f('x_parameter_package')(f_krmsd)

def parabolic_area_enclosed(f: Callable, g: Callable, i: np.uint64, d: np.char) -> NDArray[np.float64]:

    coefficients = poly.fit_polynomial(np.column_stack((f('lb_x'), f(f'{d}v_x_{i}'), f('r_x'))), np.column_stack((g('i_y'), g(f'{d}v_y_{i}'), g('i_y'))), 2)

    return ((g('i_y') * f('r_x')) - (g('i_y') * f('lb_x')))  - poly.parabolic_area(coefficients, np.column_stack((f('lb_x'), f('r_x'))))

def first_upper_parabolic_area_enclosed() -> Callable:
    
    f_parab = lambda f, g: parabolic_area_enclosed(f,g,0,'u')

    return lambda f: f('x_y_parameter_package')(f_parab)

def second_upper_parabolic_area_enclosed() -> Callable:
        
    f_parab = lambda f, g: parabolic_area_enclosed(f,g,1,'u')

    return lambda f: f('x_y_parameter_package')(f_parab)

def first_lower_parabolic_area_enclosed() -> Callable:
        
    f_parab = lambda f, g: parabolic_area_enclosed(f,g,0,'l')

    return lambda f: f('x_y_parameter_package')(f_parab)
    
def second_lower_parabolic_area_enclosed() -> Callable:
        
    f_parab = lambda f, g: parabolic_area_enclosed(f,g,1,'l')

    return lambda f: f('x_y_parameter_package')(f_parab)

def cubic_area_enclosed() -> Callable:

    f_coeff = lambda f,g: poly.fit_polynomial(np.column_stack((f('lb_x'), f('uv_x_0'), f('uv_x_1'), f('r_x'))), np.column_stack((g('i_y'), g('uv_y_0'), g('uv_y_1'), g('i_y'))), 3)
    f_cubic = lambda f,g: ((g('i_y') * f('r_x')) - (g('i_y') * f('lb_x')))  - poly.cubic_area(f_coeff(f,g), np.column_stack((f('lb_x'), f('r_x'))))

    return lambda f: f('x_y_parameter_package')(f_cubic)

def euclidean_distance(f: Callable, g: Callable, i: np.uint64) -> NDArray[np.float64]:

    return np.sqrt(((f(f'uv_x_{i}') - f('i_x'))**2) + ((g(f'uv_y_{i}') - g('i_y'))**2))

def first_euclidean_distance() -> Callable:
    
    f_eucl = lambda f,g: euclidean_distance(f,g,0)

    return lambda f: f('x_y_parameter_package')(f_eucl)

def second_euclidean_distance() -> Callable:
    
    f_eucl = lambda f,g: euclidean_distance(f,g,1)

    return lambda f: f('x_y_parameter_package')(f_eucl)

def slope(f: Callable, g: Callable, i: np.uint64) -> NDArray[np.float64]:

    return np.abs(np.divide((g(f'uv_y_{i}') - g('i_y')), (f(f'uv_x_{i}') - f('i_x'))))

def first_slope() -> Callable:
    
    f_slope = lambda f,g: slope(f,g,0)

    return lambda f: f('x_y_parameter_package')(f_slope)

def second_slope() -> Callable:
    
    f_slope = lambda f,g: slope(f,g,1)

    return lambda f: f('x_y_parameter_package')(f_slope)

def percentage_diff(g: Callable, i: np.uint64) -> NDArray[np.float64]:

    return 100*np.divide(np.abs(g(f'uv_y_{i}') - g('i_y')), g('i_y'))

def first_percentage_diff() -> Callable:
    
    f_perc = lambda g: percentage_diff(g, 0)

    return lambda f: f('y_parameter_package')(f_perc)

def second_percentage_diff() -> Callable:
    
    f_perc = lambda g: percentage_diff(g, 1)

    return lambda f: f('y_parameter_package')(f_perc)
