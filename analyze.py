import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray

def upper_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.triu(np.ones(s, dtype=bool), k=1)

def lower_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.tril(np.ones(s, dtype=bool), k=-1)

def 

def left_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return (q > 0) & lower_tri_mask(q.shape)

def right_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return (q > 0) & upper_tri_mask(q.shape)





def validate(i: np.uint64, l: NDArray[np.uint64], r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b) & ((r - l) > i)

def barrier_mask(columns: NDArray[np.uint64], l: NDArray[np.uint64], r: NDArray[np.uint64]) -> tuple[Callable, Callable]:

    return lambda : (columns <= l[:,None]) | (columns >= r[:,None]), \
           lambda m, i : m | (columns == i[:,None])

def upper_vertices(q: NDArray[np.float64], f: Callable[[NDArray[np.float64], np.float64], NDArray[np.float64]]) -> NDArray[np.uint64]:
    
    return lambda : np.argmax(np.where(m, -np.inf, q), axis=1), \
           lambda v : np.argmax(np.where(f(m, v), -np.inf, q), axis=1)

def lower_vertices(q: NDArray[np.float64], f: Callable[[NDArray[np.float64], np.float64], NDArray[np.float64]]) -> NDArray[np.uint64]:

    return lambda : np.argmin(np.where(m, np.inf, q), axis=1), \
           lambda v : np.argmin(np.where(f(m, v), np.inf, q), axis=1)
    
def vertices():

    m = f[0]()

    return lambda lower_vertices(m, f[1]),
           lambda upper_vertices(f[1](filteredresisanceindices), f[1])


v=call vertices
upper_vertices = v[1]()
first_upper = upper_vertices[0]()
second_upper  = upper_vertices[1](first_upper)



def barrier_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_point = lambda p: np.where(p == 0, len(p), p)

    return default_point(np.argmax(m, axis=1))

def left_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_point = lambda p: np.where(p == len(p)-1, -1, p)

    return default_point(m.shape[1] - np.argmax(m[:,::-1], axis=1) - 1)

def right_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return np.argmax(m, axis=1)

def above_resistance_line_mask(q: NDArray[np.float64]) -> Callable[[Callable], NDArray[np.bool_]]:

    m = (q > 0)

    return lambda d:  m & d(m.shape)

def points_above_resistance_line(q: NDArray[np.float64]):

    f = above_resistance_line_mask(q)

    return lambda t: t(f())

def recursive_fun():

    m = right_points_mask()

    r = first_right_points(m)

    l = first_left_points()

    b = barrier_points()

    v = validate(r,l,b)

    d = data(r[v], l[v], b[v])

    n = new_mask(m[v], r[v])