import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray

def upper_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.triu(np.ones(s, dtype=bool), k=1)

def lower_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.tril(np.ones(s, dtype=bool), k=-1)

def left_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return (q > 0) & lower_tri_mask(q.shape)

def right_barrier(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_point = lambda p: np.where(p == 0, len(p), p)

    return default_point(np.argmax(m, axis=1))

def right_mask(q: NDArray[np.float64]) -> NDArray[np.bool_]:

    return (q > 0) & upper_tri_mask(q.shape)

def left_point(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_point = lambda p: np.where(p == len(p)-1, -1, p)

    return default_point(m.shape[1] - np.argmax(m[:,::-1], axis=1) - 1)

def right_point(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return np.argmax(m, axis=1)

def validate(i: np.uint64, l: NDArray[np.uint64], r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b) & ((r - l) > i)

def infinity_barrier(columns: NDArray[np.uint64], l: NDArray[np.uint64], r: NDArray[np.uint64]) -> tuple[Callable, Callable]:

    m = (columns <= l[:,None]) | (columns >= r[:,None])

    e = m | np.eye(len(columns), dtype=bool)

    return lambda q, i : np.where(m, i, q), \
           lambda q, i : np.where(e, i, q)

def lower_vertex(q: NDArray[np.float64], f: Callable[[NDArray[np.float64], np.float64], NDArray[np.float64]]) -> NDArray[np.uint64]:

    return np.argmin(f(q, np.inf), axis=1)

def upper_vertex(q: NDArray[np.float64], f: Callable[[NDArray[np.float64], np.float64], NDArray[np.float64]]) -> NDArray[np.uint64]:

    return np.argmax(f(q, -np.inf), axis=1)