import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray

def upper_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.triu(np.ones(s, dtype=bool), k=1)

def lower_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:

    return np.tril(np.ones(s, dtype=bool), k=-1)

def left_mask(r: NDArray[np.float64]) -> NDArray[np.bool_]:

    return (r > 0) & lower_tri_mask(r.shape)

def right_barrier(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_point = lambda p: np.where(p == 0, len(p), p)

    return default_point(np.argmax(m, axis=1))

def right_mask(r: NDArray[np.float64]) -> NDArray[np.bool_]:

    return (r > 0) & upper_tri_mask(r.shape)

def left_point(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_point = lambda p: np.where(p == len(p)-1, -1, p)

    return default_point(m.shape[1] - np.argmax(m[:,::-1], axis=1) - 1)

def right_point(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    return np.argmax(m, axis=1)

def validate(l: NDArray[np.uint64], r: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.bool_]:

    return (r <= b) & ((r - l) > 3)