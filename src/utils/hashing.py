"""Feature hashing utilities for high-cardinality categorical features and crosses."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from typing import List


def _hash_value(value: str, space: int) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest, 16) % space


def hash_categorical(series: pd.Series, space: int) -> csr_matrix:
    indices = series.apply(lambda x: _hash_value(x, space)).to_numpy()
    data = np.ones(len(series))
    indptr = np.arange(len(series) + 1)
    return csr_matrix((data, indices, indptr), shape=(len(series), space))


def hash_feature_cross(series_a: pd.Series, series_b: pd.Series, space: int) -> csr_matrix:
    crossed = series_a.astype(str) + "_x_" + series_b.astype(str)
    return hash_categorical(crossed, space)


def stack_sparse(matrices: List[csr_matrix]) -> csr_matrix:
    return hstack(matrices).tocsr()
