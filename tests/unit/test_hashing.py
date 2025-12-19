from src.utils.hashing import hash_categorical, hash_feature_cross
import pandas as pd


def test_hash_categorical_shape():
    series = pd.Series(["A", "B", "C"])
    mat = hash_categorical(series, 16)
    assert mat.shape == (3, 16)


def test_hash_feature_cross_nonempty():
    a = pd.Series(["A", "B"])
    b = pd.Series(["1", "2"])
    mat = hash_feature_cross(a, b, 8)
    assert mat.nnz == 2
