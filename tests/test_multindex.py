"""Sanity checks for multi-index utilities."""

import sys
from pathlib import Path
import numpy as np

# Make the package importable
_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from greedyopm.polynomials.multindex import (
    _normalize_degrees,
    generate_multi_indices,
    graded_lex_multi_indices,
    graded_lex_multi_indices_for_degrees,
    new_multi_indices_on_coord_add,
    compute_parent_info,
    compute_parent_info_for_new,
)


def test_normalize_degrees():
    assert _normalize_degrees(2) == [2]
    assert _normalize_degrees(3) == [2, 3]
    assert _normalize_degrees([3]) == [3]
    assert _normalize_degrees([3, 2]) == [2, 3]
    try:
        _normalize_degrees([1])
        assert False, "should have raised"
    except ValueError:
        pass


def test_generate_multi_indices():
    # r = 2, q = 2: (2,0), (1,1), (0,2)
    out = generate_multi_indices(2, 2)
    expected = np.array([[2, 0], [1, 1], [0, 2]], dtype=np.int32)
    assert np.array_equal(out, expected), out


def test_graded_lex_multi_indices():
    # r = 2, p = 2, min 0: [(0,0)], [(1,0), (0,1)], [(2,0), (1,1), (0,2)]
    out = graded_lex_multi_indices(2, 2, min_degree=0)
    expected = np.array([
        [0, 0],
        [1, 0], [0, 1],
        [2, 0], [1, 1], [0, 2],
    ], dtype=np.int32)
    assert np.array_equal(out, expected), out


def test_new_multi_indices():
    # Adding the 3rd coordinate with degrees=[2].
    # Expected new features: alpha = (0,0,1) (linear), and the degree-2
    # multi-indices with alpha_3 >= 1: (1,0,1), (0,1,1), (0,0,2).
    new = new_multi_indices_on_coord_add(3, [2], include_new_linear=True)
    assert new.shape == (4, 3), new.shape
    expected_set = {(0,0,1), (1,0,1), (0,1,1), (0,0,2)}
    actual_set = {tuple(row) for row in new}
    assert actual_set == expected_set, actual_set


def test_parent_info():
    mi = graded_lex_multi_indices(2, 2, min_degree=0)
    ell_arr, parent_arr = compute_parent_info(mi)
    # row 0: (0,0) -> parent_arr[0] = -1
    assert parent_arr[0] == -1 and ell_arr[0] == -1
    # row 1: (1,0) -> last nonzero = coord 0, beta = (0,0) at row 0
    assert parent_arr[1] == 0 and ell_arr[1] == 0
    # row 2: (0,1) -> last nonzero = coord 1, beta = (0,0) at row 0
    assert parent_arr[2] == 0 and ell_arr[2] == 1
    # row 3: (2,0) -> beta = (1,0) at row 1, ell = 0
    assert parent_arr[3] == 1 and ell_arr[3] == 0
    # row 4: (1,1) -> last nonzero = coord 1, beta = (1,0) at row 1
    assert parent_arr[4] == 1 and ell_arr[4] == 1
    # row 5: (0,2) -> last nonzero = coord 1, beta = (0,1) at row 2
    assert parent_arr[5] == 2 and ell_arr[5] == 1


def test_parent_info_for_new():
    # old multi-indices for r=2, p=2 (padded to 3 dims)
    old_mi = graded_lex_multi_indices(2, 2, min_degree=0)  # (6, 2)
    old_mi_padded = np.hstack([old_mi, np.zeros((6, 1), dtype=np.int32)])  # (6, 3)
    new_mi = new_multi_indices_on_coord_add(3, [2], include_new_linear=True)  # (4, 3)

    parent_idx = compute_parent_info_for_new(
        old_mi_padded, new_mi, new_coord=2,
    )
    # combined buffer:
    #   0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (2,0,0), 4: (1,1,0), 5: (0,2,0)  [old]
    #   6: (0,0,1), 7: (1,0,1), 8: (0,1,1), 9: (0,0,2)                          [new]
    # parent of (0,0,1) = (0,0,0) = row 0
    # parent of (1,0,1) = (1,0,0) = row 1
    # parent of (0,1,1) = (0,1,0) = row 2
    # parent of (0,0,2) = (0,0,1) = row 6 (in combined buffer)
    expected = np.array([0, 1, 2, 6], dtype=np.int32)
    assert np.array_equal(parent_idx, expected), parent_idx


if __name__ == "__main__":
    test_normalize_degrees()
    test_generate_multi_indices()
    test_graded_lex_multi_indices()
    test_new_multi_indices()
    test_parent_info()
    test_parent_info_for_new()
    print("All multi-index tests passed.")