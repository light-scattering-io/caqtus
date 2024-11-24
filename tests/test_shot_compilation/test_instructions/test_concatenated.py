from typing import assert_type

import numpy as np

from caqtus.instructions import pattern, Concatenated, Pattern, Empty


def test_bool_concatenated():
    p1 = pattern([True, False, True])
    assert not isinstance(p1, Empty)
    p2 = pattern([True, False])
    assert not isinstance(p2, Empty)
    c = Concatenated(p1, p2)
    assert c.dtype() == np.dtype(np.bool)
    assert_type(c, Concatenated[Pattern[np.bool], Pattern[np.bool]])


def test_get_index():
    p = pattern([1, 2, 3])
    assert not isinstance(p, Empty)
    c = Concatenated(p, p)
    assert c[2] == 3
    assert c[3] == 1
    assert_type(c[0], np.int64)


def test_slice():
    l1 = pattern([1, 2, 3])
    assert not isinstance(l1, Empty)
    l2 = pattern([4, 5, 6])
    assert not isinstance(l2, Empty)

    c = Concatenated(l1, l2)
    assert_type(c[0:3], Pattern[np.int64])
    assert isinstance(c[0:3], Pattern)
