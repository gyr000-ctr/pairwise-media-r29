import math

import pytest

import orca
from orca import Orca

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = [((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)), ((0.5, 0.0), (0.0, 0.0), (1.0, 0.0))]

CASES = [
    ("pref", lambda: len(Orca().preferred((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))) == 2),
    ("avoid", lambda: len(Orca().avoid(A, 0)) == 2),
    ("alone", lambda: Orca().avoid([((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))], 0) == (1.0, 0.0)),
    ("bad_radius", lambda: _raises(lambda: Orca(0.0, 2.0))),
    ("bad_horizon", lambda: _raises(lambda: Orca(0.5, 0.0))),
    ("bad_index", lambda: _raises(lambda: Orca().avoid(A, 9))),
    ("empty", lambda: _raises(lambda: Orca().avoid([], 0))),
    ("type", lambda: isinstance(Orca().avoid(A, 0), tuple)),
    ("det", lambda: Orca().avoid(A, 0) == Orca().avoid(A, 0)),
    ("bounded", lambda: math.hypot(*Orca().avoid(A, 0)) <= 1.0 + 1e-9),
    ("nan", lambda: _raises(lambda: Orca().preferred((float('nan'), 0.0), (0, 0), (1, 0)))),
    ("no_mut", lambda: Orca().avoid([list(a) for a in A], 0) is not None),
    ("pref_type", lambda: isinstance(Orca().preferred((0.0, 0.0), (0.0, 0.0), (2.0, 0.0)), tuple)),
    ("symmetry", lambda: math.hypot(*Orca().avoid(A, 1)) <= 1.0 + 1e-9),
    ("float", lambda: isinstance(Orca().avoid(A, 0)[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
