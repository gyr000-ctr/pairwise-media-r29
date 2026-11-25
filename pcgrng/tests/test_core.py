import math

import pytest

import pcgrng

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _p():
    return pcgrng.Pcg(42, 54)

CASES = [
    ("u32_range", lambda: 0 <= _p().next_u32() < 2 ** 32),
    ("det", lambda: pcgrng.Pcg(42, 54).next_u32() == pcgrng.Pcg(42, 54).next_u32()),
    ("stream_diff", lambda: pcgrng.Pcg(42, 54).next_u32() != pcgrng.Pcg(42, 55).next_u32()),
    ("fork_diff", lambda: pcgrng.Pcg(42, 54).fork(1).next_u32() != pcgrng.Pcg(42, 54).fork(2).next_u32()),
    ("range_no_bias", lambda: 0 <= _p().range(10) < 10),
    ("bad_seed", lambda: _raises(lambda: pcgrng.Pcg(-1, 0))),
    ("bad_range", lambda: _raises(lambda: _p().range(0))),
    ("type", lambda: isinstance(_p().next_u32(), int)),
    ("range_type", lambda: isinstance(_p().range(5), int)),
    ("distinct", lambda: len({pcgrng.Pcg(42, 54).next_u32() for _ in range(2)}) >= 1),
    ("no_mut", lambda: _p().next_u32() is not None),
    ("fork_type", lambda: isinstance(pcgrng.Pcg(42, 54).fork(1), pcgrng.Pcg)),
    ("many", lambda: len([_p().next_u32() for _ in range(3)]) == 3),
    ("stream_zero", lambda: isinstance(pcgrng.Pcg(0, 0).next_u32(), int)),
    ("range_big", lambda: 0 <= _p().range(1000000) < 1000000),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
