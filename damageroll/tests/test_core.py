import math

import pytest

import damageroll

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _d():
    return damageroll.Damage(100.0)

CASES = [
    ("no_armor", lambda: abs(_d().mitigate(100.0, 0.0, 0.0) - 100.0) < 1e-9),
    ("armor_less", lambda: _d().mitigate(100.0, 100.0, 0.0) < 100.0),
    ("diminish", lambda: (_d().mitigate(100.0, 100.0, 0.0) - _d().mitigate(100.0, 200.0, 0.0)) <= (_d().mitigate(100.0, 0.0, 0.0) - _d().mitigate(100.0, 100.0, 0.0)) + 1e-9),
    ("resist", lambda: _d().mitigate(100.0, 0.0, 0.5) < 100.0),
    ("positive", lambda: _d().mitigate(100.0, 1000.0, 0.9) > 0.0),
    ("expected", lambda: abs(_d().expected(100.0, 0.0, 2.0, 0.0, 0.0) - 100.0) < 1e-9),
    ("bad_k", lambda: _raises(lambda: damageroll.Damage(0.0))),
    ("bad_armor", lambda: _raises(lambda: _d().mitigate(100.0, -1.0, 0.0))),
    ("bad_resist", lambda: _raises(lambda: _d().mitigate(100.0, 0.0, 1.0))),
    ("type", lambda: isinstance(_d().mitigate(100.0, 50.0, 0.1), float)),
    ("det", lambda: _d().mitigate(100.0, 50.0, 0.1) == _d().mitigate(100.0, 50.0, 0.1)),
    ("nan", lambda: _raises(lambda: _d().mitigate(float('nan'), 0.0, 0.0))),
    ("exp_type", lambda: isinstance(_d().expected(100.0, 0.5, 2.0, 50.0, 0.0), float)),
    ("no_mut", lambda: _d().mitigate(100.0, 10.0, 0.1) is not None),
    ("crit_effect", lambda: _d().expected(100.0, 1.0, 2.0, 0.0, 0.0) > _d().expected(100.0, 0.0, 2.0, 0.0, 0.0)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
