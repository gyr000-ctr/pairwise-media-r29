import math

import pytest

import skillcd

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return skillcd.SkillCd(3, 1.0, 0.0)

CASES = [
    ("initial", lambda: _s().charges() == 3),
    ("use", lambda: (_s().try_use(0.0), _s().charges())[1] == 2),
    ("recharge", lambda: (lambda s: (s.try_use(0.0), s.tick(1.0), s.charges())[2] == 3)(_s())),
    ("empty_fail", lambda: (lambda s: [s.try_use(0.0) for _ in range(3)] and s.try_use(0.0) is False)(_s())),
    ("no_overflow", lambda: (lambda s: (s.tick(100.0), s.charges())[1] == 3)(_s())),
    ("bad_charges", lambda: _raises(lambda: skillcd.SkillCd(0, 1.0))),
    ("bad_recharge", lambda: _raises(lambda: skillcd.SkillCd(3, 0.0))),
    ("bad_cdr", lambda: _raises(lambda: skillcd.SkillCd(3, 1.0, 1.0))),
    ("cdr", lambda: skillcd.SkillCd(3, 1.0, 0.5) is not None),
    ("type", lambda: isinstance(_s().charges(), int)),
    ("det", lambda: _s().charges() == _s().charges()),
    ("nan", lambda: _raises(lambda: _s().tick(float('nan')))),
    ("no_mut", lambda: _s().tick(0.1) is not None),
    ("tick_type", lambda: isinstance(_s().try_use(0.0), bool)),
    ("negative_dt", lambda: _raises(lambda: _s().tick(-1.0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
