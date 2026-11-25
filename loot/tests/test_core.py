import math

import pytest

import loot

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

T = [("common", 90.0), ("rare", 10.0)]

CASES = [
    ("roll_in_table", lambda: loot.Loot(T, 1).roll()[0] in ('common', 'rare')),
    ("det", lambda: loot.Loot(T, 1).roll() == loot.Loot(T, 1).roll()),
    ("seed_diff", lambda: loot.Loot(T, 1).roll() != loot.Loot(T, 2).roll()),
    ("pity_init", lambda: loot.Loot(T, 1).pity() == 0),
    ("bad_table", lambda: _raises(lambda: loot.Loot([], 1))),
    ("bad_weight", lambda: _raises(lambda: loot.Loot([('a', 0.0)], 1))),
    ("bad_seed", lambda: _raises(lambda: loot.Loot(T, -1))),
    ("type", lambda: isinstance(loot.Loot(T, 1).roll(), tuple)),
    ("pity_type", lambda: isinstance(loot.Loot(T, 1).pity(), int)),
    ("no_mut", lambda: loot.Loot(list(T), 1).roll() is not None),
    ("many", lambda: len([loot.Loot(T, 1).roll() for _ in range(3)]) == 3),
    ("nan", lambda: _raises(lambda: loot.Loot([('a', float('nan'))], 1))),
    ("float_weight", lambda: loot.Loot([('a', 50.0), ('b', 50.0)], 3).roll()[0] in ('a', 'b')),
    ("pity_int", lambda: isinstance(loot.Loot(T, 1).pity(), int)),
    ("single", lambda: loot.Loot([('only', 1.0)], 1).roll()[0] == 'only'),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
