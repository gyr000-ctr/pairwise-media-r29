import math
import random
import time

import pytest

import damageroll


def test_armor_matches_analytic():
    d = damageroll.Damage(100.0)
    for armor in (0.0, 25.0, 100.0, 333.0, 10000.0):
        expect = 250.0 * 100.0 / (100.0 + armor)
        assert abs(d.mitigate(250.0, armor, 0.0) - expect) <= 1e-9


def test_armor_marginal_diminishing():
    d = damageroll.Damage(50.0)
    prev = None
    for a in range(0, 2000, 10):
        red = 1.0 - d.mitigate(100.0, float(a), 0.0) / 100.0
        if prev is not None:
            assert red - prev[1] <= prev[0] + 1e-12
        prev = (red - prev[1] if prev else float("inf"), red) if prev else (float("inf"), red)


def test_penetration_then_multiplicative_resist():
    d = damageroll.Damage(100.0)
    # 穿透后每段抗性 max(0, r - pen)，再乘法叠加
    got = d.mitigate(100.0, 0.0, [0.5, 0.4], penetration=0.2)
    expect = 100.0 * (1 - 0.3) * (1 - 0.2)
    assert abs(got - expect) <= 1e-9
    # 穿透溢出时抗性归零，不为负
    assert abs(d.mitigate(100.0, 0.0, 0.3, penetration=0.9) - 100.0) <= 1e-9


def test_damage_floor_positive():
    d = damageroll.Damage(10.0)
    assert d.mitigate(1.0, 1e9, [0.9, 0.9, 0.9]) > 0.0


def test_crit_moments_vs_monte_carlo():
    d = damageroll.Damage(100.0)
    args = (100.0, 0.3, 2.5, 80.0, 0.2, 0.1)
    exp = d.expected(*args)
    var = d.variance(*args)
    rng = random.Random(20261126)
    n = 200_000
    samples = [d.sample(*args, rng=rng) for _ in range(n)]
    mean = sum(samples) / n
    mvar = sum((s - mean) ** 2 for s in samples) / n
    assert abs(mean - exp) / exp <= 0.01
    assert abs(mvar - var) / var <= 0.01


def test_order_observable():
    d = damageroll.Damage(100.0)
    steps = d.explain(100.0, 50.0, 0.4, penetration=0.1)
    assert [s[0] for s in steps] == ["input", "penetration", "resist", "armor"]
    assert steps[-1][1] == d.mitigate(100.0, 50.0, 0.4, penetration=0.1)


def test_performance_10k():
    d = damageroll.Damage(100.0)
    t0 = time.perf_counter()
    for i in range(10_000):
        d.mitigate(100.0, float(i % 500), 0.3, penetration=0.05)
    assert time.perf_counter() - t0 < 0.2


def test_distinguishable_errors():
    d = damageroll.Damage(100.0)
    cases = [
        (lambda: damageroll.Damage(0.0), "armor_k"),
        (lambda: damageroll.Damage(float("inf")), "armor_k"),
        (lambda: d.mitigate(-1.0, 0.0, 0.0), "dmg"),
        (lambda: d.mitigate(1.0, -1.0, 0.0), "armor"),
        (lambda: d.mitigate(1.0, 0.0, 1.0), "resist"),
        (lambda: d.mitigate(1.0, 0.0, 0.0, penetration=-0.1), "penetration"),
        (lambda: d.expected(1.0, 1.5, 2.0, 0.0, 0.0), "crit"),
        (lambda: d.expected(1.0, 0.5, 0.5, 0.0, 0.0), "cdmg"),
        (lambda: d.mitigate(float("nan"), 0.0, 0.0), "有限值"),
    ]
    msgs = []
    for fn, key in cases:
        with pytest.raises((ValueError, TypeError)) as ei:
            fn()
        msgs.append(str(ei.value))
        assert key in str(ei.value)
    assert len(set(msgs)) == len(msgs)  # 报错信息两两可区分


def test_type_error_for_non_numeric():
    d = damageroll.Damage(100.0)
    with pytest.raises(TypeError):
        d.mitigate("100", 0.0, 0.0)
    with pytest.raises(TypeError):
        damageroll.Damage("100")


def test_concurrent_use():
    import concurrent.futures

    d = damageroll.Damage(100.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        vals = list(ex.map(lambda i: d.mitigate(100.0, float(i), 0.2), range(1000)))
    assert all(vals[i] == d.mitigate(100.0, float(i), 0.2) for i in range(1000))
