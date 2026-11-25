"""damageroll 核心 API：护甲 / 抗性 / 穿透 / 暴击伤害结算。

结算顺序（固定，且可通过 ``Damage.explain`` 观测）::

    1. 穿透 penetration：每段抗性 r_i -> max(0, r_i - penetration)
    2. 抗性 resist：穿透后各段抗性乘法叠加 mult_res = Π(1 - r_i)
    3. 护甲 armor：递减公式 mult_armor = K / (K + armor)

最终伤害 = dmg * mult_res * mult_armor。

性质：
- 护甲为递减公式（非线性抵消），减伤率 armor/(K+armor) 的边际收益单调递减；
- 每段抗性强制 < 1，故 mult_res > 0，伤害下限恒大于 0（dmg > 0 时）；
- 暴击期望 / 方差为解析式，可与蒙特卡洛对照；
- 实例不可变、无共享状态，方法可并发调用。
"""

import math
import random

__all__ = ["Damage"]

_NUM = (int, float)


def _is_num(x):
    return isinstance(x, _NUM) and not isinstance(x, bool)


class Damage:
    """伤害结算器。``armor_k`` 为护甲递减公式常数 K（必须为正有限数）。"""

    __slots__ = ("armor_k",)

    def __init__(self, armor_k):
        if not _is_num(armor_k):
            raise TypeError(
                "armor_k 必须为数值类型，得到 %s" % type(armor_k).__name__
            )
        if not math.isfinite(armor_k):
            raise ValueError("armor_k 必须为有限值，得到 %r" % (armor_k,))
        if armor_k <= 0:
            raise ValueError("armor_k 必须为正数，得到 %r" % (armor_k,))
        self.armor_k = float(armor_k)

    # ------------------------------------------------------------------
    # 参数校验（每类非法参数给出可区分的报错信息）
    # ------------------------------------------------------------------
    @staticmethod
    def _check(name, value, lo=None, hi=None, hi_open=False):
        if not _is_num(value):
            raise TypeError(
                "%s 必须为数值类型，得到 %s" % (name, type(value).__name__)
            )
        if not math.isfinite(value):
            raise ValueError("%s 必须为有限值（非 NaN/Inf），得到 %r" % (name, value))
        if lo is not None and value < lo:
            raise ValueError("%s 必须 >= %r，得到 %r" % (name, lo, value))
        if hi is not None:
            if hi_open and value >= hi:
                raise ValueError("%s 必须 < %r，得到 %r" % (name, hi, value))
            if not hi_open and value > hi:
                raise ValueError("%s 必须 <= %r，得到 %r" % (name, hi, value))
        return float(value)

    @classmethod
    def _resist_list(cls, resist):
        if _is_num(resist):
            resist = (resist,)
        elif isinstance(resist, (list, tuple)):
            if not resist:
                raise ValueError("resist 列表不能为空")
        else:
            raise TypeError(
                "resist 必须为数值或数值序列，得到 %s" % type(resist).__name__
            )
        return [cls._check("resist[%d]" % i, r, lo=0.0, hi=1.0, hi_open=True)
                for i, r in enumerate(resist)]

    # ------------------------------------------------------------------
    # 结算
    # ------------------------------------------------------------------
    def _multiplier(self, armor, resists, penetration):
        """按固定顺序计算总承伤系数：先穿透 -> 抗性乘法叠加 -> 护甲。"""
        mult = 1.0
        for r in resists:
            eff = r - penetration          # 1. 先穿透
            if eff < 0.0:
                eff = 0.0
            mult *= 1.0 - eff              # 2. 抗性乘法叠加（恒 > 0）
        mult *= self.armor_k / (self.armor_k + armor)  # 3. 护甲递减公式
        return mult

    def mitigate(self, dmg, armor, resist=0.0, penetration=0.0):
        """结算非暴击伤害，返回 float（dmg > 0 时结果恒 > 0）。"""
        dmg = self._check("dmg", dmg, lo=0.0)
        armor = self._check("armor", armor, lo=0.0)
        resists = self._resist_list(resist)
        penetration = self._check("penetration", penetration, lo=0.0, hi=1.0)
        return dmg * self._multiplier(armor, resists, penetration)

    def expected(self, dmg, crit, cdmg, armor, resist=0.0, penetration=0.0):
        """解析期望伤害：E = base * (1 + crit * (cdmg - 1))。"""
        crit = self._check("crit", crit, lo=0.0, hi=1.0)
        cdmg = self._check("cdmg", cdmg, lo=1.0)
        base = self.mitigate(dmg, armor, resist, penetration)
        return base * (1.0 + crit * (cdmg - 1.0))

    def variance(self, dmg, crit, cdmg, armor, resist=0.0, penetration=0.0):
        """解析伤害方差：Var = base^2 * crit * (1 - crit) * (cdmg - 1)^2。"""
        crit = self._check("crit", crit, lo=0.0, hi=1.0)
        cdmg = self._check("cdmg", cdmg, lo=1.0)
        base = self.mitigate(dmg, armor, resist, penetration)
        d = cdmg - 1.0
        return base * base * crit * (1.0 - crit) * d * d

    def sample(self, dmg, crit, cdmg, armor, resist=0.0, penetration=0.0,
               rng=None):
        """蒙特卡洛单次结算（暴击按概率随机）。``rng`` 可传 random.Random。"""
        crit = self._check("crit", crit, lo=0.0, hi=1.0)
        cdmg = self._check("cdmg", cdmg, lo=1.0)
        base = self.mitigate(dmg, armor, resist, penetration)
        roll = rng.random() if rng is not None else random.random()
        return base * cdmg if roll < crit else base

    def explain(self, dmg, armor, resist=0.0, penetration=0.0):
        """返回有序结算步骤列表，使“先穿透后护甲”的顺序可被外部观测。"""
        dmg = self._check("dmg", dmg, lo=0.0)
        armor = self._check("armor", armor, lo=0.0)
        resists = self._resist_list(resist)
        penetration = self._check("penetration", penetration, lo=0.0, hi=1.0)

        steps = [("input", dmg)]
        effective = [max(0.0, r - penetration) for r in resists]
        steps.append(("penetration", tuple(effective)))
        mult_res = 1.0
        for r in effective:
            mult_res *= 1.0 - r
        after_resist = dmg * mult_res
        steps.append(("resist", after_resist))
        final = after_resist * self.armor_k / (self.armor_k + armor)
        steps.append(("armor", final))
        return steps
