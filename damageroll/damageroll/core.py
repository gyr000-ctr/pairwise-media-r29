"""damageroll 核心 API（初始未实现）。"""


class Damage:
    def __init__(self, armor_k):
        """构造"""
        raise NotImplementedError

    def mitigate(self, dmg, armor, resist):
        """减伤"""
        raise NotImplementedError

    def expected(self, dmg, crit, cdmg, armor, resist):
        """期望伤害"""
        raise NotImplementedError
