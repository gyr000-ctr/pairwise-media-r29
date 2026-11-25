"""skillcd 核心 API（初始未实现）。"""


class SkillCd:
    def __init__(self, charges, recharge, cdr=0.0):
        """构造"""
        raise NotImplementedError

    def try_use(self, t):
        """尝试使用"""
        raise NotImplementedError

    def tick(self, dt):
        """推进"""
        raise NotImplementedError

    def charges(self):
        """当前层数"""
        raise NotImplementedError
