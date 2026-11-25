"""orca 核心 API（初始未实现）。"""


class Orca:
    def __init__(self, radius, horizon):
        """构造"""
        raise NotImplementedError

    def preferred(self, vel, pos, goal):
        """期望速度"""
        raise NotImplementedError

    def avoid(self, agents, i):
        """避障速度"""
        raise NotImplementedError
