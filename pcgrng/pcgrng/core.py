"""pcgrng 核心 API（初始未实现）。"""


class Pcg:
    def __init__(self, seed, stream=0):
        """构造"""
        raise NotImplementedError

    def next_u32(self):
        """下一个 32 位"""
        raise NotImplementedError

    def fork(self, n):
        """分叉"""
        raise NotImplementedError

    def range(self, n):
        """范围映射"""
        raise NotImplementedError
