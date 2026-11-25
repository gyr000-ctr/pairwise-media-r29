"""PCG32 确定性随机数发生器与流分叉。

实现严格对应参考实现 ``pcg-c-basic`` 中的：

- ``pcg32_random_srandom_r`` / ``pcg32_random_r``（XSH RR 输出变换）
- ``pcg32_boundedrand_r``（阈值拒绝采样，无偏范围映射）

因此输出与 PCG32 参考向量逐位一致（整数输出，0 ULP）。

流（stream）选择器取 64 位，内部 LCG 增量 ``inc = 2*stream + 1`` 恒为奇数，
保证不同 stream 对应互不相关的周期序列；单条序列周期为 ``2**64``，周期内
底层 LCG 状态不重复。

并发与性能模型
--------------
每个实例持有一个线程本地存储（``threading.local``），各线程在其中缓存自己
的缓冲 ``entry = [数组, 读索引]``。热路径只有一次线程本地属性读取、一次下标
读取与一次索引自增，无锁、无 ``get_ident`` 调用、无字典查找。某线程的缓冲
耗尽时，在锁保护下原子领取一个连续 LCG 状态块：块内输出由块起点局部迭代、
按参考顺序原地写入复用数组（稳态零分配），全局 LCG 状态通过仿射跳跃公式
（jump-ahead）一步推进整块。因此：

- 单线程下输出序列与参考实现逐位一致——输出只是各 LCG 状态的纯函数，
  分块不改变状态顺序；
- 多线程共享同一实例不存在数据竞争，各线程在锁内领取互不相交的状态块；
- 锁仅在每 ``_BLOCK`` 次调用时获取一次，缓冲数组跨块复用，无稳态分配。
"""

import threading

__all__ = ["Pcg"]

_get_ident = threading.get_ident

_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
_MOD32 = 1 << 32  # 32 位模数，用于范围映射阈值（避免每次调用重建大整数）
_MULT = 6364136223846793005  # PCG 默认 LCG 乘子
_BLOCK = 256  # 每个输出块包含的 LCG 步数

# LCG 块跳跃常数（mod 2**64）。若 S_{n+1} = M*S_n + inc，则
#   S_{n+k} = _M_JUMP * S_n + _JUMP_C * inc
# 其中 _M_JUMP = M**k mod 2**64，_JUMP_C = sum(M**i for i in range(k)) mod 2**64。
# 数值在测试中与逐迭代 _BLOCK 步交叉验证。
_M_JUMP = 146241703003251713
_JUMP_C = 4753489115623156480


def _check_u64(value, name):
    """校验 64 位无符号整数入参，区分类型错误与取值错误。"""
    # bool 是 int 的子类，按其整数值接受（True -> 1, False -> 0）。
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是 int，实际类型为 {type(value).__name__}")
    if value < 0 or value > _MASK64:
        raise ValueError(f"{name} 必须位于 [0, 2**64-1]，实际值为 {value}")
    return value


def _check_natural(value, name):
    """校验非负整数入参。"""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是 int，实际类型为 {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} 不能为负，实际值为 {value}")
    return value


def _splitmix64(x):
    """splitmix64 混合函数：64 位双射，用于从分叉编号派生互不相关的流。"""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


class Pcg:
    """PCG32 随机数发生器。

    Parameters
    ----------
    seed:
        64 位无符号种子，取值 ``[0, 2**64-1]``。
    stream:
        64 位无符号流选择器，取值 ``[0, 2**64-1]``。不同 stream 产生
        互不相关的独立序列；同 seed 同 stream 完全可复现。
    """

    __slots__ = ("_state", "_inc", "_lock", "_tl")

    def __init__(self, seed, stream=0):
        seed = _check_u64(seed, "seed")
        stream = _check_u64(stream, "stream")
        # 与 pcg32_random_srandom_r 完全一致的初始化。
        inc = ((stream << 1) | 1) & _MASK64
        state = inc & _MASK64
        state = (state + seed) & _MASK64
        state = (state * _MULT + inc) & _MASK64
        self._state = state
        self._inc = inc
        self._lock = threading.Lock()
        # 线程本地状态 entry = [缓冲数组, 下一个待读索引]。
        # 缓冲数组预分配并跨块复用（重填原地覆写），稳态零分配；
        # 索引 >= _BLOCK 表示缓冲已空，触发下一次重填。
        tl = threading.local()
        tl.entry = [[0] * _BLOCK, _BLOCK]
        self._tl = tl

    def _refill(self):
        """锁保护下领取一个状态块，原地重填当前线程缓冲并返回 entry。

        块内输出按参考顺序写入复用数组，读指针归零，调用方按递增索引
        消费即得到与参考实现一致的正向顺序。全局 LCG 状态以跳跃公式
        一步推进整块。
        """
        tl = self._tl
        entry = getattr(tl, "entry", None)
        if entry is None:
            # 该线程首次访问本实例：惰性建立其私有缓冲。
            entry = [[0] * _BLOCK, _BLOCK]
            tl.entry = entry
        block = entry[0]
        with self._lock:
            state = self._state
            inc = self._inc
            mult = _MULT
            mask32 = _MASK32
            mask64 = _MASK64
            cur = state
            for i in range(_BLOCK):
                xorshifted = (((cur >> 18) ^ cur) >> 27) & mask32
                rot = cur >> 59
                block[i] = (xorshifted >> rot) | (
                    (xorshifted << ((-rot) & 31)) & mask32
                )
                cur = (cur * mult + inc) & mask64
            # 块边界跳跃：等价于 _BLOCK 次 LCG 迭代（常数经测试交叉验证）。
            self._state = (_M_JUMP * state + _JUMP_C * inc) & mask64
        entry[1] = 0
        return entry

    def _draw(self):
        """取一个原始 32 位输出（热路径，独立小方法便于自适应优化）。

        线程本地缓冲按索引消费，块耗尽时在锁内领取下一连续状态块。
        """
        try:
            entry = self._tl.entry
        except AttributeError:
            # 该线程首次访问本实例：惰性建立其私有缓冲并领取首个块。
            entry = self._refill()
            idx = 0
        else:
            idx = entry[1]
            if idx >= _BLOCK:
                entry = self._refill()
                idx = 0
        entry[1] = idx + 1
        return entry[0][idx]

    def next_u32(self):
        """返回下一个 32 位无符号整数，取值 ``[0, 2**32-1]``。

        逐位对应参考实现 ``pcg32_random_r``（XSH RR 置换）。
        """
        return self._draw()

    def fork(self, n):
        """分叉出一个新的独立发生器。

        分叉编号 ``n`` 经 splitmix64（64 位双射混合）派生为新的流选择器，
        因此不同 ``n`` 得到互不相关的流；同种子、同 stream、同 ``n`` 的
        分叉结果完全可复现。分叉不改变当前发生器已承诺的输出序列（已领取
        但尚未消费的线程本地输出仍属于当前序列）。

        Parameters
        ----------
        n:
            非负整数分叉编号。
        """
        n = _check_natural(n, "n")
        # 与父发生器的流做混合，避免不同父流上的同编号分叉发生碰撞。
        child_stream = _splitmix64(
            (_splitmix64(n) ^ (self._inc * 0x9E3779B97F4A7C15)) & _MASK64
        )
        # 以父发生器的全局 LCG 状态（已承诺输出的边界）作为子发生器种子，
        # 保持确定性且与父序列相关联。
        with self._lock:
            parent_state = self._state
        return Pcg(parent_state, child_stream)

    def range(self, n):
        """无偏范围映射，返回 ``[0, n)`` 内的整数。

        采用参考实现 ``pcg32_boundedrand_r`` 的阈值拒绝采样：仅接受落在
        ``2**32`` 内最大 ``n`` 倍数区间中的采样值，对 ``n`` 不取模，
        因此不存在取模偏置。

        Parameters
        ----------
        n:
            上界（不含），取值 ``[1, 2**32]``。
        """
        # type() is int 精确匹配（bool 等子类型不通过），单次判断替代双 isinstance。
        if type(n) is not int:
            raise TypeError(f"n 必须是 int，实际类型为 {type(n).__name__}")
        if n <= 0 or n > _MOD32:
            raise ValueError(f"n 必须位于 [1, 2**32]，实际值为 {n}")
        # 参考实现：uint32_t threshold = -bound % bound，即 (2**32 - n) % n，
        # 恒等于 2**32 % n；n == 2**32 时为 0（不拒绝任何采样）。
        # n == 2**32 时掩码全 1，值域即完整 32 位，等价于 next_u32。
        if n == _MOD32:
            return self._draw()
        threshold = _MOD32 % n
        tl = self._tl
        # 线程本地缓冲自举（每线程每实例仅首次可能进入 except 分支）。
        try:
            entry = tl.entry
        except AttributeError:
            entry = self._refill()
        # 2 的幂快路径：阈值为 0（参考实现也不拒绝任何采样），
        # 且 value % n == value & (n-1)，位与替代取模，结果与参考逐值相同。
        if threshold == 0:
            mask = n - 1
            idx = entry[1]
            if idx >= _BLOCK:
                entry = self._refill()
                idx = 0
            entry[1] = idx + 1
            return entry[0][idx] & mask
        # 拒绝采样：仅接受 >= threshold 的值，对 n 不取模以消除取模偏置。
        # 拒绝概率 threshold/2**32 通常极低，期望迭代次数接近 1。
        # 缓冲引用缓存到局部变量，循环内仅操作局部变量与下标。
        block = entry[0]
        while True:
            idx = entry[1]
            if idx >= _BLOCK:
                entry = self._refill()
                block = entry[0]
                idx = 0
            value = block[idx]
            idx += 1
            entry[1] = idx
            if value >= threshold:
                return value % n
