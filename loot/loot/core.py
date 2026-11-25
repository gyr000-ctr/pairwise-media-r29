"""loot 核心：按权重累积抽样、保底（pity）与分层池。

- 概率与权重比一致：单次均匀抽样 + 累积权重二分查找。
- 保底：连续未命中稀有达到 N-1 次后，第 N 次必出稀有；命中稀有即重置计数；
  保底只在第 N 次触发，不提前。
- 分层池（StratifiedLoot）：每个分层使用由主种子确定性派生的独立随机流，
  互不污染；同种子可复现，不同种子序列不相关。
- 错误分类：空池、负/零权重、非有限权重、非法种子分别使用不同的
  ValueError 子类，可区分捕获。
- 并发安全：roll/pity 均在实例锁内执行。
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
from bisect import bisect_right

__all__ = [
    "Loot",
    "StratifiedLoot",
    "LootError",
    "EmptyPoolError",
    "InvalidWeightError",
    "NonFiniteWeightError",
    "InvalidSeedError",
    "InvalidRarityError",
    "InvalidPityError",
    "MalformedEntryError",
    "DEFAULT_RARITY_ORDER",
]

#: 默认稀有度次序（低到高）。稀有度达到 rare_threshold 即视为“稀有”。
DEFAULT_RARITY_ORDER = ("common", "uncommon", "rare", "epic", "legendary")


class LootError(Exception):
    """loot 相关错误的基类。"""


class EmptyPoolError(LootError, ValueError):
    """奖池为空（没有任何条目，或分层池为空）。"""


class InvalidWeightError(LootError, ValueError):
    """权重非法：非数值、负数或零。"""


class NonFiniteWeightError(LootError, ValueError):
    """权重非有限值（NaN 或 ±inf）。"""


class InvalidSeedError(LootError, ValueError):
    """种子非法（必须为 None 或非负 int）。"""


class InvalidRarityError(LootError, ValueError):
    """稀有度非法或不在稀有度次序中。"""


class InvalidPityError(LootError, ValueError):
    """保底参数非法（必须为 None 或 >= 1 的 int）。"""


class MalformedEntryError(LootError, ValueError):
    """奖池条目格式非法（应为 (name, weight) 或 (name, weight, rarity)）。"""


def _validate_seed(seed):
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidSeedError(
            f"seed 必须为 None 或非负 int，得到 {seed!r}（{type(seed).__name__}）"
        )
    if seed < 0:
        raise InvalidSeedError(f"seed 必须为非负整数，得到 {seed}")
    return seed


def _validate_weight(weight):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise InvalidWeightError(
            f"权重必须是数值，得到 {weight!r}（{type(weight).__name__}）"
        )
    w = float(weight)
    if math.isnan(w) or math.isinf(w):
        raise NonFiniteWeightError(f"权重必须为有限值，得到 {w}")
    if w <= 0.0:
        raise InvalidWeightError(f"权重必须为正数，得到 {w}")
    return w


def _validate_pity(pity):
    if pity is None:
        return None
    if isinstance(pity, bool) or not isinstance(pity, int) or pity < 1:
        raise InvalidPityError(f"pity 必须为 None 或 >= 1 的 int，得到 {pity!r}")
    return pity


def _rarity_rank(rarity, order):
    """把稀有度映射为可比较的数值：数值稀有度直接用其值，字符串按次序表取序号。"""
    if isinstance(rarity, bool):
        raise InvalidRarityError(f"非法稀有度 {rarity!r}")
    if isinstance(rarity, (int, float)):
        r = float(rarity)
        if not math.isfinite(r):
            raise InvalidRarityError(f"稀有度必须为有限值，得到 {rarity!r}")
        return r
    if isinstance(rarity, str):
        try:
            return float(order.index(rarity))
        except ValueError:
            raise InvalidRarityError(
                f"未知稀有度 {rarity!r}，不在稀有度次序 {tuple(order)!r} 中"
            ) from None
    raise InvalidRarityError(f"非法稀有度 {rarity!r}（{type(rarity).__name__}）")


class Loot:
    """单池加权抽样器。

    table: [(name, weight)] 或 [(name, weight, rarity)]，weight 必须为正有限数。
    seed: 非负 int（同种子序列可复现）或 None（不可复现）。
    pity: None 或 >= 1 的 int；连续 pity-1 次未出稀有后，下一次必出稀有。
    rare_threshold: 达到该稀有度即视为稀有（默认 'rare'）。
    rarity_order: 自定义稀有度次序（字符串稀有度时使用）。
    """

    def __init__(self, table, seed, *, pity=None, rare_threshold="rare",
                 rarity_order=None):
        self._lock = threading.Lock()
        self._seed = _validate_seed(seed)
        self._pity = _validate_pity(pity)
        order = tuple(rarity_order) if rarity_order is not None else DEFAULT_RARITY_ORDER
        self._rare_rank = _rarity_rank(rare_threshold, order)

        if table is None:
            raise TypeError("table 不能为 None")
        entries = list(table)
        if not entries:
            raise EmptyPoolError("奖池为空：没有任何条目")

        names = []
        rarities = []
        cum = []
        is_rare = []
        rare_items = []
        rare_cum = []
        total = 0.0
        rare_total = 0.0
        for entry in entries:
            try:
                n = len(entry)
            except TypeError:
                raise MalformedEntryError(
                    f"条目必须为 (name, weight) 或 (name, weight, rarity)，得到 {entry!r}"
                ) from None
            if n == 2:
                name, weight = entry
                rarity = "common"
            elif n == 3:
                name, weight, rarity = entry
            else:
                raise MalformedEntryError(
                    f"条目必须为 (name, weight) 或 (name, weight, rarity)，得到 {entry!r}"
                )
            w = _validate_weight(weight)
            rare = _rarity_rank(rarity, order) >= self._rare_rank
            total += w
            names.append(name)
            rarities.append(rarity)
            cum.append(total)
            is_rare.append(rare)
            if rare:
                rare_total += w
                rare_items.append(len(names) - 1)
                rare_cum.append(rare_total)

        self._names = names
        self._rarities = rarities
        self._cum = cum
        self._total = total
        self._is_rare = is_rare
        self._rare_items = rare_items
        self._rare_cum = rare_cum
        self._rare_total = rare_total
        self._rng = random.Random(seed)
        self._counter = 0

    def roll(self):
        """抽一次，返回 (name, rarity) 元组。"""
        with self._lock:
            rng = self._rng
            rare_cum = self._rare_cum
            if (self._pity is not None and rare_cum
                    and self._counter == self._pity - 1):
                # 保底触发：仅在稀有条目内按权重抽样，不污染主序列分布语义。
                r = rng.random() * self._rare_total
                i = bisect_right(rare_cum, r)
                if i >= len(rare_cum):
                    i = len(rare_cum) - 1
                idx = self._rare_items[i]
            else:
                cum = self._cum
                r = rng.random() * self._total
                idx = bisect_right(cum, r)
                if idx >= len(cum):  # 浮点舍入兜底
                    idx = len(cum) - 1
            if self._is_rare[idx]:
                self._counter = 0
            else:
                self._counter += 1
            return (self._names[idx], self._rarities[idx])

    def pity(self):
        """当前保底计数（自上次命中稀有以来的连续未命中次数）。"""
        with self._lock:
            return self._counter


def _derive_seed(seed, key):
    """由主种子与分层名确定性派生子种子（跨进程稳定，不依赖 PYTHONHASHSEED）。"""
    digest = hashlib.sha256(f"loot-stratum|{seed}|{key!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class StratifiedLoot:
    """分层池：{分层名: 条目列表}，各分层独立抽样、互不污染。

    每个分层拥有独立的随机流（由 seed 与分层名经 SHA-256 派生）与独立的
    保底计数；从某分层抽取不会改变其他分层的序列。
    """

    def __init__(self, pools, seed, *, pity=None, rare_threshold="rare",
                 rarity_order=None):
        self._seed = _validate_seed(seed)
        if pools is None:
            raise TypeError("pools 不能为 None")
        if not isinstance(pools, dict):
            raise TypeError(f"pools 必须为 dict（分层名 -> 条目列表），得到 {type(pools).__name__}")
        if not pools:
            raise EmptyPoolError("分层池为空：没有任何分层")
        self._pools = {}
        for key, table in pools.items():
            sub_seed = None if seed is None else _derive_seed(seed, key)
            self._pools[key] = Loot(
                table,
                sub_seed,
                pity=pity,
                rare_threshold=rare_threshold,
                rarity_order=rarity_order,
            )

    @property
    def strata(self):
        """全部分层名。"""
        return tuple(self._pools)

    def _resolve(self, stratum):
        if stratum is None:
            if len(self._pools) == 1:
                return next(iter(self._pools.values()))
            raise ValueError("存在多个分层，必须指定 stratum")
        try:
            return self._pools[stratum]
        except KeyError:
            raise KeyError(f"未知分层 {stratum!r}，可用分层：{sorted(self._pools)!r}") from None

    def roll(self, stratum=None):
        """从指定分层抽一次，返回 (name, rarity) 元组。"""
        return self._resolve(stratum).roll()

    def pity(self, stratum=None):
        """指定分层的保底计数；多分层且未指定时返回 {分层名: 计数}。"""
        if stratum is None and len(self._pools) > 1:
            return {key: pool.pity() for key, pool in self._pools.items()}
        return self._resolve(stratum).pity()
