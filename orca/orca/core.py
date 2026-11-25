"""orca —— 群体避障（RVO 近似的 ORCA 实现）。

本模块实现 ORCA（Optimal Reciprocal Collision Avoidance）的二维近似：
对每个邻居构造一个速度障碍半平面（责任各半，即互惠因子 0.5），再求解

    min ||v - v_pref||
    s.t. 半平面约束（det(d_i, p_i - v) <= 0）
         ||v|| <= max_speed

该二次规划在可行时等价于原文的增量线性规划 ``linearProgram2/1``；当所有
半平面的交集与速度圆盘不相交时，用 ``linearProgram3`` 做最小外推（贴边时
沿当前最优解继续推，不会来回抖动）。

约定
----
* 智能体以 ``(position, velocity, goal)`` 三元组描述，每个分量是长度为 2
  的实数序列，例如 ``((x, y), (vx, vy), (gx, gy))``。
* 所有公开方法都是无状态的（除只读参数外不写任何实例状态），因此同一
  实例可以被多个线程并发调用。
* 非法参数抛出可区分的 ``ValueError`` 子类：``InvalidRadius`` /
  ``InvalidHorizon`` / ``InvalidMaxSpeed`` / ``InvalidIndex`` /
  ``EmptyAgents`` / ``MalformedAgent`` / ``NonFiniteInput``。
"""

from __future__ import annotations

import math

__all__ = [
    "RVO_EPSILON",
    "Orca",
    "OrcaError",
    "InvalidRadius",
    "InvalidHorizon",
    "InvalidMaxSpeed",
    "InvalidIndex",
    "EmptyAgents",
    "MalformedAgent",
    "NonFiniteInput",
]

#: 判定“平行”的容差。原实现基于单精度取 1e-5，这里使用双精度并收紧。
RVO_EPSILON = 1e-9

#: 外推（交集为空）时参与投影的最近约束数上限。交集非空时根本不会走到
#: 这里；交集为空时题面只要求有限、有界、贴边不抖，因此可以截断以保证
#: 最坏情况下的耗时上界。截断是确定性的（取最近的若干条），所以结果对
#: 输入仍然连续。
_LP3_MAX_LINES = 48

#: 外推时最多进行多少次“投影域重建”。同样只影响交集为空的病态密集输入。
_LP3_MAX_REBUILDS = 96


class OrcaError(ValueError):
    """ORCA 参数/输入错误基类（可直接按 ``ValueError`` 捕获）。"""


class InvalidRadius(OrcaError):
    """半径非法（非正、非有限或非实数）。"""


class InvalidHorizon(OrcaError):
    """时间视野非法（非正、非有限或非实数）。"""


class InvalidMaxSpeed(OrcaError):
    """最大速度非法（非正、非有限或非实数）。"""


class InvalidIndex(OrcaError):
    """智能体下标非法（越界或非整数）。"""


class EmptyAgents(OrcaError):
    """智能体集合为空。"""


class MalformedAgent(OrcaError, TypeError):
    """智能体描述结构非法（不是 ``(pos, vel, goal)`` 这种结构）。"""


class NonFiniteInput(OrcaError):
    """输入包含 NaN 或无穷。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _positive_real(value, exc, name):
    """把 *value* 转成正的有限浮点数，否则抛出 *exc*。"""
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise exc("%s must be a real number, got %r" % (name, value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise exc("%s must be a real number, got %r" % (name, value)) from None
    if not math.isfinite(number):
        raise NonFiniteInput("%s must be finite, got %r" % (name, value))
    if number <= 0.0:
        raise exc("%s must be > 0, got %r" % (name, value))
    return number


def _coord(value, name):
    """把单个坐标转成有限浮点数。"""
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise MalformedAgent("%s must be a real number, got %r" % (name, value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MalformedAgent(
            "%s must be a real number, got %r" % (name, value)
        ) from None
    if not math.isfinite(number):
        raise NonFiniteInput("%s must be finite, got %r" % (name, value))
    return number


def _vec(value, name):
    """把二元序列转成 ``(float, float)``。"""
    if isinstance(value, (str, bytes, dict)):
        raise MalformedAgent("%s must be a sequence of two numbers" % (name,))
    try:
        x, y = value
    except (TypeError, ValueError):
        raise MalformedAgent(
            "%s must be a sequence of two numbers, got %r" % (name, value)
        ) from None
    return (_coord(x, name + "[0]"), _coord(y, name + "[1]"))


def _agent(value, name):
    """把 ``(position, velocity, goal)`` 转成三个 ``(float, float)``。"""
    if isinstance(value, (str, bytes, dict)):
        raise MalformedAgent(
            "%s must be a (position, velocity, goal) triple" % (name,)
        )
    try:
        position, velocity, goal = value
    except (TypeError, ValueError):
        raise MalformedAgent(
            "%s must be a (position, velocity, goal) triple, got %r"
            % (name, value)
        ) from None
    return (
        _vec(position, name + ".position"),
        _vec(velocity, name + ".velocity"),
        _vec(goal, name + ".goal"),
    )


def _parse_neighbor(value, index):
    """快速解析邻居的 ``position`` 与 ``velocity``（不读 goal）。

    与 :func:`_agent` 语义一致，但对常见的 ``float`` 序列走快路径，避免
    大规模场景下重复的类型转换开销。
    """
    if type(value) is tuple and len(value) == 3:
        position = value[0]
        velocity = value[1]
        if (
            type(position) is tuple
            and type(velocity) is tuple
            and len(position) == 2
            and len(velocity) == 2
            and type(position[0]) is float
            and type(position[1]) is float
            and type(velocity[0]) is float
            and type(velocity[1]) is float
        ):
            px = position[0]
            py = position[1]
            vx = velocity[0]
            vy = velocity[1]
            # ``x - x == 0`` 为真当且仅当 ``x`` 有限（NaN 与无穷都会失败）。
            if (
                px - px == 0.0
                and py - py == 0.0
                and vx - vx == 0.0
                and vy - vy == 0.0
            ):
                return position, velocity
    name = "agents[%d]" % (index,)
    position, velocity, _goal = _agent(value, name)
    return position, velocity


# ---------------------------------------------------------------------------
# 线性规划（对应 RVO2 的 linearProgram1/2/3）
# ---------------------------------------------------------------------------


def _linear_program1(lines, line_no, radius, opt_x, opt_y, direction_opt):
    """在速度圆盘与 ``lines[:line_no]`` 的交上，沿第 *line_no* 条直线优化。

    返回最优点，若约束不可行（含与速度圆盘无交）返回 ``None``。
    """
    px, py, dx, dy = lines[line_no]
    dot = px * dx + py * dy
    discriminant = dot * dot + radius * radius - (px * px + py * py)
    if discriminant < 0.0:
        # 最大速度圆盘完全排除该直线。
        return None

    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc

    for k in range(line_no):
        qx, qy, ex, ey = lines[k]
        denominator = dx * ey - dy * ex
        numerator = ex * (py - qy) - ey * (px - qx)
        if abs(denominator) <= RVO_EPSILON:
            # 两条直线（几乎）平行：若可行域在错误的一侧则无解。
            if numerator < 0.0:
                return None
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            if t < t_right:
                t_right = t
        else:
            if t > t_left:
                t_left = t
        if t_left > t_right:
            return None

    if direction_opt:
        # 只优化方向：取更接近期望方向的那一端。
        if opt_x * dx + opt_y * dy > 0.0:
            t = t_right
        else:
            t = t_left
    else:
        # 优化最近的投影点。
        t = dx * (opt_x - px) + dy * (opt_y - py)
        if t < t_left:
            t = t_left
        elif t > t_right:
            t = t_right
    return (px + t * dx, py + t * dy)


def _linear_program2(lines, radius, opt_x, opt_y, direction_opt):
    """在速度圆盘与全部半平面之交上求最优点。

    返回 ``(失败/完成的下标, 结果)``：下标等于 ``len(lines)`` 表示成功，
    否则表示在该条直线处失败，结果为失败前保留的点。

    这是经典的 Seidel 型增量线性规划。约束的先后顺序不影响最终的最优
    解（交集非空时结果精确到机器精度），但会显著影响迭代次数；调用方
    会先把最“紧”的约束排在前面。
    """
    if direction_opt:
        rx, ry = opt_x * radius, opt_y * radius
    else:
        opt_sq = opt_x * opt_x + opt_y * opt_y
        if opt_sq > radius * radius:
            length = math.sqrt(opt_sq)
            rx, ry = opt_x / length * radius, opt_y / length * radius
        else:
            rx, ry = opt_x, opt_y

    for k in range(len(lines)):
        px, py, dx, dy = lines[k]
        if dx * (py - ry) - dy * (px - rx) > 0.0:
            # 当前最优解违反约束 k，重新求解。
            previous = (rx, ry)
            result = _linear_program1(
                lines, k, radius, opt_x, opt_y, direction_opt
            )
            if result is None:
                return k, previous
            rx, ry = result
    return len(lines), (rx, ry)


def _linear_program3(lines, num_obstacle_lines, begin, radius, result):
    """半平面交集不可行时的最小外推（对应 :func:`linearProgram3`）。

    与原实现语义一致：逐条检查失败点之后的约束，若当前点违反第 ``i`` 条，
    就把它与前面各直线两两取“中线”，在半平面族上交出新的可行域，并在
    其中找沿 ``i`` 的垂直方向最远（即让 ``i`` 的违反量最小）的点。``result``
    始终被保留为可行域内的点，因此解有限、有界，且沿当前点继续外推，
    贴边时不会在两个解之间抖动。

    交集为空属于病态密集输入，题面只要求有限、有界、不抖；因此这里把
    参与(重新)投影的约束数量限制为常数，保证最坏情况下的耗时上界，同时
    因为窗口按当前违反量确定性地选取，结果对输入仍然连续。
    """
    rx, ry = result
    distance = 0.0

    if len(lines) - begin > _LP3_MAX_REBUILDS * 4:
        order = sorted(
            range(begin, len(lines)),
            key=lambda k: -(lines[k][2] * (lines[k][1] - ry)
                            - lines[k][3] * (lines[k][0] - rx)),
        )[: _LP3_MAX_REBUILDS * 2]
    else:
        order = range(begin, len(lines))

    rebuilds = 0
    for pos in range(len(order)):
        if rebuilds >= _LP3_MAX_REBUILDS:
            break
        i = order[pos]
        px, py, dx, dy = lines[i]
        if dx * (py - ry) - dy * (px - rx) <= distance:
            continue

        rebuilds += 1

        # 当前点违反第 i 条。用障碍线 + 该条附近最近的若干直线构造投影域。
        projected = list(lines[:num_obstacle_lines])
        for j in range(max(num_obstacle_lines, i - _LP3_MAX_LINES), i):
            qx, qy, ex, ey = lines[j]
            determinant = dx * ey - dy * ex
            if abs(determinant) <= RVO_EPSILON:
                if dx * ex + dy * ey > 0.0:
                    # 同向平行线，保留约束更强的一条即可。
                    continue
                line_x = 0.5 * (px + qx)
                line_y = 0.5 * (py + qy)
            else:
                factor = (ex * (py - qy) - ey * (px - qx)) / determinant
                line_x = px + factor * dx
                line_y = py + factor * dy
            ndx = ex - dx
            ndy = ey - dy
            length = math.hypot(ndx, ndy)
            if length <= 0.0:
                continue
            projected.append((line_x, line_y, ndx / length, ndy / length))

        previous = (rx, ry)
        count, candidate = _linear_program2(
            projected, radius, -dy, dx, True
        )
        if count < len(projected):
            # 理论上不会发生；数值退化时保留原点，保证稳定。
            candidate = previous
        rx, ry = candidate
        distance = dx * (py - ry) - dy * (px - rx)

    return (rx, ry)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


class Orca:
    """基于半平面速度障碍的互惠避障（ORCA）。"""

    __slots__ = ("radius", "horizon", "max_speed", "time_step")

    def __init__(self, radius=0.5, horizon=2.0, max_speed=1.0, time_step=None):
        """构造。

        参数
        ----
        radius : float
            单个智能体的半径（正值、有限）。
        horizon : float
            碰撞预测时间视野（正值、有限）。
        max_speed : float
            速度上界（正值、有限），默认 1.0。
        time_step : float, optional
            已经重叠（碰撞）时用于投影的时间步长，默认取 ``horizon``。
        """
        radius = _positive_real(radius, InvalidRadius, "radius")
        horizon = _positive_real(horizon, InvalidHorizon, "horizon")
        max_speed = _positive_real(max_speed, InvalidMaxSpeed, "max_speed")
        if time_step is None:
            time_step = horizon
        else:
            time_step = _positive_real(time_step, InvalidHorizon, "time_step")

        self.radius = radius
        self.horizon = horizon
        self.max_speed = max_speed
        self.time_step = time_step

    # -- 期望速度 ----------------------------------------------------------

    def preferred(self, vel, pos, goal):
        """返回朝目标点、模长为 ``max_speed`` 的期望速度。

        ``vel`` 是当前速度（保留参数以便与模拟循环衔接，不改变期望方向）；
        当已位于目标点时返回零速度。
        """
        _vec(vel, "vel")
        pos = _vec(pos, "pos")
        goal = _vec(goal, "goal")

        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise NonFiniteInput("goal - pos overflows")
        distance = math.hypot(dx, dy)
        if distance == 0.0:
            return (0.0, 0.0)
        scale = self.max_speed / distance
        vx = dx * scale
        vy = dy * scale
        # 浮点取舍可能让模长以 1 ulp 超出 max_speed；这里保证返回值严格
        # 落在速度圆盘内，这样“无邻居时保持期望速度”可以直接原样返回，
        # 不必再经过线性规划的归一化。
        norm = math.hypot(vx, vy)
        if norm > self.max_speed:
            correction = self.max_speed / norm
            vx *= correction
            vy *= correction
        # ``+ 0.0`` 把 -0.0 归一成 0.0，保证结果可比较、可复现。
        return (vx + 0.0, vy + 0.0)

    # -- 避障速度 ----------------------------------------------------------

    def avoid(self, agents, i):
        """计算第 *i* 个智能体的避障速度，返回 ``(vx, vy)``。

        ``agents`` 为 ``(position, velocity, goal)`` 序列；不会修改入参。
        """
        n = self._scene_size(agents)
        index = self._check_index(i, n)

        pos_i, vel_i, goal_i = _agent(agents[index], "agents[%d]" % (index,))
        preferred = self.preferred(vel_i, pos_i, goal_i)

        lines = []
        inv_horizon = 1.0 / self.horizon
        combined_radius = self.radius + self.radius
        max_speed = self.max_speed

        # 注意：不能按距离剔除“远处”的邻居。ORCA 半平面是无限延伸的直线，
        # 即使几何上不可能在视野内相撞，其半平面仍可能切到速度圆盘；只有
        # 下面 ``offset >= max_speed`` 的判定才是精确冗余。
        for j in range(n):
            if j == index:
                continue
            pos_j, vel_j = _parse_neighbor(agents[j], j)
            rx = pos_j[0] - pos_i[0]
            ry = pos_j[1] - pos_i[1]
            dist_sq = rx * rx + ry * ry
            if not math.isfinite(dist_sq):
                raise NonFiniteInput(
                    "distance to agents[%d] overflows" % (j,)
                )

            line = self._orca_line(
                pos_i, vel_i, pos_j, vel_j,
                dist_sq, combined_radius, inv_horizon,
            )
            # 半平面约束为 ``det(direction, point - v) <= 0``，等价于
            # ``a . v <= b``，其中 ``a = (dy, -dx)`` 且
            # ``b = dy * px - dx * py``。``direction`` 是单位向量，所以
            # ``b`` 就是原点到直线的有符号距离；若 ``b >= max_speed``，
            # 速度圆盘内任意速度都满足该约束，直线必然冗余。这里用
            # ``|a|`` 归一化，即便退化输入下 ``direction`` 不是严格单位
            # 向量也依然精确。
            px, py, dx, dy = line
            offset = dy * px - dx * py
            direction_length = math.hypot(dx, dy)
            if direction_length > 0.0:
                offset /= direction_length
                if offset >= max_speed:
                    continue
            else:
                # 方向退化：约束退化为对 point 的一个点约束，仍按保守值处理。
                offset = math.hypot(px, py)
            # 越小的 ``offset`` 越贴近速度圆盘，也就越可能成为紧约束。
            # 增量 LP 的耗时强烈依赖约束顺序，按“紧到松”排序可让可行的
            # 场景几乎一遍通过（顺序不改变最优解）。
            lines.append((offset, (px, py, dx, dy)))

        lines.sort(key=lambda item: item[0])
        constraints = [item[1] for item in lines]

        if not constraints:
            # 没有邻居，或所有约束对速度圆盘都冗余：期望速度本身就是最优解，
            # 直接原样返回（满足“无邻居时保持期望速度”）。
            return preferred

        count, result = _linear_program2(
            constraints, max_speed, preferred[0], preferred[1], False
        )
        if count < len(constraints):
            result = _linear_program3(
                constraints, 0, count, max_speed, result
            )

        vx, vy = result
        if not (math.isfinite(vx) and math.isfinite(vy)):
            # 理论上不会出现；退回期望速度以保证输出始终有限。
            vx, vy = preferred
        return (float(vx) + 0.0, float(vy) + 0.0)

    # -- 内部工具 ----------------------------------------------------------

    def _orca_line(self, pos_i, vel_i, pos_j, vel_j, dist_sq,
                   combined_radius, inv_horizon):
        """构造约束 ``det(direction, point - v) <= 0`` 的 ORCA 直线。"""
        rx = pos_j[0] - pos_i[0]
        ry = pos_j[1] - pos_i[1]
        rvx = vel_i[0] - vel_j[0]
        rvy = vel_i[1] - vel_j[1]
        if not (math.isfinite(rx) and math.isfinite(ry)
                and math.isfinite(rvx) and math.isfinite(rvy)):
            raise NonFiniteInput("relative position/velocity overflows")

        combined_sq = combined_radius * combined_radius

        if dist_sq > combined_sq:
            # 未重叠：在截断锥的圆弧或两条腿上投影。
            wx = rvx - inv_horizon * rx
            wy = rvy - inv_horizon * ry
            w_sq = wx * wx + wy * wy
            dot = wx * rx + wy * ry
            leg_sq = dist_sq - combined_sq
            if leg_sq < 0.0:
                # 数值上 ``dist_sq`` 略小于 ``combined_sq``（浮点舍入）。
                leg_sq = 0.0
            leg = math.sqrt(leg_sq)
            side = rx * wy - ry * wx

            # 判定是否投影到截断圆：``dot < 0`` 表示正在接近，且 ``w`` 落在
            # 以 ``rel`` 为轴、半径 ``combined_radius`` 的柱面之外。
            # 另外，当 ``side`` 接近 0（两者正对彼此）时也回退到截断圆：
            # 此时腿分支与“已重叠”的截断圆分支给出不同的解，恰好跨过
            # ``dist == combined_radius`` 会跳变；统一走截断圆即可保持连续。
            # ``w_sq == 0`` 时 ``w / |w|`` 未定义（相对速度恰为 ``rel /
            # horizon``，必在锥内），安全地留给腿分支。
            tolerance = RVO_EPSILON * math.sqrt(dist_sq * w_sq)
            use_cutoff = w_sq > 0.0 and (
                (dot < 0.0 and dot * dot > combined_sq * w_sq)
                or abs(side) <= tolerance
            )

            if use_cutoff:
                w_length = math.sqrt(w_sq)
                ux = (combined_radius * inv_horizon - w_length) * (wx / w_length)
                uy = (combined_radius * inv_horizon - w_length) * (wy / w_length)
                dx = wy / w_length
                dy = -wx / w_length
            elif w_sq <= 0.0 and abs(side) <= RVO_EPSILON * dist_sq:
                # ``w`` 为零（相对速度恰为 ``rel / horizon``）且两者正对：
                # 此时没有可用的法向，直接取截断圆在 ``rel`` 方向上的
                # 外向法线，保证结果连续且方向单位化。
                inv_dist = 1.0 / math.sqrt(dist_sq)
                wx = rx * inv_dist
                wy = ry * inv_dist
                ux = (combined_radius * inv_horizon) * wx
                uy = (combined_radius * inv_horizon) * wy
                dx = wy
                dy = -wx
            else:
                if side > 0.0:
                    # 左腿。
                    dx = (rx * leg - ry * combined_radius) / dist_sq
                    dy = (rx * combined_radius + ry * leg) / dist_sq
                else:
                    # 右腿。
                    dx = -(rx * leg + ry * combined_radius) / dist_sq
                    dy = -(-rx * combined_radius + ry * leg) / dist_sq
                projection = rvx * dx + rvy * dy
                ux = projection * dx - rvx
                uy = projection * dy - rvy
        else:
            # 已经重叠：投影到时间步长对应的截断圆。
            inv_time_step = 1.0 / self.time_step
            wx = rvx - inv_time_step * rx
            wy = rvy - inv_time_step * ry
            w_length = math.hypot(wx, wy)
            if w_length <= 0.0:
                # 完全重合且相对速度为零：沿 +x 方向确定性地推开。
                ux = combined_radius * inv_time_step
                uy = 0.0
                dx = 0.0
                dy = -1.0
            else:
                uwx = wx / w_length
                uwy = wy / w_length
                ux = (combined_radius * inv_time_step - w_length) * uwx
                uy = (combined_radius * inv_time_step - w_length) * uwy
                dx = uwy
                dy = -uwx

        # 互惠因子 0.5：双方各承担一半的避让量。
        point_x = vel_i[0] + 0.5 * ux
        point_y = vel_i[1] + 0.5 * uy
        return (point_x, point_y, dx, dy)

    @staticmethod
    def _scene_size(agents):
        if agents is None:
            raise EmptyAgents("agents must not be None")
        try:
            n = len(agents)
        except TypeError:
            raise MalformedAgent(
                "agents must be a sequence of agent triples"
            ) from None
        if n == 0:
            raise EmptyAgents("agents must not be empty")
        return n

    @staticmethod
    def _check_index(i, n):
        if isinstance(i, bool) or not isinstance(i, int):
            raise InvalidIndex("index must be an integer, got %r" % (i,))
        if i < 0 or i >= n:
            raise InvalidIndex(
                "index %d out of range for %d agents" % (i, n)
            )
        return i
