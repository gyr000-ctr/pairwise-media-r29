# damageroll

## 题面（可直接粘贴）

子目录：damageroll。伤害结算（护甲/抗性/暴击）。约束：护甲按递减公式（不得线性抵消）且收益单调递减；抗性乘法叠加且下限不为 0（伤害 >0）；暴击期望值与解析期望一致（≤1e-9）；结算顺序固定（先抗性后护甲）；1 万次结算不超过 200 ms；非法参数可区分报错。验收：cd damageroll && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd damageroll && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
