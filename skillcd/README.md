# skillcd

## 题面（可直接粘贴）

子目录：skillcd。技能冷却与充能。约束：多层充能（最多 N 层）恢复按层间隔推进且不溢出；冷却缩减按乘法叠乘且下限；时间缩放后剩余时间按缩放累积（≤1e-9）；使用在无充能时失败并给出可区分原因；10 万次 tick 不超过 200 ms；并发安全。验收：cd skillcd && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd skillcd && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
