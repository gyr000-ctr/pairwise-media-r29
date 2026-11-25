# pcgrng

## 题面（可直接粘贴）

子目录：pcgrng。确定性随机（PCG）与分叉。约束：输出必须与 PCG32 参考向量一致（≤1 个 ULP）；同种子同流可复现；分叉（stream）互不相关；周期内无重复；必须支持范围映射且无偏（拒绝采样）；100 万次不超过 300 ms；非法种子可区分报错。验收：cd pcgrng && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd pcgrng && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
