# PyTorch Profiler 使用指南

## 已集成功能

我已经在 `run_alphafold.py` 中集成了 PyTorch Profiler，它会自动分析性能瓶颈。

## 如何运行

直接运行您的脚本即可，profiler 会自动在第一个 seed 上启用：

```bash
sbatch run_af3_optimized.sh
```

## 输出内容

### 1. 控制台输出

运行完成后，会在终端看到两个表格：

**Top 20 CPU 时间消耗操作**
- 显示哪些操作最耗时
- 关注 `Self CPU total` 列（最重要）

**Top 20 内存消耗操作**
- 显示哪些操作最占内存
- 关注 `Self CPU Mem` 列

### 2. Chrome Trace 文件

生成 `profiler_trace.json` 文件，可以：

1. 在浏览器中打开 `chrome://tracing`
2. 点击 "Load" 加载 `profiler_trace.json`
3. 可视化查看时间线

## 如何分析瓶颈

### 查看 CPU 时间表格

重点关注这些操作：
- `aten::mm` / `aten::matmul` - 矩阵乘法
- `aten::linear` - 全连接层
- `aten::softmax` - Softmax（attention）
- `aten::layer_norm` - LayerNorm
- `aten::addmm` - 矩阵加法乘法

如果看到：
- **`aten::einsum` 占比高** → 需要优化 Triangle Multiplication
- **`aten::softmax` + `aten::matmul` 在 attention 中占比高** → 需要使用 SDPA
- **某些操作重复出现多次** → 可能需要缓存

### 示例输出解读

```
Self CPU %      Self CPU   CPU total  CPU time avg     # of Calls  Name
---------------------------------------------------------
    30.00%      21.000s     21.000s      2.100s              10  aten::einsum
    20.00%      14.000s     14.000s      0.029s             480  aten::matmul
    10.00%       7.000s      7.000s      0.146s              48  aten::softmax
```

这表示：
- **einsum 占 30% 时间** → 优先优化目标！
- matmul 占 20%
- softmax 占 10%

## 下一步

运行一次后，查看输出，告诉我 Top 5 最耗时的操作，我会针对性优化。
