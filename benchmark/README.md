# ReAct on HotpotQA — Benchmark

按 Yao et al., 2023 *"ReAct: Synergizing Reasoning and Acting in Language Models"* 复现：在 **HotpotQA** dev 集随机采样 100 条，使用 **Wikipedia API** 三类动作 `Search[entity]`、`Lookup[keyword]`、`Finish[answer]` 评测 ReAct 智能体。

## 结构

```
benchmark/
├── wiki_tools.py        # Wikipedia API: Search / Lookup（论文 3.1 节）
├── prompt_hotpot.py     # ReAct few-shot 提示词（论文 Appendix C）
├── hotpot_react.py      # 评测用 ReAct agent（非流式 + stop=["Observation"]）
├── load_data.py         # 下载 HotpotQA dev 并随机采样 100 条（seed=42）
├── evaluate.py          # 官方 HotpotQA 风格 EM / F1
├── run_benchmark.py     # 主入口
├── data/                # 缓存的原始数据 + 100 条采样
└── results/             # 评测输出
```

## 用法

```bash
# 跑小样本快速验证
python -m benchmark.run_benchmark --n 5 --verbose

# 完整 100 条评测
python -m benchmark.run_benchmark --n 100
```

结果会写到 `benchmark/results/react_hotpot_n100_*.json`，每条记录包含完整 `thought / action / observation` 轨迹，便于错例分析。
