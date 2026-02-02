# ParallelAdvisor

> Teaching LLMs when and how to parallelize code.

An RL environment for training LLMs to recommend optimal parallelization strategies for C++ code.

## The Problem

Parallelizing code is hard. You need to consider:
- Problem size vs. thread overhead
- Data dependencies and race conditions
- Memory access patterns (cache locality, false sharing)
- Hardware capabilities (cores, memory bandwidth, GPU)

LLMs can reason about code, but they often give generic advice. This project trains LLMs to give **specific, actionable, and correct** parallelization recommendations.

## How It Works

```
Input: Serial C++ code + hardware spec
        ↓
LLM Analysis → Recommendation + Reasoning
        ↓  
Code Generation (parallel version)
        ↓
Verification:
  1. Compiles?
  2. Correct output?
  3. Speedup achieved?
        ↓
Reward → RL Training
```

## Features

- **Strategy Selection**: Serial, std::thread(N), OpenMP(N), CUDA (Phase 2)
- **Thread Count**: Recommends optimal thread count
- **Reasoning**: Explains why a strategy is appropriate
- **Verification**: Compiles, runs, and benchmarks automatically
- **RL Training**: Uses Prime Intellect Verifiers framework

## Quick Start

```bash
# Clone
git clone https://github.com/JoshuaSchell/parallel-advisor.git
cd parallel-advisor

# Setup
uv sync
prime lab setup --skip-install

# Run evaluation
prime eval run parallel-advisor -m gpt-4o-mini
```

## Project Structure

```
parallel-advisor/
├── environments/
│   └── parallel_advisor/
│       ├── parallel_advisor.py  # Main environment
│       ├── dataset.py           # C++ code samples
│       ├── rubric.py            # Scoring (compile + correctness + speedup)
│       └── compiler.py          # Build & benchmark utilities
├── configs/
│   └── train.yaml               # RL training config
├── scripts/
│   └── eval_baseline.py         # Zero-shot evaluation
└── README.md
```

## Roadmap

- [x] **Phase 1**: Serial → {Serial, Threads, OpenMP}
- [ ] **Phase 2**: Add CUDA/GPU
- [ ] **Phase 3**: RL training loop
- [ ] **Phase 4**: Natural language task input

## Blog Posts

1. "Teaching LLMs When to Parallelize" (Phase 1 results)
2. "Adding GPU Awareness to ParallelAdvisor" (Phase 2)
3. "RL Fine-tuning for Parallelization" (Phase 3)

## Authors

- Josh Schell ([@JoshuaSchell](https://github.com/JoshuaSchell))

## License

MIT
