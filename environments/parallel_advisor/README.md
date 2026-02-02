# ParallelAdvisor

LLM-based parallelization strategy recommendation using [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers).

## Overview

Given serial C++ code and hardware specifications, ParallelAdvisor recommends the optimal parallelization strategy:

- **SERIAL**: Keep code serial (overhead would hurt performance)
- **THREADS(N)**: Use `std::thread` with N threads
- **OPENMP(N)**: Use OpenMP with N threads
- **CUDA**: Use GPU (Phase 2)

## Features

- Structured recommendation with reasoning
- Automatic compilation and benchmarking
- Correctness verification against serial baseline
- Speedup-based reward for RL training

## Quick Start

```bash
# Install verifiers
uv add verifiers

# Install this environment
prime env install ./environments/parallel_advisor

# Run evaluation
prime eval run parallel-advisor -m gpt-4o-mini
```

## Usage

```python
import verifiers as vf
from parallel_advisor import load_environment

# Load environment
env = load_environment(
    dataset_name="basic",
    hardware_cores=8,
    hardware_threads=16,
)

# Run evaluation
results = await env.evaluate(model="gpt-4o")
```

## Reward Structure

| Condition | Reward |
|-----------|--------|
| Failed to parse/compile | 0.0 |
| Compiled but wrong output | 0.2 |
| Correct output, no speedup | 0.5 |
| Correct + speedup | 0.5 - 1.0 (linear scale) |

## Dataset

The basic dataset includes:

- Array sum (reduction)
- Matrix multiplication (nested loops)
- Vector addition (embarrassingly parallel)
- Small loop (overhead-dominated)
- Dot product (reduction)
- Mandelbrot (compute-bound, irregular)
- Histogram (atomics)
- Prefix sum (data-dependent)

## Project Phases

### Phase 1: MVP (Current)
- Serial → {Serial, std::thread, OpenMP}
- Thread count recommendation
- Compile + correctness + speedup verification

### Phase 2: CUDA
- Add GPU option
- Data transfer overhead consideration
- Kernel launch overhead vs. problem size

### Phase 3: RL Training
- Fine-tune with prime-rl
- Improve recommendations via speedup reward
