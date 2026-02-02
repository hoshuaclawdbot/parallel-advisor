"""
ParallelAdvisor Environment for Verifiers.

Main environment that:
1. Presents serial C++ code + hardware spec
2. Gets LLM recommendation (strategy + thread count + code)
3. Compiles and benchmarks the result
4. Returns reward based on correctness + speedup
"""

import verifiers as vf
from typing import Optional
from .dataset import load_dataset, CodeSample
from .rubric import ParallelAdvisorRubric


SYSTEM_PROMPT = """You are an expert parallel programming advisor. Given serial C++ code and hardware specifications, you recommend the optimal parallelization strategy.

Your options are:
- SERIAL: Keep the code serial (parallelization overhead would hurt performance)
- THREADS(N): Use std::thread with N threads
- OPENMP(N): Use OpenMP with N threads
- CUDA: Use GPU acceleration (when GPU is available)

Consider:
- Problem size vs. thread/kernel launch overhead
- Data dependencies and race conditions
- Memory access patterns (coalescing for GPU, cache locality for CPU)
- Data transfer overhead (CPU↔GPU memory copies)
- Hardware capabilities (CPU cores, GPU compute capability)
- Whether the workload is compute-bound or memory-bound

For CUDA recommendations:
- Consider data transfer time vs. computation time
- Ensure sufficient parallelism to saturate GPU (thousands of threads)
- Memory-bound kernels may not benefit much from GPU
- Small problem sizes often perform better on CPU

Always provide working, compilable code."""


USER_TEMPLATE = """## Hardware Specification
- CPU Cores: {cores}
- CPU Threads: {threads}
- GPU Available: {gpu_available}
- GPU: {gpu_name}

## Serial C++ Code
```cpp
{code}
```

## Task
1. Analyze this code for parallelization opportunities
2. Recommend the best strategy: SERIAL | THREADS(N) | OPENMP(N) | CUDA
3. Explain your reasoning briefly (consider data transfer overhead for GPU)
4. Generate the complete parallelized code (or original if SERIAL)

## Output Format
RECOMMENDATION: <strategy>
THREADS: <N or N/A for CUDA/SERIAL>
REASONING: <1-3 sentences explaining why this strategy is optimal>
CODE:
```cpp
<complete compilable code>
```

Note: For CUDA, generate a complete .cu file with kernel and host code."""


class ParallelAdvisorEnv(vf.SingleTurnEnv):
    """
    Single-turn environment for parallelization recommendation.
    
    Input: Serial C++ code + hardware spec
    Output: Strategy recommendation + parallelized code
    Reward: Compilation success + correctness + speedup
    """
    
    def __init__(
        self,
        dataset: list[CodeSample],
        rubric: ParallelAdvisorRubric,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        # Format dataset for verifiers
        formatted_dataset = [
            {
                "prompt": USER_TEMPLATE.format(
                    cores=sample.hardware.cores,
                    threads=sample.hardware.threads,
                    gpu_available="Yes" if sample.hardware.gpu else "No",
                    gpu_name=sample.hardware.gpu_name or "N/A",
                    code=sample.code,
                ),
                "sample": sample,  # Pass full sample for rubric
            }
            for sample in dataset
        ]
        
        super().__init__(
            dataset=formatted_dataset,
            rubric=rubric,
            system_prompt=system_prompt,
        )


def load_environment(
    dataset_name: str = "basic",
    hardware_cores: int = 8,
    hardware_threads: int = 16,
    gpu_available: bool = False,
    gpu_name: str = None,
) -> vf.Environment:
    """
    Load the ParallelAdvisor environment.
    
    Args:
        dataset_name: Which dataset to load ("basic", "gpu", "full")
        hardware_cores: Number of CPU cores to report to the model
        hardware_threads: Number of CPU threads to report to the model
        gpu_available: Whether a GPU is available
        gpu_name: GPU model name (e.g., "NVIDIA A100")
    
    Returns:
        Configured ParallelAdvisor environment
    """
    from .dataset import HardwareSpec
    
    # Override hardware spec for all samples
    hardware = HardwareSpec(
        cores=hardware_cores,
        threads=hardware_threads,
        gpu=gpu_available,
        gpu_name=gpu_name,
    )
    dataset = load_dataset(dataset_name, hardware_override=hardware)
    
    rubric = ParallelAdvisorRubric(gpu_available=gpu_available)
    
    return ParallelAdvisorEnv(
        dataset=dataset,
        rubric=rubric,
    )
