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

Consider:
- Problem size vs. thread overhead
- Data dependencies and race conditions
- Memory access patterns
- Hardware capabilities

Always provide working, compilable code."""


USER_TEMPLATE = """## Hardware Specification
- CPU Cores: {cores}
- CPU Threads: {threads}

## Serial C++ Code
```cpp
{code}
```

## Task
1. Analyze this code for parallelization opportunities
2. Recommend the best strategy: SERIAL | THREADS(N) | OPENMP(N)
3. Explain your reasoning briefly
4. Generate the complete parallelized code (or original if SERIAL)

## Output Format
RECOMMENDATION: <strategy>
THREADS: <N or N/A>
REASONING: <1-2 sentences>
CODE:
```cpp
<complete compilable code>
```"""


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
) -> vf.Environment:
    """
    Load the ParallelAdvisor environment.
    
    Args:
        dataset_name: Which dataset to load ("basic", "polybench", "custom")
        hardware_cores: Number of CPU cores to report to the model
        hardware_threads: Number of CPU threads to report to the model
    
    Returns:
        Configured ParallelAdvisor environment
    """
    from .dataset import HardwareSpec
    
    # Override hardware spec for all samples
    hardware = HardwareSpec(cores=hardware_cores, threads=hardware_threads)
    dataset = load_dataset(dataset_name, hardware_override=hardware)
    
    rubric = ParallelAdvisorRubric()
    
    return ParallelAdvisorEnv(
        dataset=dataset,
        rubric=rubric,
    )
