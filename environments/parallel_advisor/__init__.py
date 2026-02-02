"""
ParallelAdvisor: LLM-based parallelization strategy recommendation.

Given serial C++ code and hardware specs, recommends:
- Serial (no parallelization)
- std::thread with N threads
- OpenMP with N threads
- CUDA (Phase 2)

Uses Prime Intellect's Verifiers framework for RL training.
"""

from .parallel_advisor import load_environment, ParallelAdvisorEnv
from .dataset import load_dataset, CodeSample
from .rubric import ParallelAdvisorRubric
from .compiler import compile_and_run, CompileResult

__all__ = [
    "load_environment",
    "ParallelAdvisorEnv", 
    "load_dataset",
    "CodeSample",
    "ParallelAdvisorRubric",
    "compile_and_run",
    "CompileResult",
]
