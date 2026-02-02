"""
Compilation and execution utilities for ParallelAdvisor.

Handles:
- Compiling serial, threaded, and OpenMP code
- Running and timing execution
- Comparing outputs for correctness
"""

import asyncio
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class CompileResult:
    """Result of compilation and execution."""
    compiled: bool
    compile_error: Optional[str] = None
    ran: bool = False
    run_error: Optional[str] = None
    output: Optional[str] = None
    time_ms: Optional[float] = None
    correct: bool = False
    speedup: Optional[float] = None


def get_compile_command(filepath: str, strategy: str, output: str) -> list[str]:
    """Get compilation command for the given strategy."""
    if strategy == "cuda":
        # CUDA compilation with nvcc
        return [
            "nvcc",
            "-O3",
            "-std=c++17",
            "--gpu-architecture=sm_70",  # Volta and newer
            filepath,
            "-o", output
        ]
    
    base_cmd = ["g++", "-O3", "-std=c++17"]
    
    if strategy == "openmp":
        base_cmd.append("-fopenmp")
    elif strategy == "threads":
        base_cmd.append("-pthread")
    # serial needs no extra flags
    
    return base_cmd + [filepath, "-o", output]


def get_file_extension(strategy: str) -> str:
    """Get appropriate file extension for the strategy."""
    return ".cu" if strategy == "cuda" else ".cpp"


def extract_time_from_output(output: str) -> Optional[float]:
    """Extract timing from program output (expects 'Time: X ms' format)."""
    match = re.search(r'Time:\s*([\d.]+)\s*ms', output)
    if match:
        return float(match.group(1))
    return None


def extract_result_from_output(output: str) -> str:
    """Extract the result value (non-timing lines) for correctness comparison."""
    lines = output.strip().split('\n')
    result_lines = [line for line in lines if 'Time:' not in line]
    return '\n'.join(result_lines)


async def compile_and_run(
    code: str,
    strategy: str,
    threads: Optional[int],
    original_code: str,
    timeout_seconds: float = 60.0,
    gpu_available: bool = False,
) -> CompileResult:
    """
    Compile and run the parallelized code, comparing with original.
    
    Args:
        code: Parallelized code to test
        strategy: "serial", "threads", "openmp", or "cuda"
        threads: Number of threads (for threads/openmp)
        original_code: Original serial code for comparison
        timeout_seconds: Max time for compilation + execution
        gpu_available: Whether GPU compilation should be attempted
    
    Returns:
        CompileResult with compilation status, correctness, and speedup
    """
    result = CompileResult(compiled=False)
    
    # Check if CUDA requested but no GPU
    if strategy == "cuda" and not gpu_available:
        result.compile_error = "CUDA requested but no GPU available"
        return result
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write parallel code with appropriate extension
        ext = get_file_extension(strategy)
        parallel_src = os.path.join(tmpdir, f"parallel{ext}")
        parallel_bin = os.path.join(tmpdir, "parallel")
        with open(parallel_src, "w") as f:
            f.write(code)
        
        # Write original code
        serial_src = os.path.join(tmpdir, "serial.cpp")
        serial_bin = os.path.join(tmpdir, "serial")
        with open(serial_src, "w") as f:
            f.write(original_code)
        
        try:
            # Compile parallel version
            compile_cmd = get_compile_command(parallel_src, strategy, parallel_bin)
            proc = await asyncio.create_subprocess_exec(
                *compile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            
            if proc.returncode != 0:
                result.compile_error = stderr.decode()
                return result
            
            result.compiled = True
            
            # Compile serial version
            serial_compile_cmd = get_compile_command(serial_src, "serial", serial_bin)
            proc = await asyncio.create_subprocess_exec(
                *serial_compile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30.0)
            
            # Set thread count for OpenMP
            env = os.environ.copy()
            if strategy == "openmp" and threads:
                env["OMP_NUM_THREADS"] = str(threads)
            
            # Run serial version
            proc = await asyncio.create_subprocess_exec(
                serial_bin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), 
                timeout=timeout_seconds
            )
            serial_output = stdout.decode()
            serial_time = extract_time_from_output(serial_output)
            serial_result = extract_result_from_output(serial_output)
            
            # Run parallel version
            proc = await asyncio.create_subprocess_exec(
                parallel_bin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds
            )
            
            if proc.returncode != 0:
                result.run_error = stderr.decode()
                result.ran = False
                return result
            
            result.ran = True
            parallel_output = stdout.decode()
            result.output = parallel_output
            
            parallel_time = extract_time_from_output(parallel_output)
            parallel_result = extract_result_from_output(parallel_output)
            
            result.time_ms = parallel_time
            
            # Check correctness
            result.correct = (parallel_result.strip() == serial_result.strip())
            
            # Calculate speedup
            if serial_time and parallel_time and parallel_time > 0:
                result.speedup = serial_time / parallel_time
            
        except asyncio.TimeoutError:
            result.run_error = "Execution timed out"
        except Exception as e:
            result.run_error = str(e)
    
    return result


# Synchronous wrapper for non-async contexts
def compile_and_run_sync(
    code: str,
    strategy: str,
    threads: Optional[int],
    original_code: str,
    timeout_seconds: float = 60.0,
) -> CompileResult:
    """Synchronous wrapper for compile_and_run."""
    return asyncio.run(compile_and_run(
        code=code,
        strategy=strategy,
        threads=threads,
        original_code=original_code,
        timeout_seconds=timeout_seconds,
    ))
