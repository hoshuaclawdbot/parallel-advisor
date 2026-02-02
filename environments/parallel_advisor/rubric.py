"""
Rubric (reward function) for ParallelAdvisor.

Scores LLM output based on:
1. Compilation success
2. Correctness (output matches serial baseline)
3. Speedup achieved
"""

import re
import verifiers as vf
from typing import Optional
from dataclasses import dataclass
from .compiler import compile_and_run, CompileResult


@dataclass
class ParsedRecommendation:
    """Parsed LLM recommendation."""
    strategy: str  # "serial", "threads", "openmp"
    threads: Optional[int]
    reasoning: str
    code: str
    parse_error: Optional[str] = None


def parse_recommendation(completion: str) -> ParsedRecommendation:
    """
    Parse LLM output into structured recommendation.
    
    Expected format:
    RECOMMENDATION: <strategy>
    THREADS: <N or N/A>
    REASONING: <text>
    CODE:
    ```cpp or ```cuda
    <code>
    ```
    """
    try:
        # Extract recommendation
        rec_match = re.search(r'RECOMMENDATION:\s*(SERIAL|THREADS\(\d+\)|OPENMP\(\d+\)|CUDA)', completion, re.IGNORECASE)
        if not rec_match:
            return ParsedRecommendation(
                strategy="unknown",
                threads=None,
                reasoning="",
                code="",
                parse_error="Could not find RECOMMENDATION line"
            )
        
        rec_str = rec_match.group(1).upper()
        
        # Parse strategy and threads
        if rec_str == "SERIAL":
            strategy = "serial"
            threads = None
        elif rec_str.startswith("THREADS"):
            strategy = "threads"
            threads = int(re.search(r'\((\d+)\)', rec_str).group(1))
        elif rec_str.startswith("OPENMP"):
            strategy = "openmp"
            threads = int(re.search(r'\((\d+)\)', rec_str).group(1))
        elif rec_str == "CUDA":
            strategy = "cuda"
            threads = None
        else:
            strategy = "unknown"
            threads = None
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=CODE:|$)', completion, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract code (try cpp, cuda, c++, or no specifier)
        code_match = re.search(r'```(?:cpp|cuda|c\+\+)?\s*\n(.*?)\n```', completion, re.DOTALL)
        if not code_match:
            # Try without language specifier
            code_match = re.search(r'```\s*\n(.*?)\n```', completion, re.DOTALL)
        
        code = code_match.group(1).strip() if code_match else ""
        
        return ParsedRecommendation(
            strategy=strategy,
            threads=threads,
            reasoning=reasoning,
            code=code,
        )
    
    except Exception as e:
        return ParsedRecommendation(
            strategy="unknown",
            threads=None,
            reasoning="",
            code="",
            parse_error=str(e)
        )


class ParallelAdvisorRubric(vf.Rubric):
    """
    Rubric for scoring parallelization recommendations.
    
    Reward structure:
    - 0.0: Failed to parse or compile
    - 0.3: Compiled but wrong output
    - 0.5: Correct output, no speedup
    - 0.5-1.0: Correct + speedup (linear scale)
    """
    
    def __init__(
        self,
        compile_weight: float = 0.2,
        correctness_weight: float = 0.3,
        speedup_weight: float = 0.5,
        max_speedup_reward: float = 10.0,  # Speedup of 10x = max reward
        gpu_available: bool = False,
    ):
        self.compile_weight = compile_weight
        self.correctness_weight = correctness_weight
        self.speedup_weight = speedup_weight
        self.max_speedup_reward = max_speedup_reward
        self.gpu_available = gpu_available
        
        super().__init__(funcs=[self.score_recommendation])
    
    async def score_recommendation(
        self,
        completion: list[dict],  # Chat completion messages
        sample: "CodeSample",     # From dataset
    ) -> float:
        """
        Score a parallelization recommendation.
        
        Args:
            completion: LLM completion (list of messages)
            sample: Original code sample with metadata
        
        Returns:
            Reward score between 0.0 and 1.0
        """
        # Get the assistant's response
        response_text = ""
        for msg in completion:
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                break
        
        if not response_text:
            return 0.0
        
        # Parse the recommendation
        rec = parse_recommendation(response_text)
        
        if rec.parse_error or not rec.code:
            return 0.0
        
        # Compile and run
        result = await compile_and_run(
            code=rec.code,
            strategy=rec.strategy,
            threads=rec.threads,
            original_code=sample.code,
            gpu_available=self.gpu_available,
        )
        
        if not result.compiled:
            return 0.0
        
        reward = self.compile_weight
        
        if result.correct:
            reward += self.correctness_weight
            
            # Speedup bonus
            if result.speedup and result.speedup > 1.0:
                speedup_bonus = min(
                    self.speedup_weight,
                    self.speedup_weight * (result.speedup - 1.0) / (self.max_speedup_reward - 1.0)
                )
                reward += speedup_bonus
        
        return min(1.0, reward)
