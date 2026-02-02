#!/usr/bin/env python3
"""
Evaluate baseline LLM performance on ParallelAdvisor.

Usage:
    python scripts/eval_baseline.py --model gpt-4o-mini
    python scripts/eval_baseline.py --model claude-3-5-sonnet-20241022
"""

import argparse
import asyncio
import json
from datetime import datetime

# Add parent to path for local development
import sys
sys.path.insert(0, "environments")

from parallel_advisor import load_environment


async def main():
    parser = argparse.ArgumentParser(description="Evaluate ParallelAdvisor baseline")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to evaluate")
    parser.add_argument("--dataset", default="basic", help="Dataset to use")
    parser.add_argument("--cores", type=int, default=8, help="CPU cores to report")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads to report")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Loading environment with dataset='{args.dataset}'...")
    env = load_environment(
        dataset_name=args.dataset,
        hardware_cores=args.cores,
        hardware_threads=args.threads,
    )
    
    print(f"Evaluating model: {args.model}")
    print(f"Hardware: {args.cores} cores, {args.threads} threads")
    print("-" * 60)
    
    # Note: In practice, you'd use `prime eval run` or the Verifiers API
    # This is a placeholder for manual testing
    
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "hardware": {"cores": args.cores, "threads": args.threads},
        "timestamp": datetime.now().isoformat(),
        "samples": [],
    }
    
    print("\nTo run actual evaluation, use:")
    print(f"  prime eval run parallel-advisor -m {args.model}")
    print("\nOr use the Verifiers Python API directly.")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
