#!/usr/bin/env python3
"""Run comprehensive benchmarks for Bit-Axon model.

Usage:
    python scripts/benchmark.py --config-small
    python scripts/benchmark.py --seq-lengths 128 512 1024 2048
    python scripts/benchmark.py --seq-lengths 128 512 --iterations 10
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Benchmark Bit-Axon model")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--config-small", action="store_true", help="Use small config for testing")
    args = parser.parse_args()

    from bit_axon.config import BitAxonConfig
    from bit_axon.profiling.benchmark import BenchmarkSuite

    config = (
        BitAxonConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            d_source_model=128,
            vocab_size=1024,
        )
        if args.config_small
        else BitAxonConfig()
    )

    suite = BenchmarkSuite(config)
    print("Bit-Axon Benchmark Suite")
    print(f"Config: hidden_dim={config.hidden_dim}, layers={config.num_layers}, vocab={config.vocab_size}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Iterations: {args.warmup} warmup + {args.iterations} timed")
    print()

    results = suite.benchmark_sequence_lengths(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    )

    print(results.to_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
