# Evaluation

::: bit_axon.evaluation
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - compute_perplexity
        - WikiTextDataset

## Benchmark Configuration

::: bit_axon.evaluation.tasks.BenchmarkConfig
    handler: python
    options:
      show_source: false

## Benchmark Registry

::: bit_axon.evaluation.tasks.BENCHMARK_REGISTRY
    handler: python
    options:
      show_source: false

## Benchmark Runner

::: bit_axon.evaluation.benchmark.evaluate_benchmark
    handler: python
    options:
      show_source: false

::: bit_axon.evaluation.benchmark.evaluate_benchmarks
    handler: python
    options:
      show_source: false

::: bit_axon.evaluation.benchmark.BenchmarkResult
    handler: python
    options:
      show_source: false
