# Benchmarks

This project contains two kinds of [BenchmarkDotNet](https://benchmarkdotnet.org/) benchmarks:

- **Regression suite** (`Regression/`, category `regression`): benchmarks the actual FaceAiSharp code paths
  (end-to-end detection, inference + postprocessing, image→tensor conversion, non-max suppression,
  face alignment, embedding generation and embedding math, eye state detection). This suite is tracked
  over time in CI so performance improvements and regressions become visible.
- **Exploratory benchmarks** (everything else): one-off comparisons that informed past implementation
  choices (e.g. different image→tensor strategies, FaceONNX vs. ImageSharp alignment). They are kept for
  reference and can be run on demand, but they are not tracked.

## Running locally

```shell
# list everything
dotnet run -c Release -f net10.0 -- --list flat

# run the tracked regression suite
dotnet run -c Release -f net10.0 -- --anyCategories regression

# run a subset, quickly
dotnet run -c Release -f net10.0 -- --filter '*FaceDetection*' --job short
```

The required ONNX models are restored via the `FaceAiSharp.Bundle` project reference and test images are
copied from `examples/` to the build output, so no manual setup is needed.

## CI story

Benchmarks intentionally do **not** run on every push or PR — a full run takes a while and shared CI
runners are too noisy to use as a hard gate. Instead, `.github/workflows/benchmarks.yml` provides:

- **Continuous tracking on `master`:** pushes that touch `src/FaceAiSharp`, the benchmarks or the
  dependency versions run the regression suite. Results are appended to the `benchmarks` branch
  (`dev/bench/`) via [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark),
  which renders a chart page and posts a commit comment if a benchmark gets >1.75x slower.
- **On-demand runs:** trigger the workflow manually (workflow_dispatch) on any branch, e.g. to compare a
  perf PR against master before merging. Inputs allow choosing the category, a `--filter` glob and the
  BenchmarkDotNet job (`short` for a quick signal, `default` for accurate numbers). Manual runs upload
  the full BenchmarkDotNet artifacts but do not touch the tracked history.

When working on a performance change, the most reliable comparison is still a local run of the relevant
`--filter` on master vs. your branch on the same, otherwise idle machine.
