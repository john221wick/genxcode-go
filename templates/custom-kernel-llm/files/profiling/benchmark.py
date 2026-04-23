import torch

from profiling.pytorch_profiler import PyTorchProfiler


class BenchmarkSuite:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.results = {}

    def run_all(self):
        from src.model import GPTModel

        backends = ["torch"]
        if self.device == "cuda":
            backends.extend(["triton"])

        for backend in backends:
            try:
                print(f"\n{'#' * 60}")
                print(f"# Benchmarking: {backend}")
                print(f"{'#' * 60}")

                cfg = self.cfg
                cfg.kernel_backend = backend
                model = GPTModel(cfg).to(self.device)
                profiler = PyTorchProfiler(model, self.device)
                self.results[backend] = profiler.benchmark(input_shape=(1, cfg.max_len))
            except Exception as e:
                print(f"  Skipping {backend}: {e}")
                self.results[backend] = {"error": str(e)}

        if self.device == "cuda":
            try:
                print(f"\n{'#' * 60}")
                print(f"# Benchmarking: torch.compile")
                print(f"{'#' * 60}")

                cfg = self.cfg
                cfg.kernel_backend = "torch"
                model = GPTModel(cfg).to(self.device)
                profiler = PyTorchProfiler(model, self.device)
                self.results["torch.compile"] = profiler.benchmark_with_compile(
                    input_shape=(1, cfg.max_len)
                )
            except Exception as e:
                print(f"  Skipping torch.compile: {e}")
                self.results["torch.compile"] = {"error": str(e)}

        self._print_summary()
        return self.results

    def _print_summary(self):
        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        for backend, result in self.results.items():
            if "error" in result:
                print(f"{backend:20s}: ERROR - {result['error']}")
            else:
                print(
                    f"{backend:20s}: {result['mean_ms']:.2f}ms +- {result['std_ms']:.2f}ms"
                )
