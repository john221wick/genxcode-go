import torch
import time
from pathlib import Path


class PyTorchProfiler:
    def __init__(self, model, device, output_dir="./prof_logs"):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile(self, dataloader, num_steps=10, wait=1, warmup=3, active=5):
        from torch.profiler import (
            profile,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        print(f"\n{'=' * 60}")
        print(f"PyTorch Profiler: {num_steps} steps, device={self.device}")
        print(f"{'=' * 60}\n")

        self.model.eval()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            schedule=schedule(
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(str(self.output_dir)),
        ) as prof:
            for step, batch in enumerate(dataloader):
                if step >= num_steps:
                    break
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                logits, loss = self.model(input_ids, targets)
                loss.backward()
                prof.step()

        print("\n--- Top 15 ops by CUDA time ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        print("\n--- Top 10 ops by self CUDA memory ---")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

        trace_path = self.output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\nChrome trace exported to {trace_path}")
        print("Open at chrome://tracing or https://ui.perfetto.dev")

        return prof

    def benchmark(self, input_shape=(1, 256), n_warmup=5, n_runs=20):
        print(f"\n{'=' * 60}")
        print(
            f"Benchmark: input_shape={input_shape}, n_warmup={n_warmup}, n_runs={n_runs}"
        )
        print(f"Device: {self.device}")
        print(f"{'=' * 60}\n")

        self.model.eval()
        input_ids = torch.randint(0, 50257, input_shape, device=self.device)

        print("Warming up...")
        for _ in range(n_warmup):
            with torch.no_grad():
                self.model(input_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    torch.cuda.synchronize()
                    start_event.record()
                    self.model(input_ids)
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))

            times_tensor = torch.tensor(times)
            mean_ms = times_tensor.mean().item()
            std_ms = times_tensor.std().item()
            print(f"Mean: {mean_ms:.2f}ms +- {std_ms:.2f}ms")
            print(
                f"Min: {times_tensor.min().item():.2f}ms, Max: {times_tensor.max().item():.2f}ms"
            )
        else:
            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    start = time.perf_counter()
                    self.model(input_ids)
                    if self.device == "mps":
                        torch.mps.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)

            mean_ms = sum(times) / len(times)
            std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
            print(f"Mean: {mean_ms:.2f}ms +- {std_ms:.2f}ms")

        return {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "n_runs": n_runs,
            "device": self.device,
        }

    def benchmark_with_compile(self, input_shape=(1, 256), n_warmup=5, n_runs=20):
        print(f"\n{'=' * 60}")
        print("Benchmark with torch.compile")
        print(f"{'=' * 60}\n")

        compiled_model = torch.compile(self.model)
        compiled_model.eval()
        input_ids = torch.randint(0, 50257, input_shape, device=self.device)

        print("Warming up (includes compilation)...")
        for _ in range(n_warmup + 3):
            with torch.no_grad():
                compiled_model(input_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()

        original_model = self.model
        self.model = compiled_model
        result = self.benchmark(input_shape, 0, n_runs)
        self.model = original_model
        return result

    def memory_stats(self):
        if self.device != "cuda":
            print(f"Memory stats not available on {self.device}")
            return {}
        stats = {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        print(f"\n--- GPU Memory Stats ---")
        print(f"Allocated: {stats['allocated_mb']:.1f} MB")
        print(f"Reserved:  {stats['reserved_mb']:.1f} MB")
        print(f"Peak:      {stats['max_allocated_mb']:.1f} MB")
        torch.cuda.reset_peak_memory_stats()
        return stats
