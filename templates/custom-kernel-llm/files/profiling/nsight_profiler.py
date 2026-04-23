import torch


class NsightProfiler:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def nsys_command(output="trace", trace_cuda=True, trace_nvtx=True, trace_osrt=True):
        cmd = ["nsys", "profile", f"-o={output}"]
        traces = []
        if trace_cuda:
            traces.append("cuda")
        if trace_nvtx:
            traces.append("nvtx")
        if trace_osrt:
            traces.append("osrt")
        if traces:
            cmd.append(f"--trace={','.join(traces)}")
        cmd.append("--stats=true")
        cmd.append("python")
        return " ".join(cmd)

    @staticmethod
    def ncu_command(kernel_name=None, skip=5, count=3):
        cmd = ["ncu", "--set", "full"]
        if kernel_name:
            cmd.extend(
                [
                    "--kernel-name",
                    f"'*{kernel_name}*'",
                    "--launch-skip",
                    str(skip),
                    "--launch-count",
                    str(count),
                ]
            )
        cmd.append("python")
        return " ".join(cmd)

    def annotate(self, name, color="blue"):
        return NVTXRange(name, self.device)

    def annotate_push(self, name):
        if self.device == "cuda":
            torch.cuda.nvtx.range_push(name)

    def annotate_pop(self):
        if self.device == "cuda":
            torch.cuda.nvtx.range_pop()


class NVTXRange:
    def __init__(self, name, device):
        self.name = name
        self.device = device

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *args):
        if self.device == "cuda":
            torch.cuda.nvtx.range_pop()
