"""
hardware.py - Describe the machine a run happened on, so results from different
machines (for example a Mac and a Kaggle GPU notebook) are never mixed up.
"""

import platform
import subprocess

import torch

# The hardware fields saved in every result JSON (under "hardware") and in results.csv.
HARDWARE_COLUMNS = ["system", "cpu_name", "gpu_name", "gpu_count", "torch_version", "cuda_version"]


def cpu_name():
    """CPU model, e.g. 'Apple M4' or 'Intel(R) Xeon(R) CPU @ 2.00GHz'. None if unknown."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            return out or None
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


def gpu_name():
    """Name(s) of the NVIDIA GPUs, 'Apple GPU (MPS)' on a Mac, or None if there is no GPU."""
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return " + ".join(sorted(set(names)))  # e.g. 'Tesla T4' for two identical GPUs
    if torch.backends.mps.is_available():
        return "Apple GPU (MPS)"
    return None


def hardware_info():
    """A dict with the HARDWARE_COLUMNS fields for this machine."""
    return {
        "system": f"{platform.system()}-{platform.machine()}",  # e.g. 'Darwin-arm64', 'Linux-x86_64'
        "cpu_name": cpu_name(),
        "gpu_name": gpu_name(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,  # None for CPU-only or Mac builds of PyTorch
    }


if __name__ == "__main__":
    for key, value in hardware_info().items():
        print(f"{key:14s} {value}")
