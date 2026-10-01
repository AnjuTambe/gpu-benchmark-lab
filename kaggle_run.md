# Running on a free Kaggle GPU

> **Not tested on a GPU yet.** Everything in this project was developed and run on a
> Mac without an NVIDIA GPU. The CUDA/NCCL code paths below have never actually run.
> Treat this first Kaggle run as the test, and expect to fix things.

## Before you start (once)

1. Push this project to GitHub (see the end of the main conversation / README).
2. On <https://www.kaggle.com>, create a **New Notebook**.
3. In the notebook's right-hand panel, **Settings**:
   - **Accelerator**: `GPU T4 x2` (two GPUs, so the 2-process runs can happen).
     `GPU P100` has only one GPU; then the 2-GPU rows are skipped.
   - **Internet**: **On** (needed to clone from GitHub; Kaggle requires a
     phone-verified account for this).
4. Paste each cell below into its own notebook cell and run them in order.

If your GitHub repo is **private**, cloning needs a token. The simplest option is to
make the repo public, or add a GitHub token as a Kaggle **Secret** and use
`https://<token>@github.com/...` in the clone URL.

---

### Cell 1: check the GPU and PyTorch

```python
!nvidia-smi
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch build):", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
```

You should see `CUDA available: True` and `GPU count: 2` on `GPU T4 x2`.

### Cell 2: get the code

```python
%cd /kaggle/working
!git clone https://github.com/AnjuTambe/gpu-benchmark-lab.git
%cd /kaggle/working/gpu-benchmark-lab
!git log --oneline | head -3
```

### Cell 3: install requirements, keeping Kaggle's PyTorch

Kaggle already has a CUDA build of PyTorch. This installs everything in
`requirements.txt` **except** torch, so it is not downloaded again.

```python
!grep -v -E '^torch([<>=!~ ]|$)' requirements.txt > /tmp/requirements-no-torch.txt
!cat /tmp/requirements-no-torch.txt
!pip install -q -r /tmp/requirements-no-torch.txt
!python -c "import torch; print('torch still', torch.__version__, 'cuda', torch.version.cuda)"
```

### Cell 4: machine info and tests

```python
!python hardware.py
!pytest -q
```

`hardware.py` prints what is saved with every result (GPU name, GPU count, torch and
CUDA versions), so Kaggle results are never mixed up with Mac results.

### Cell 5: the GPU benchmark

```python
!python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cuda
```

This runs every combination 3 times (1 GPU and 2 GPUs, batch 128 and 256) with NCCL,
and prints a table. With only 1 GPU, the 2-GPU rows are listed as
`skipped: needs 2 GPUs, only 1 available`.

Optional: keep these numbers as a Kaggle baseline for later comparisons.

```python
!python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cuda --save-baseline baselines/kaggle_t4x2.csv
```

### Cell 6: profile on the GPU

```python
!OMP_NUM_THREADS=2 torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 train.py --device cuda --batch-size 256 --profile --out results/profile/cuda_p1_bs256_sync.json
!OMP_NUM_THREADS=2 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 train.py --device cuda --batch-size 256 --profile --out results/profile/cuda_p2_bs256_sync.json
!head -3 results/profile/cuda_p1_bs256_sync_rank0_profile.txt
!head -3 results/profile/cuda_p2_bs256_sync_rank0_profile.txt
```

See `docs/profiling.md` for how to read these and open the `_trace.json` files.

### Cell 7: the failure demo (includes a real CUDA out-of-memory error)

```python
!python benchmark.py --failure-demo --device cuda
```

On CUDA, `--failure-mode oom` really tries to allocate 1 PiB of GPU memory, so the
analyzer should report `CUDA_OUT_OF_MEMORY` from PyTorch's real error message.
(`bad-config` uses the gloo backend even on GPUs, so it fails with a clear timeout.)

### Cell 8: download everything

```python
%cd /kaggle/working
!zip -qr gpu-benchmark-results.zip gpu-benchmark-lab/results gpu-benchmark-lab/baselines
!ls -lh gpu-benchmark-results.zip
from IPython.display import FileLink
FileLink("gpu-benchmark-results.zip")
```

Click the link to download. The zip is also listed under **Output** (`/kaggle/working`)
in the right-hand panel. It includes the trace files, which are not in git.

---

## What to look at afterwards

- `results/<timestamp>/results.csv`: the table, now with `gpu_name`, `gpu_count`,
  `torch_version` and `cuda_version` columns.
- Any `FAILED` rows: open the matching `runs/*.log` file; the `failure_type` column
  has the analyzer's guess-free classification.
- The profile summaries: on a GPU, check whether the all-reduce numbers look sensible
  (this part of the code is the least certain; see "Untested" below).

## Untested on a GPU (as of this writing)

- NCCL process group setup and `torch.cuda.set_device` per process
- DDP on CUDA with 1 and 2 GPUs, and the CUDA timing synchronization
- The isolated all-reduce test (`allreduce_ms`) with NCCL
- `--profile` on CUDA: the per-range GPU waits, the CUDA activity in traces, and the
  detection of NCCL all-reduce kernels for the all-reduce numbers
- The real CUDA OOM in `--failure-mode oom` and its classification from a real log
- `worker-crash`, `bad-batch` and `bad-config` on CUDA
- The 2-GPU skip message on a 1-GPU machine
- `hardware.py` GPU name and CUDA version reporting
