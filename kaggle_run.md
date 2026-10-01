# Running on a free Kaggle GPU

> **Tested on Kaggle `GPU T4 x2` on 2026-10-01** (PyTorch 2.10.0+cu128, CUDA 12.8).
> The results of that run are in `results/20261001-183623/` (benchmark),
> `results/20261001-185440-failure-demo/`, `results/profile/cuda_*` and
> `baselines/kaggle_t4x2.csv`. See "What the first run showed" at the end.

## Before you start (once)

1. On <https://www.kaggle.com>, **verify your phone number** (Settings → Phone
   Verification). Without it, GPUs and Internet are greyed out.
2. Create a notebook: **+ Create → New Notebook**.
3. In the notebook, **Settings** (top menu) or **Session options** (bottom of the
   right-hand panel):
   - **Accelerator**: `GPU T4 x2` (two GPUs, so the 2-process runs can happen).
     `GPU P100` has only one GPU; then the 2-GPU rows are skipped.
   - **Internet**: **On** (needed to clone from GitHub). If you just switched it on,
     restart the session (⏻ button) before it takes effect.
4. Paste each cell below into its own notebook cell (**+ Code**) and run it with
   **Shift + Enter**, in order.

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

**Option A, with Internet:**

```python
%cd /kaggle/working
!git clone https://github.com/AnjuTambe/gpu-benchmark-lab.git
%cd /kaggle/working/gpu-benchmark-lab
!git log --oneline | head -3
```

If this fails with `Could not resolve host: github.com`, Internet is not on for this
session. Either fix that (see above) or use option B.

**Option B, without Internet (this is what the first run used):**

1. On your computer, make a zip of the project. From the project folder:
   `git archive --format=zip --prefix=gpu-benchmark-lab/ -o ~/Desktop/gpu-benchmark-lab.zip HEAD`
2. In the notebook's right-hand panel: **Input → Upload → New Dataset**, drop the zip
   in, give it a name, **Create**, and wait for "ready to use".
3. Run this cell. It finds the newest uploaded copy of the code (the one that has
   `profile_stats.py`) and copies it to a writable folder, keeping any existing `results/`:

```python
import glob, os, shutil
new = [p for p in glob.glob("/kaggle/input/**/train.py", recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(p), "profile_stats.py"))]
print("Using:", new[0])
shutil.copytree(os.path.dirname(new[0]), "/kaggle/working/gpu-benchmark-lab", dirs_exist_ok=True)
%cd /kaggle/working/gpu-benchmark-lab
!ls
```

### Cell 3: install requirements, keeping Kaggle's PyTorch

Kaggle already has a CUDA build of PyTorch. **With Internet**, this installs everything
in `requirements.txt` **except** torch, so it is not downloaded again:

```python
!grep -v -E '^torch([<>=!~ ]|$)' requirements.txt > /tmp/requirements-no-torch.txt
!pip install -q -r /tmp/requirements-no-torch.txt
!python -c "import torch; print('torch still', torch.__version__, 'cuda', torch.version.cuda)"
```

**Without Internet**, skip the install and just check that everything is already there
(it was, on the first run):

```python
!python -c "import torch, numpy, pytest; print('torch', torch.__version__, '| numpy', numpy.__version__, '| pytest', pytest.__version__)"
```

### Cell 4: machine info and tests

```python
!python hardware.py
!pytest -q
```

`hardware.py` prints what is saved with every result (GPU name, GPU count, torch and
CUDA versions), so Kaggle results are never mixed up with Mac results. All tests
should pass.

### Cell 5: the GPU benchmark (and a Kaggle baseline)

```python
!python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cuda --save-baseline baselines/kaggle_t4x2.csv
```

This runs every combination 3 times (1 GPU and 2 GPUs, batch 128 and 256) with NCCL,
prints a table and saves it as the Kaggle baseline. It took about 10 minutes. With
only 1 GPU, the 2-GPU rows are listed as `skipped: needs 2 GPUs, only 1 available`.

To check a later run for slowdowns against that baseline:

```python
!python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cuda --baseline baselines/kaggle_t4x2.csv
```

### Cell 6: profile on the GPU

```python
!OMP_NUM_THREADS=2 torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 train.py --device cuda --batch-size 256 --profile --out results/profile/cuda_p1_bs256_sync.json
!OMP_NUM_THREADS=2 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 train.py --device cuda --batch-size 256 --profile --out results/profile/cuda_p2_bs256_sync.json
!head -3 results/profile/cuda_p1_bs256_sync_rank0_profile.txt
!head -3 results/profile/cuda_p2_bs256_sync_rank0_profile.txt
```

See `docs/profiling.md` for how to read these and open the `_trace.json` files.
Warnings like `The hostname of the client socket cannot be retrieved` are harmless.

### Cell 7: the failure demo (includes a real CUDA out-of-memory error)

```python
!python benchmark.py --failure-demo --device cuda
```

On CUDA, `--failure-mode oom` really tries to allocate 1 PiB of GPU memory, and the
analyzer reports `CUDA_OUT_OF_MEMORY` from PyTorch's real error message.
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

**When you are done, stop the session (⏻ button)** so it does not use up your weekly
GPU hours.

---

## What the first run showed (Kaggle 2 × Tesla T4, 2026-10-01)

Benchmark, median of 3 repeats (`results/20261001-183623/results.csv`):

| GPUs | Batch | Samples/s | p50 step | All-reduce (isolated) | Scaling |
|---|---|---|---|---|---|
| 1 | 128 | 46,009.1 | 2.693 ms | — | 100% |
| 2 | 128 | 41,261.9 | 6.043 ms | 3.162 ms | 44.8% |
| 1 | 256 | 62,078.2 | 3.818 ms | — | 100% |
| 2 | 256 | 67,425.2 | 7.249 ms | 3.166 ms | 54.3% |

- **2 GPUs barely help** for this tiny model: at batch 128 two GPUs are *slower* in
  total than one.
- **The extra time is gradient communication.** In the batch-256 profile, backward
  grew by 3.46 ms per step with 2 GPUs, and the NCCL all-reduce kernels ran for 3.45 ms
  inside backward. Forward and optimizer barely changed. (On the Mac CPU, by contrast,
  a large part of the slowdown came from the two processes competing for the CPU.)
- **The GPU waits for data.** In the 1-GPU profile, building each batch with the
  DataLoader on the CPU took 2.81 ms per profiled step, more than forward and backward
  together. Copying the batch to the GPU took only about 0.25 ms (CPU side).
- The failure demo classified all four failures correctly, including a **real** CUDA
  OOM (`Tried to allocate 1048576.00 GiB. GPU 1 has a total capacity of 14.56 GiB`).

Profiled steps are slower than normal steps (profiler overhead plus a GPU wait at the
end of each range), so read the profile percentages as approximate.

## Tested and untested on a GPU

Tested on Kaggle 2 × T4: NCCL setup with 1 and 2 GPUs, DDP and timing on CUDA, the
NCCL all-reduce test, `--profile` on CUDA (after fixing a bug where GPU-side copies of
the ranges showed as 0%), detection of NCCL all-reduce kernels, the real CUDA OOM and
the other three failure modes, `hardware.py`, and the regression baseline file.

Still untested:

- The 2-GPU skip message on a 1-GPU machine (e.g. `GPU P100`)
- Comparing a later Kaggle run against `baselines/kaggle_t4x2.csv` with `--baseline`
- More than 2 GPUs, other GPU types, and multi-machine runs
- Nsight Systems (`nsys`), see `docs/profiling.md`
