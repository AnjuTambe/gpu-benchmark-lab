# GPU Benchmark Lab

A small tool that measures how fast a tiny PyTorch model trains on a CPU or on one or
more GPUs, breaks training on purpose, and explains what went wrong.

It runs on a laptop CPU first (developed on a Mac with no NVIDIA GPU) and uses a GPU
automatically when one is available. It has been run on an Apple M4 Mac (CPU) and on
Kaggle with 2 × NVIDIA Tesla T4. All numbers below come from real runs saved in this repo.

## What it does

- **Benchmark**: trains a 3-layer MLP on fake data and reports samples/sec, p50/p95 step
  time, all-reduce time and scaling efficiency, for 1 or more processes with
  DistributedDataParallel (gloo on CPU, NCCL on CUDA).
- **Failures on purpose**: out of memory, a crashing worker, a NaN batch and a broken
  distributed config, plus an analyzer that reads the log and says what happened.
- **Regression gate**: compares a run with a saved baseline and exits with code 1 if
  throughput dropped, a run failed, or the hardware doesn't match.
- **Profiling**: splits each step into forward / backward / optimizer with
  `torch.profiler` and measures how much all-reduce overlaps backward.

## Results at a glance

Batch size is per process. Scaling efficiency = samples/sec ÷ (processes × 1-process
samples/sec). Median of 3 repeats.

**Apple M4 Mac, CPU** (2 threads per process; `baselines/cpu_mac.csv`)

| Processes | Batch | Samples/s | p50 step | All-reduce | Scaling |
|---|---|---|---|---|---|
| 1 | 128 | 25,232.9 | 5.044 ms | — | 100% |
| 2 | 128 | 14,126.4 | 18.110 ms | 3.692 ms | 28.0% |
| 1 | 256 | 35,607.4 | 7.167 ms | — | 100% |
| 2 | 256 | 22,476.5 | 22.526 ms | 3.724 ms | 31.6% |

**Kaggle, 2 × Tesla T4** (`baselines/kaggle_t4x2.csv`)

| GPUs | Batch | Samples/s | p50 step | All-reduce | Scaling |
|---|---|---|---|---|---|
| 1 | 128 | 46,009.1 | 2.693 ms | — | 100% |
| 2 | 128 | 41,261.9 | 6.043 ms | 3.162 ms | 44.8% |
| 1 | 256 | 62,078.2 | 3.818 ms | — | 100% |
| 2 | 256 | 67,425.2 | 7.249 ms | 3.166 ms | 54.3% |

What the profiles (batch 256, `results/profile/`) show:

- **This model is too small to gain much from a second device.** Scaling never gets above
  54%; at batch 128, two GPUs are slower in total than one, and on the Mac CPU two
  processes are slower than one at both batch sizes.
- **On the Mac CPU**, about 7.7 ms of each 2-process step was the main thread waiting
  inside backward for the gradient all-reduce. The rest of the slowdown came mostly from
  the two processes competing for the same CPU cores (slower math in forward and backward).
- **On the T4 GPUs**, backward grew by about 3.5 ms with 2 GPUs, and the NCCL all-reduce
  kernels ran for 3.45 ms inside backward: the extra time is almost all communication.
- **On one GPU, the GPU waits for data.** Building each batch with the DataLoader on the
  CPU took about 2.8 ms per profiled step, more than forward and backward together.

Profiled steps are slower than normal ones because of profiler overhead, so treat profile
percentages as approximate.

## Quick start (CPU, macOS or Linux)

```bash
git clone https://github.com/AnjuTambe/gpu-benchmark-lab.git
cd gpu-benchmark-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py            # one run: prints throughput and step times
pytest -q                  # unit tests (no GPU or torchrun needed)
```

`train.py` picks CUDA, then Apple MPS, then CPU. Force a device with `--device cpu|mps|cuda`.

## Usage

### Two processes with DDP

```bash
# macOS: these extra settings avoid a hostname lookup problem in torchrun/gloo
OMP_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 train.py --device cpu
```

On Linux, `GLOO_SOCKET_IFNAME=lo0` is not needed. Useful `train.py` options:
`--steps`, `--batch-size`, `--sync-mode sync|no_sync|no_ddp` (to isolate communication
cost), `--profile`, `--failure-mode ...`.

### Sweep many settings with one command

```bash
python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cpu
```

Runs every combination through `torchrun` (3 repeats each, a timeout per run) and saves
everything in a new `results/<timestamp>/` folder with a `results.csv`. Combinations the
machine can't run (for example 2 GPUs on a 1-GPU machine) are skipped and listed.

### Failure demo

```bash
python benchmark.py --failure-demo --device cpu
```

```
FAILURE DETECTED
Type: CUDA_OUT_OF_MEMORY
Rank: 1
Reason: The GPU ran out of memory. Log line: [rank1]: torch.OutOfMemoryError: CUDA out of memory. ...
Suggested action: Reduce --batch-size or the model size. ...
```

Types: `CUDA_OUT_OF_MEMORY`, `OUT_OF_MEMORY`, `WORKER_CRASH`, `NAN_LOSS`,
`DISTRIBUTED_CONFIG_ERROR`, `TIMEOUT`, and `UNKNOWN` when no rule matches (the analyzer
never guesses). On CUDA, the out-of-memory case is a real allocation failure; on CPU it
is simulated with a realistic message.

### Regression gate

```bash
python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cpu --save-baseline baselines/my_machine.csv
python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cpu --baseline baselines/my_machine.csv
echo $?    # 0 = pass, 1 = regression, failed run, or nothing could be compared
```

Rows are only compared with a baseline from the same machine (system, CPU and GPU name).
The default tolerance is 15% (`--tolerance`), because laptop CPU numbers are noisy.

### Profiling

```bash
OMP_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 \
  train.py --device cpu --profile --out results/profile/cpu_p2_bs256_sync.json
```

Writes a text summary and a Chrome trace per process. See
[docs/profiling.md](docs/profiling.md) for how to read them.

## Running on a GPU

[kaggle_run.md](kaggle_run.md) has copy-paste notebook cells for a free Kaggle
`GPU T4 x2` notebook: setup, the GPU benchmark, profiling, the failure demo, and
downloading the results.

## Project layout

| File | What it does |
|---|---|
| `train.py` | The model, training loop, timing, DDP, `--profile` and `--failure-mode` |
| `benchmark.py` | Runs sweeps through torchrun, the failure demo and the regression gate |
| `analyzer.py` | Classifies a failed run from its exit code and log |
| `regression.py` | Compares results with a baseline |
| `profile_stats.py` | Turns profiler output into the forward/backward/optimizer split |
| `hardware.py` | Records the machine (system, CPU, GPU, torch and CUDA versions) |
| `baselines/` | Saved baselines: `cpu_mac.csv`, `kaggle_t4x2.csv` |
| `results/` | Saved runs, sweeps, failure demos and profile summaries |
| `docs/profiling.md` | How to read profiles and open trace files |
| `tests/` | Unit tests with fake logs, CSVs and profiler events |

## Tests

```bash
pytest -q
```

The tests use small fake inputs and need neither a GPU nor `torchrun`. GitHub Actions
runs them on every push with CPU-only PyTorch (`.github/workflows/tests.yml`).

## Limitations

- The model is tiny and the data is random, on purpose: this measures overheads
  (communication, data loading, process contention), not real model training.
- Tested on an Apple M4 Mac (CPU) and on Kaggle 2 × Tesla T4 only. More than 2 GPUs,
  other GPU types and multi-machine runs have not been tested.
- Nsight Systems (`nsys`) is mentioned in the docs but has not been run.
