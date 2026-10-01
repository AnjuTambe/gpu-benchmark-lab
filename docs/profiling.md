# Profiling guide

`train.py --profile` records 10 training steps with `torch.profiler` and saves two
files per process, next to the run's result JSON:

| File | What it is |
|---|---|
| `<name>_rank<N>_profile.txt` | Text summary: step-time split, all-reduce numbers, top operations by time |
| `<name>_rank<N>_trace.json` | Timeline of every operation (Chrome trace format) |

The profiled steps run **after warmup and before the timed run**, so the profiler's
own overhead does not change the throughput numbers. Profiled steps are a little
slower than normal ones (about 10% in our CPU runs) because of that overhead.

## Run it

```bash
source .venv/bin/activate

# 1 process
OMP_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=29500 \
  train.py --device cpu --batch-size 256 --profile --out results/profile/cpu_p1_bs256_sync.json

# 2 processes (each rank writes its own profile files)
OMP_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 \
  train.py --device cpu --batch-size 256 --profile --out results/profile/cpu_p2_bs256_sync.json
```

The step-time split from rank 0 is also saved in the result JSON under `"profile"`.

## The three parts of a training step

Each step is wrapped in named ranges, so they show up in the summary and the trace:

- **forward**: the model looks at a batch and makes predictions, and the loss
  (how wrong the predictions are) is computed.
- **backward**: PyTorch works out, for every weight, which direction to nudge it to
  make the loss smaller (the *gradients*). With DDP and more than one process, the
  gradients are also **averaged across processes** here (the all-reduce), so every
  process ends up with the same update.
- **optimizer**: the weights are actually nudged using the gradients, and the
  gradients are cleared for the next step.

Anything outside these three (loading the batch, the NaN check) is reported as **other**.

## Reading the summary (`_profile.txt`)

The first lines are the most useful:

```
Average step: 25.465 ms  |  forward 21.1%  backward 71.1%  optimizer 5.2%  other 2.7%
All-reduce: {"calls_per_step": 2.0, ..., "busy_ms_per_step_during_backward": 8.8, ...}
```

Below that is PyTorch's table of operations. The columns that matter most:

- **CPU total**: time spent in this operation *including* everything it called.
  `backward` CPU total = the whole backward pass.
- **Self CPU**: time spent in this operation *itself*, not in anything it called.
  A large Self CPU on a range like `backward` means time where nothing else was
  recorded on that thread, i.e. the thread was **waiting**.
- **# of Calls**: how many times it ran in the 10 profiled steps. Divide totals by 10
  for per-step numbers.

Common operation names: `aten::addmm` / `aten::mm` are matrix multiplications (the
real math of the linear layers); `aten::copy_` is copying data;
`gloo:all_reduce` is the gradient averaging between processes.

### All-reduce numbers (2 or more processes only)

With the gloo backend, DDP *starts* each all-reduce on the main thread
(`c10d::allreduce_`, a few microseconds) and the real work runs on gloo's own
background threads (`gloo:all_reduce`) **at the same time** as the backward math.

- `calls_per_step`: how many all-reduces per step. DDP groups gradients into buckets
  and sends one all-reduce per bucket.
- `work_ms_per_step_summed`: all-reduce durations added up. Buckets can run at the
  same time on different threads, so this can be more than the wall time.
- `busy_ms_per_step_during_backward`: wall time during backward while at least one
  all-reduce was running. It overlaps with the backward math, so it is **not** simply
  added on top of the step time.

## Opening the trace (`_trace.json`)

Either of these works; both run locally in your browser:

- **Perfetto** (recommended): go to <https://ui.perfetto.dev>, click **Open trace file**,
  pick the `_trace.json` file.
- **Chrome**: open `chrome://tracing` in Chrome, click **Load**, pick the file.

What you see: one row per thread, time going left to right. Use **W/S** to zoom in
and out and **A/D** to move left and right. Click a block to see its name and duration.

- The main thread shows `ProfilerStep#N` blocks, each containing `forward`,
  `backward` and `optimizer`.
- With 2 processes, gloo's background threads show `gloo:all_reduce` blocks that
  overlap with `backward`.
- An empty gap inside `backward` on the main thread, lined up with the end of the last
  `gloo:all_reduce`, is the main thread waiting for the gradients to arrive.

## Optional: Nsight Systems (NVIDIA GPUs only)

On a machine with an NVIDIA GPU, NVIDIA's `nsys` (Nsight Systems) can show GPU kernels,
CUDA memory copies and NCCL communication on one timeline, in more detail than
`torch.profiler`. A typical command would be:

```bash
nsys profile -o my_profile torchrun --nproc_per_node=2 train.py --device cuda
```

**This has not been run in this project.** It was developed on a Mac with no NVIDIA
GPU, and `nsys` does not work there. Treat the command above as a starting point, not
as something tested.
