# SROIE Benchmark — InternVL3.5-8B (vLLM Data-Parallel)

Standalone notebook for benchmarking InternVL3.5-8B on the
[ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) receipt key-information extraction task.

Uses **vLLM** for inference with **data parallelism** — one independent vLLM engine
per GPU, images partitioned round-robin across workers. Scales linearly with GPU count.

## Prerequisites

1. **Kaggle API credentials** at `~/.kaggle/kaggle.json`
   ([setup guide](https://www.kaggle.com/docs/api#authentication))
2. **InternVL3.5-8B model** available at the path configured in the notebook
   (default: `/home/jovyan/nfs_share/models/InternVL3_5-8B`)
3. **GPU environment** with `vllm`, `Pillow`, and `tqdm`

## Setup

Copy the notebook to any directory on the GPU server:

```bash
mkdir -p ~/my_benchmarks
cp sroie_benchmark_internvl3.ipynb ~/my_benchmarks/
cd ~/my_benchmarks
```

That's it. No other files are needed.

## Running

Open the notebook and run all cells. The notebook will:

1. Create a `data/sroie/` subdirectory relative to the notebook
2. Install the `kaggle` CLI if not already available
3. Download the SROIE dataset (~834 MB) from Kaggle into `data/sroie/`
4. Launch data-parallel vLLM workers and run extraction across all GPUs
5. Report per-field precision, recall, F1, and overall F1
6. Export per-image CSV and summary JSON to `data/sroie/output/`

On subsequent runs the dataset is reused automatically (no re-download).

## Configuration

Edit the first code cell to adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/sroie` | Where the dataset is stored |
| `MODEL_PATH` | `/home/jovyan/nfs_share/models/InternVL3_5-8B` | Path to the model weights |
| `NUM_GPUS` | `2` | Number of data-parallel workers (one vLLM engine per GPU) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | GPU memory fraction for vLLM |
| `MAX_MODEL_LEN` | `8192` | Maximum model context length |
| `MAX_IMAGES` | `None` (all) | Limit images for quick test runs (e.g. `10`) |
| `MAX_NEW_TOKENS` | `256` | Max tokens per inference call |

## Output

```
data/sroie/
  output/
    sroie_internvl3_per_image.csv    # Per-image GT vs predicted, with match flags
    sroie_internvl3_summary.json     # Overall and per-field F1/precision/recall
```

## Directory Structure After First Run

```
your_directory/
  sroie_benchmark_internvl3.ipynb
  data/
    sroie/
      test/
        img/          # 347 receipt images
        entities/     # Ground truth (JSON per image)
      train/
        img/
        entities/
      output/         # Benchmark results
```
