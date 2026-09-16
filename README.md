# Quantization buys memory, not energy

Replication package for:

> Gaur L, Raman R. Quantization buys memory, not energy: measured inference
> energy of 4-bit language models across three model sizes.
> *Manuscript submitted to Environmental Systems Research.*

Direct measurement of energy per 1000 generated tokens at 16-bit precision and
under 4-bit NormalFloat quantization, for three instruction-tuned models on one
70 W GPU. 48 measured runs, 8 per condition per model.

## Headline results

| Model size | Energy change | p | Memory change | Throughput change | Perplexity change |
|---|---:|---:|---:|---:|---:|
| 0.5B | **+52.5%** | <0.0001 | −53.2% | −40.2% | +24.2% |
| 1.5B | −0.3% | 0.77 | −62.4% | −36.9% | +13.5% |
| 3B | −1.6% | 0.16 | −66.5% | −36.7% | +15.6% |

Values are the change from 16-bit to 4-bit. A negative energy value would be a
saving; none of the three is a saving that survives testing. Memory falls at
every size. Throughput falls and output quality worsens at every size.

The severe penalty at 0.5B has Cohen's d = 7.76 with no overlap between
conditions: the highest-energy 16-bit run used less energy than the
lowest-energy 4-bit run.

## Repository layout

```
notebooks/
  01_pilot_0.5B.ipynb        pilot: one model, establishes the protocol
  02_two_sizes.ipynb         0.5B and 1.5B, 8 replicates per condition
  03_add_3B.ipynb            adds 3B, appends to the same results file
src/
  measure_energy.py          measurement harness
  measure_energy_pilot.py    single-model pilot version
  analysis_by_size.py        analysis with the power-capping sensitivity
  make_figures.py            figure generation
data/
  runs_48.csv                all 48 measured runs, in the schema the harness writes
                             (model, condition, block, rep, tokens_per_s,
                             wh_per_1k_tokens, model_mem_mb, perplexity, throttle)
  manifest_pilot_run.json    device, driver, package versions for the pilot
figures/
  Fig1.png Fig2.png Fig3.png figures as published
```

## Hardware and software

Measurements were taken on an NVIDIA Tesla T4, 15360 MiB, driver 580.82.07,
enforced power limit 70 W. Software: Python 3.13.15, PyTorch 2.11.0 with CUDA
12.8, Transformers 5.16.1, bitsandbytes 0.50.2.

Results are hardware-specific. Re-running on a different device will produce
different absolute figures and may produce a different balance between
conditions, which is the point of the power-capping caveat below.

## Reproducing

A CUDA GPU is required; the harness refuses to run on CPU rather than producing
numbers that cannot be defended.

```bash
pip install -r requirements.txt
python src/measure_energy.py       # or open a notebook in Colab with a T4
```

Every run appends to CSV as it completes, and re-running skips runs already
recorded, so an interrupted session resumes without repeating work. Expect
roughly 30 to 40 minutes for two sizes and 20 to 25 minutes to add the third.

To reproduce the published analysis from the supplied data without measuring:

```bash
python src/analysis_by_size.py     # reads data/runs_48.csv, prints Tables 1-3
python src/make_figures.py         # writes figures/Fig1-3.png
```

Both are run from the repository root. `analysis_by_size.py` reproduces the
power-capping table, the primary result and the sensitivity analysis exactly as
reported.

## Measurement design

**Energy** comes from the device cumulative energy counter
(`nvmlDeviceGetTotalEnergyConsumption`), with a 10 Hz power trace integrated
independently as a cross-check. In the pilot the two estimates differed by at
most 0.6%, mean 0.21%.

**Idle power** is measured before each condition block and recorded per run.

**Workload** is fixed: 8 prompts, greedy decoding, `min_new_tokens` equal to
`max_new_tokens` at 64, batch size 1, so every run produces exactly 512 tokens.
Token counts are measured from the output, never assumed.

**Replication** uses two blocks of four replicates with the condition order
reversed in the second block, so drift over a session does not fall
consistently on one condition. An untimed warm-up precedes each block.

## The power-capping caveat

The 16-bit condition reached the 70 W device ceiling in 0 of 8 runs at 0.5B,
6 of 8 at 1.5B and 8 of 8 at 3B. The 4-bit condition never reached it.

Capping is therefore systematically associated with the condition, and
excluding capped runs would remove the higher-power condition preferentially.
All runs are retained in the primary analysis and the alternatives are reported:
at 1.5B the estimate is −0.3% using all runs, −0.1% excluding capped runs and
−0.4% using only the capped 16-bit runs. At 3B no uncapped 16-bit run exists,
so the comparison is necessarily between a capped and an uncapped condition and
the parity there may reflect the ceiling rather than equal efficiency.

**Repeating this on a device with more power headroom is the most useful
extension of this work.** If you do, the harness will run unmodified.

## Scope

One quantization implementation and one data type. Single-stream generation at
batch size 1; dequantization overhead amortises differently with batching.
Three sizes from one model family. Device energy only, excluding host processor,
memory and cooling. Perplexity on one short passage is a proxy, and it did not
distinguish the 1.5B and 3B models from each other, so it should be read as a
within-size comparison between precisions.

## Citation

See `CITATION.cff`.

## Licence

Code under the MIT Licence (`LICENSE`); measurement data under CC BY 4.0.
