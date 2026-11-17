# Data for DataCenterEnv

## `gpu_power.csv`: measured GPU board power

One row per (GPU, workload). Columns:

| Column | Meaning |
|---|---|
| `gpu` | GPU type: T4, L4, A10, A100-40GB, H100, H200, B200 |
| `workload` | workload id (table below) |
| `tdp_w` | the board's enforced power limit as reported by NVML, in W |
| `active_w` | mean board power over 180 s measurement windows of the workload, in W |
| `active_w_sd` | standard deviation of `active_w` across runs |
| `n_runs` | number of runs averaged |
| `idle_w` | reserved idle power: mean board power over a 600 s idle period right after a job, with the model still loaded, in W (3 runs per GPU) |
| `coldstart_j` | board energy from the first NVML read in a fresh container to the first generated token, in J (3 runs per GPU) |

| Workload | What ran |
|---|---|
| W1 | Qwen2.5-1.5B-Instruct generation, batch 1, 128 new tokens (transformers, greedy) |
| W1b | Qwen2.5-7B-Instruct generation, as W1 |
| W2 | as W1, batch 32 |
| W3 | Qwen2.5-1.5B LoRA fine-tuning (r 16, alpha 32, q/k/v/o, 512 tokens), batch 2 |
| W4 | ResNet-50 training from random initialization, batch 128 |
| W5 | GPT-2-small-shaped training from random initialization, 1024 tokens, batch 4 |
| W6 | Qwen2.5-7B-Instruct offline batch inference in vLLM 0.10.1.1, 256 requests, 128 output tokens |

The environment only uses W3 to W6, the batch workloads a scheduler can defer. W1, W1b and W2 are interactive serving and are kept in the table for completeness.

**How it was measured.** I ran everything on 2026-11-12 on Modal serverless GPUs. Energy is the difference of NVML's total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`) between phase marks, divided by the window length. These are board values only: host CPU, memory, networking, cooling and facility overhead (PUE) are not included. T4 to H200 ran torch 2.5.1 and transformers 4.46.3. B200 and W6 ran torch 2.7.1 and transformers 4.55.2. W1b and W6 were not run on the T4, so on a T4 use the `tdp` model or a workload mix without W6.

**Source and license.** Each row is the mean per (GPU, workload) of the per-run tables in [`power_runs.csv`](https://github.com/prawalpokharel/cs249r_book/blob/acc980e0d8406739a04fda00d8a731c790b35e63/mlsysim/paper/data/serverless-gpu-energy/power_runs.csv) (148 runs), with idle and cold-start values from [`measured_gpu_energy.yaml`](https://github.com/prawalpokharel/cs249r_book/blob/acc980e0d8406739a04fda00d8a731c790b35e63/mlsysim/mlsysim/engine/data/measured_gpu_energy.yaml), both at commit `acc980e`. I took these measurements and wrote both files, which are published under MLSys·im's Apache-2.0 licence. As their author, I contribute this derived table under SustainGym's CC BY 4.0.

**Caveats.** One measurement day, one cloud provider, 3 to 5 runs per cell, and one batch size per workload. The software stack matters: H100 runs of W1 and W4 repeated on the newer stack used 9.4% and 6.0% less energy per unit of work at the same throughput.

## Job traces (not bundled)

`prepare_helios.py` downloads the public Helios traces (SenseTime GPU clusters, April to September 2020, 36 MB zip, checked by SHA-256) and writes one cluster's GPU jobs as a CSV for `TraceJobGenerator`:

```bash
python -m sustaingym.data.data_center.prepare_helios --cluster Venus --out helios_venus.csv.gz
```

The Helios traces are released under CC BY 4.0 at https://github.com/S-Lab-System-Group/HeliosData. If you use them, cite Hu et al., "Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters", SC '21, https://doi.org/10.1145/3458817.3476223.

Helios does not record which workload each job ran. `TraceJobGenerator` assigns workloads by a rule (`workload_rule='mix'` or `'shape'`, see `jobs.py`).

## Carbon intensity

The environment reuses the MOER data already bundled in `data/moer/` (SGIP signal, CAISO PG&E and SCE, May 2019 to August 2021), read through `data/load_moer.py`.
