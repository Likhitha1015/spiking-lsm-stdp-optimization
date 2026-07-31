# Controlled Recurrent Heterogeneity in Spiking Liquid State Machines

Implementation for the paper submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS).

Code: https://github.com/Likhitha1015/spiking-lsm-stdp-optimization

## Requirements
- Python >= 3.8
- NEST Simulator 3.9
- Python dependencies: `pip install -r requirements.txt`

## NEST Setup via Docker (Recommended)

    docker pull nest/nest-simulator:3.9
    docker run -it --rm -v $(pwd):/work -w /work nest/nest-simulator:3.9 bash

Or install NEST 3.9 natively: https://nest-simulator.readthedocs.io/en/v3.9/installation/index.html

## Repository Structure

    src/        Core pipeline (LSM, optimization, training, evaluation)
    datasets/   Five benchmark NPZ files (ECG, PHAL, WAF, ROBOT, JPVOW)
    results/    Saved optimization params and reservoir features per dataset
    scripts/    Shell scripts to reproduce experiments

## Reproduce Results

### Step 1: Optimize (Stage I + II)

    cd src/
    python lsm_optimise.py \
        --dataset ECG \
        --n_res 200 \
        --n_driven 20 \
        --I_scale 3000 \
        --indeg22 15 \
        --indeg23 30 \
        --optimiser BO \
        --n_folds 2 \
        --es_epochs 1 \
        --bo_calls_stdp 20 --bo_init_stdp 8 \
        --bo_calls_wd 12  --bo_init_wd 4 \
        --opt_dir ../results/OPT_ECG

### Step 2: Train and Infer

    python lsm_train_infer.py \
        --dataset ECG \
        --n_res 200 \
        --n_driven 20 \
        --I_scale 3000 \
        --indeg22 15 \
        --indeg23 30 \
        --n_epochs 10 \
        --ridge_alpha 10.0 \
        --opt_dir ../results/OPT_ECG \
        --results_dir ../results/RESULTS_ECG

### Step 3: Evaluate

    python evaluate.py \
        --dataset ECG \
        --results_dir ../results/RESULTS_ECG

## Dataset-Specific Parameters

| Dataset | n_res | ridge_alpha | Best Opt | Best Case |
|---------|-------|-------------|----------|-----------|
| ECG     | 200   | 10.0        | BO       | A         |
| PHAL    | 100   | 1.0         | GA       | C         |
| WAF     | 600   | 0.5         | BO       | C         |
| ROBOT   | 600   | 0.5         | GA       | A         |
| JPVOW   | 600   | 50.0        | BO       | B         |

Note: n_driven=20, I_scale=3000, indeg22=15, indeg23=30 are fixed across all datasets.

## Heterogeneity Cases

| Case | Weights     | Delays      |
|------|-------------|-------------|
| A    | Fixed       | Fixed       |
| B    | Fixed       | Distributed |
| C    | Distributed | Fixed       |
| D    | Distributed | Distributed |

## Best Results

| Dataset | Case | Opt | CER   | F1    | KM-ESN CER |
|---------|------|-----|-------|-------|------------|
| ECG     | A    | BO  | 0.150 | 0.803 | 0.180      |
| PHAL    | C    | GA  | 0.317 | 0.570 | 0.320      |
| WAF     | C    | BO  | 0.110 | 0.754 | 0.030      |
| ROBOT   | A    | GA  | 0.563 | 0.373 | 0.450      |
| JPVOW   | B    | BO  | 0.346 | 0.575 | 0.080      |

## Skip Optimization — Use Saved Results

Pre-computed optimization parameters and reservoir features are included in results/.
To evaluate directly without re-running NEST:

    python evaluate.py --dataset ECG --results_dir results/RESULTS_ECG

## Datasets
Sourced from Zenodo record 10852893 (https://zenodo.org/records/10852893),
consistent with the KM-ESN baseline.

## Citation

    @article{tnnls2025lsm,
      title={Controlled Recurrent Heterogeneity in Spiking Liquid State Machines
             for Multivariate Time-Series Classification},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      year={2025}
    }

## License
MIT License
