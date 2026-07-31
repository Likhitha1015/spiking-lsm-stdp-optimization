# Controlled Recurrent Heterogeneity in Spiking Liquid State Machines

Implementation for the paper submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS).

## Requirements
- Python >= 3.8
- NEST Simulator 3.9 (https://nest-simulator.readthedocs.io)
- Python dependencies: `pip install -r requirements.txt`

## Repository Structure

    src/        Core pipeline (LSM, optimization, training, evaluation)
    datasets/   Five benchmark NPZ files (ECG, PHAL, WAF, ROBOT, JPVOW)
    results/    Saved optimization params and reservoir features per dataset
    scripts/    Shell scripts to reproduce experiments

## Reproduce Results

### Step 1: Optimize (Stage I + II)

    cd src/
    python lsm_optimise.py \
        --dataset ECG --n_res 200 \
        --optimiser BO \
        --bo_calls_stdp 20 --bo_init_stdp 8 \
        --bo_calls_wd 12  --bo_init_wd 4 \
        --n_folds 2 \
        --opt_dir ../results/OPT_ECG

### Step 2: Train and Infer

    python lsm_train_infer.py \
        --dataset ECG --n_res 200 \
        --n_epochs 10 \
        --opt_dir ../results/OPT_ECG \
        --results_dir ../results/RESULTS_ECG

### Step 3: Evaluate

    python evaluate.py \
        --dataset ECG \
        --results_dir ../results/RESULTS_ECG

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

## NEST Simulator Setup

NEST 3.9 was run inside a Docker container. To reproduce the exact environment:

    docker pull nest/nest-simulator:3.9
    docker run -it --rm \
        -v $(pwd):/work \
        -w /work \
        nest/nest-simulator:3.9 \
        python src/lsm_optimise.py --dataset ECG ...

Alternatively, install NEST 3.9 natively following the official instructions:
https://nest-simulator.readthedocs.io/en/v3.9/installation/index.html

The Python NEST bindings (PyNEST) are included with the NEST installation.

## NEST Simulator Setup

NEST 3.9 was run inside a Docker container. To reproduce the exact environment:

    docker pull nest/nest-simulator:3.9
    docker run -it --rm \
        -v $(pwd):/work \
        -w /work \
        nest/nest-simulator:3.9 \
        python src/lsm_optimise.py --dataset ECG ...

Alternatively, install NEST 3.9 natively following the official instructions:
https://nest-simulator.readthedocs.io/en/v3.9/installation/index.html

The Python NEST bindings (PyNEST) are included with the NEST installation.
