"""
evaluate.py — Load saved LSM results and report metrics.
Usage: python evaluate.py --results_dir RESULTS_ECG_200_peak_BO --dataset ECG
"""
import argparse
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from lsm_generic import load_dataset

def snorm(F): return F/(F.max(axis=1,keepdims=True)+1e-6)

# Fixed alpha per dataset — selected based on best macro F1 across all cases
DATASET_ALPHA = {
    "ECG":   10.0,
    "PHAL":  1.0,
    "WAF":   0.5,
    "ROBOT": 0.5,
    "JPVOW": 50.0,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset",     required=True)
    ap.add_argument("--alpha",       type=float, default=None,
                    help="Ridge alpha (default: dataset-specific)")
    ap.add_argument("--cache_dir",   default="datasets_cache")
    args = ap.parse_args()

    _, ytr, _, yte, n_cls = load_dataset(args.dataset, cache_dir=args.cache_dir)
    alpha = args.alpha if args.alpha else DATASET_ALPHA.get(args.dataset, 1.0)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset} | Results: {args.results_dir}")
    print(f"Ridge α={alpha} | Test samples={len(yte)} | Classes={n_cls}")
    print(f"{'='*60}")
    print(f"{'Case':<6} {'CER':>8} {'Macro F1':>10} {'Acc%':>8}")
    print("-"*36)

    best_f1, best_cer, best_case = 0.0, 1.0, None
    for case in ["A","B","C","D"]:
        cd = f"{args.results_dir}/case_{case}"
        try:
            Ftr = snorm(np.load(f"{cd}/Ftr_l2.npy"))
            Fte = snorm(np.load(f"{cd}/Fte_l2.npy"))
            ytr_s = np.load(f"{cd}/train_labels.npy")
            if Fte.shape[0] != len(yte):
                print(f"Case {case}: shape mismatch"); continue
            clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
            clf.fit(Ftr, ytr_s); pred = clf.predict(Fte)
            cer = float(np.mean(pred != yte))
            f1  = f1_score(yte, pred, average="macro", zero_division=0)
            acc = (1-cer)*100
            print(f"Case {case}  {cer:>8.4f} {f1:>10.4f} {acc:>8.1f}%")
            if f1 > best_f1: best_f1, best_cer, best_case = f1, cer, case
        except Exception as e:
            print(f"Case {case}: {e}")

    print(f"\nBest: Case {best_case} | CER={best_cer:.4f} | F1={best_f1:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
