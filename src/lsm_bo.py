"""
LSM — Bayesian Optimisation
==============================
BO-based optimisation for LSM reservoir configurations.
Mirrors lsm_ga.py structure but uses Gaussian Process surrogate.

Key advantage over ES and GA:
  - Most sample-efficient: builds a surrogate model of the fitness
    landscape and picks next candidate intelligently (not randomly)
  - Fewer NEST evaluations needed for same quality result
  - Expected Improvement (EI) acquisition: balances exploration/exploitation

Library: scikit-optimize (skopt) — GP surrogate with EI acquisition

Phase 1a: BO optimises 7 shared STDP params
Phase 1b: BO optimises w/d per case A/B/C/D independently

Fitness: 5-fold CV F1 on training set only (matching Paper 3)

Output: BO_DATASET/ folder (same structure as ES/GA output)
  stdp_params.json
  case_A/opt_params.json ... case_D/opt_params.json
  bo_convergence.pdf/.svg

Usage:
  python lsm_bo.py --dataset ECG   --n_res 200 --n_driven 80
  python lsm_bo.py --dataset ROBOT --n_res 100 --n_driven 40
  python lsm_bo.py --dataset JPVOW --n_res 150 --n_driven 60

  # Fast test
  python lsm_bo.py --dataset ECG --n_res 50 --n_driven 20 \
    --bo_epochs 2 --n_folds 2 \
    --bo_calls_stdp 10 --bo_init_stdp 5 \
    --bo_calls_wd 8  --bo_init_wd 4

After BO completes, run training + inference:
  python lsm_generic.py --dataset ECG --n_res 200 --n_driven 80 \
    --skip_es --es_dir BO_ECG
"""

import os, json, gc, time, argparse, warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence as skopt_convergence
import nest

warnings.filterwarnings("ignore")
nest.set_verbosity("M_ERROR")

rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

_COL = {"A": "#2563EB", "B": "#16A34A", "C": "#D97706", "D": "#DC2626"}
_MRK = {"A": "o",       "B": "s",       "C": "^",       "D": "D"}

# Import shared components from lsm_generic
from lsm_generic import (
    load_dataset, CaseParams, NEURONS_PER_CLASS, CASE_CONFIGS,
    ZENODO_URLS, x_to_stdp, _build, make_input_map,
    inject_step, clear_input, reset_membrane,
    teach, clear_teacher, get_l2, get_l3, l3_summed,
)


# ===========================================================================
# STDP SEARCH SPACE
# ===========================================================================

param_space = {
    "Aplus":            (1e-5,  0.05),
    "Aminus":           (1e-5,  0.05),
    "tau_plus":         (5.0,   50.0),
    "Aplus_triplet":    (1e-4,  0.02),
    "Aminus_triplet":   (1e-4,  0.02),
    "tau_plus_triplet": (20.0,  300.0),
    "Wmax":             (100.0, 600.0),
}

STDP_BOUNDS = list(param_space.values())
STDP_KEYS   = list(param_space.keys())
STDP_X0     = [0.001, 0.001, 20.0, 0.005, 0.005, 100.0, 300.0]


# ===========================================================================
# 5-FOLD CV FITNESS (same as lsm_ga / lsm_generic)
# ===========================================================================

_eval_n = [0]


def _run_one_fold(w_init, d_mean, stdp, arch,
                  X_tr, y_tr, X_val, y_val,
                  n_epochs, seed):
    """Train on one CV fold, evaluate Ridge(L2) F1 on validation fold."""
    n_res    = arch["n_res"]
    n_cls    = arch["n_classes"]
    n_driven = arch["n_driven"]
    n_ch     = arch["n_features"]
    I_scale  = arch["I_scale"]
    T        = X_tr.shape[1]
    N        = len(X_tr)

    rng = np.random.default_rng(seed)
    nest.ResetKernel()
    nest.set_verbosity(0)
    nest.SetKernelStatus({"print_time": False, "rng_seed": max(1, seed)})

    p   = CaseParams("_", w_fixed=float(w_init), d_fixed=float(d_mean))
    net = _build(p, stdp, arch, "train", rng)
    L2, L3 = net["L2"], net["L3"]
    tg     = net["tg"]
    rec_l2 = net["rec_l2"]
    rec_l3 = net["rec_l3"]
    L2_ids = net["L2_ids"]

    input_maps = [make_input_map(n_res, n_driven, rng) for _ in range(n_ch)]
    Ftr   = np.zeros((N, n_res), dtype=np.float32)
    Ie_h  = np.zeros(n_cls * NEURONS_PER_CLASS, dtype=np.float32)
    t_now = 0.0

    for ep in range(n_epochs):
        last = (ep == n_epochs - 1)
        idx  = np.random.default_rng(seed + ep).permutation(N)
        for si in idx:
            x, lbl = X_tr[si], int(y_tr[si])
            reset_membrane(L2)
            reset_membrane(L3)
            nest.SetStatus(rec_l2, {"n_events": 0})
            nest.SetStatus(rec_l3, {"n_events": 0})
            teach(tg, lbl, t_now, T, n_cls)
            for t in range(T):
                inject_step(L2, x[t], input_maps, I_scale)
                nest.Simulate(1.0)
                t_now += 1.0
            clear_input(L2, input_maps)
            clear_teacher(tg, n_cls)
            nest.Simulate(30.0)
            t_now += 30.0
            l2r = get_l2(rec_l2, L2_ids)
            l3f = get_l3(rec_l3, net["L3_ids"])
            for k in range(n_cls * NEURONS_PER_CLASS):
                Ie_h[k] += (-2.0 if l3f[k] > 10
                             else (1.0 if l3f[k] == 0 else 0.0))
                Ie_h[k]  = float(np.clip(Ie_h[k], -100.0, 30.0))
                nest.SetStatus(L3[k:k+1], {"I_e": Ie_h[k]})
            if last:
                Ftr[si] = l2r

    w22 = np.array([d["weight"] for d in nest.GetStatus(net["c22"])])

    rng2 = np.random.default_rng(seed + 100)
    nest.ResetKernel()
    nest.set_verbosity(0)
    nest.SetKernelStatus({"print_time": False, "rng_seed": max(1, seed + 100)})

    p2   = CaseParams("_", w_fixed=float(w_init), d_fixed=float(d_mean))
    net2 = _build(p2, stdp, arch, "inference", rng2, w_l2l2=w22)
    L2_2     = net2["L2"]
    rec_l2_2 = net2["rec_l2"]
    L2_ids2  = net2["L2_ids"]
    imap2    = [make_input_map(n_res, n_driven, rng2) for _ in range(n_ch)]

    Nval = len(X_val)
    Fval = np.zeros((Nval, n_res), dtype=np.float32)
    t2   = 0.0
    for i in range(Nval):
        reset_membrane(L2_2)
        reset_membrane(net2["L3"])
        nest.SetStatus(rec_l2_2, {"n_events": 0})
        for t in range(T):
            inject_step(L2_2, X_val[i][t], imap2, I_scale)
            nest.Simulate(1.0)
            t2 += 1.0
        clear_input(L2_2, imap2)
        nest.Simulate(30.0)
        t2 += 30.0
        Fval[i] = get_l2(rec_l2_2, L2_ids2)

    if Ftr.max() < 0.1 or Fval.max() < 0.1:
        gc.collect()
        return 0.0

    sc  = StandardScaler()
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(sc.fit_transform(Ftr), y_tr)
    f1  = f1_score(y_val, clf.predict(sc.transform(Fval)),
                   average="macro", zero_division=0)
    gc.collect()
    return float(f1)


def evaluate(w_init, d_mean, stdp, arch, Xtr, ytr,
             n_epochs=3, seed=42, n_folds=5) -> float:
    """5-fold CV fitness — test set NEVER used here."""
    global _eval_n
    _eval_n[0] += 1
    try:
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        f1s = []
        for fi, (tri, vali) in enumerate(kf.split(Xtr)):
            f1 = _run_one_fold(w_init, d_mean, stdp, arch,
                               Xtr[tri], ytr[tri],
                               Xtr[vali], ytr[vali],
                               n_epochs, seed + fi * 1000)
            f1s.append(f1)
            gc.collect()
        mean_f1 = float(np.mean(f1s))
        print(f"  eval#{_eval_n[0]:04d}: CV_F1={mean_f1:.4f} "
              f"folds={[f'{f:.3f}' for f in f1s]} "
              f"w={w_init:.1f} d={d_mean:.2f}")
        return mean_f1
    except Exception as e:
        import traceback
        print(f"  [WARN] eval#{_eval_n[0]} failed: {e}")
        traceback.print_exc()
        return 0.0


# ===========================================================================
# BAYESIAN OPTIMISATION
# ===========================================================================

def run_bo(fitness_fn, bounds, x0,
           n_calls=50, n_initial=10,
           seed=0, label=""):
    """
    Bayesian Optimisation with Gaussian Process surrogate.

    How it works:
      1. Evaluate n_initial random points (exploration)
      2. Fit GP surrogate model to observed (x, f(x)) pairs
      3. Use Expected Improvement (EI) to select next candidate:
           EI balances: exploit high-predicted regions
                        explore high-uncertainty regions
      4. Evaluate fitness at selected candidate
      5. Update GP model with new observation
      6. Repeat steps 3-5 until n_calls exhausted

    Key advantage over ES/GA:
      - Each evaluation informs the next one (sequential, model-based)
      - Typically needs 3-5x fewer evaluations than ES/GA for same quality
      - Especially effective for expensive fitness functions (ours: ~30s each)

    Args:
        fitness_fn:  callable(x) → float (higher = better)
        bounds:      list of (low, high) tuples
        x0:          warm-start point (initial guess)
        n_calls:     total evaluations including initial random points
        n_initial:   number of random initial points before GP kicks in
        seed:        random seed
        label:       label for print output
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # skopt uses Real dimensions
    dimensions = [Real(lo[i], hi[i], name=f"x{i}")
                  for i in range(len(bounds))]

    history = []

    def neg_fitness(x):
        """skopt minimises — negate to maximise F1."""
        f = fitness_fn(np.array(x))
        history.append(f)
        return -f

    print(f"\n  [BO] {label} | n_calls={n_calls} "
          f"n_initial={n_initial}")
    print(f"  [BO] GP surrogate + EI acquisition")

    result = gp_minimize(
        neg_fitness,
        dimensions=dimensions,
        x0=[list(x0)],          # warm-start with initial guess
        n_calls=n_calls,
        n_initial_points=n_initial,
        acq_func="EI",          # Expected Improvement
        noise=1e-6,             # small noise for numerical stability
        random_state=seed,
        verbose=False,
    )

    best_x  = np.array(result.x)
    best_f  = -result.fun       # un-negate
    hist    = history           # all evaluated F1 values in order

    # Running best (for convergence plot)
    running_best = []
    cur_best = -np.inf
    for f in hist:
        if f > cur_best:
            cur_best = f
        running_best.append(cur_best)

    print(f"  [BO] Done | best F1={best_f:.4f} | "
          f"evals={len(hist)}")
    return best_x, best_f, running_best


# ===========================================================================
# CONVERGENCE PLOT
# ===========================================================================

def plot_bo_convergence(stdp_hist, wd_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BO Optimisation Convergence — 5-fold CV F1\n"
                 "(GP surrogate + Expected Improvement)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    if stdp_hist:
        # Show all evaluations (grey) and running best (blue)
        ax.scatter(range(len(stdp_hist)), stdp_hist,
                   color="lightgrey", s=20, zorder=2, label="All evals")
        # Running best already computed in run_bo
        ax.plot(stdp_hist, color="#2563EB", lw=2,
                label=f"Best={max(stdp_hist):.4f}", zorder=3)
        ax.fill_between(range(len(stdp_hist)), stdp_hist,
                        alpha=0.10, color="#2563EB")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1a — STDP Optimisation\n"
                 "(Aplus, Aminus, tau, Wmax, triplet)")
    ax.set_ylim(0, 1)
    if stdp_hist:
        ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    for clbl, res in wd_results.items():
        hist = res.get("hist", [])
        if hist:
            ax.plot(hist, color=_COL[clbl], marker=_MRK[clbl],
                    lw=2, markersize=4,
                    label=f"Case {clbl} (best={max(hist):.3f})")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1b — Weight/Delay Optimisation\n"
                 "(per-case w/d scalar or distribution)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"bo_convergence.{ext}"),
                    format=ext)
    plt.close()
    print(f"  Saved → bo_convergence.svg/.pdf")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LSM Bayesian Optimisation (GP + EI)")
    ap.add_argument("--dataset",        default="ECG")
    ap.add_argument("--n_res",          type=int,   default=200)
    ap.add_argument("--n_driven",       type=int,   default=80)
    ap.add_argument("--I_scale",        type=float, default=3000.0)
    ap.add_argument("--n_folds",        type=int,   default=5,
                    help="CV folds for fitness (default 5)")
    ap.add_argument("--bo_epochs",      type=int,   default=3,
                    help="Training epochs per CV fold during BO")
    ap.add_argument("--bo_calls_stdp",  type=int,   default=50,
                    help="Total BO evaluations for STDP phase (default 50)")
    ap.add_argument("--bo_init_stdp",   type=int,   default=10,
                    help="Random initial points for STDP (default 10)")
    ap.add_argument("--bo_calls_wd",    type=int,   default=30,
                    help="Total BO evaluations for w/d phase (default 30)")
    ap.add_argument("--bo_init_wd",     type=int,   default=8,
                    help="Random initial points for w/d (default 8)")
    ap.add_argument("--bo_dir",         default=None,
                    help="Output dir (default: BO_DATASET/)")
    ap.add_argument("--cache_dir",      default="datasets_cache")
    args = ap.parse_args()

    BO_DIR = args.bo_dir or f"BO_{args.dataset}"
    os.makedirs(BO_DIR, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  LSM — Bayesian Optimisation (GP + EI)")
    print(f"  Dataset: {args.dataset} | n_res={args.n_res} "
          f"n_driven={args.n_driven}")
    print(f"  STDP: {args.bo_calls_stdp} calls ({args.bo_init_stdp} random init)")
    print(f"  w/d:  {args.bo_calls_wd} calls ({args.bo_init_wd} random init)")
    print(f"  {args.n_folds}-fold CV | epochs={args.bo_epochs}")
    print(f"  Output: {BO_DIR}/")
    print(f"{'='*60}")
    print(f"\n  BO advantage: ~3-5x fewer evals than ES/GA")
    print(f"  Total evals: "
          f"{args.bo_calls_stdp + 4*args.bo_calls_wd} "
          f"(vs ES: ~{(10*8+1) + 4*(8*8+1)} typical)")

    # Load training data only
    print(f"\n--- Loading {args.dataset} ---")
    Xtr, ytr, Xte, yte, n_cls = load_dataset(
        args.dataset, cache_dir=args.cache_dir)
    n_ch = Xtr.shape[2]
    print(f"  Train: {Xtr.shape} | Classes: {n_cls}")

    arch = {
        "n_res":      args.n_res,
        "n_classes":  n_cls,
        "n_features": n_ch,
        "indeg22":    min(8, args.n_res - 1),
        "indeg23":    30,
        "n_driven":   args.n_driven,
        "I_scale":    args.I_scale,
    }

    _eval_n[0] = 0

    # ── Phase 1a: STDP optimisation ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1a — STDP Optimisation (BO)")
    print(f"  7 params: {STDP_KEYS}")
    print(f"  Fixed: w=150pA d=1ms | {args.n_folds}-fold CV on Xtr only")
    print(f"{'='*60}")

    def stdp_fit(x):
        return evaluate(150.0, 1.0, x_to_stdp(x), arch, Xtr, ytr,
                        n_epochs=args.bo_epochs, n_folds=args.n_folds)

    best_stdp_arr, best_stdp_f1, stdp_hist = run_bo(
        stdp_fit, STDP_BOUNDS, STDP_X0,
        n_calls=args.bo_calls_stdp,
        n_initial=args.bo_init_stdp,
        seed=0, label="STDP")

    best_stdp = x_to_stdp(best_stdp_arr)

    with open(os.path.join(BO_DIR, "stdp_params.json"), "w") as f:
        json.dump({"stdp": best_stdp, "f1": best_stdp_f1,
                   "optimiser": "BO", "n_evals": len(stdp_hist)}, f, indent=2)

    print(f"\n  STDP best CV F1 = {best_stdp_f1:.4f} "
          f"(in {len(stdp_hist)} evaluations)")
    print(f"  Saved → {BO_DIR}/stdp_params.json")
    for k, v in best_stdp.items():
        if k != "synapse_model":
            print(f"    {k}: {v:.6f}")

    # ── Phase 1b: weight/delay per case ────────────────────────────────────
    wd_results = {}

    for clbl, cfg in CASE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Phase 1b — Case {clbl}: {cfg['names']} (BO)")
        print(f"{'='*60}")

        def wd_fit(x, cfg=cfg):
            w = float(np.clip(cfg["w_fn"](x), 10.0,
                              best_stdp["Wmax"] * 0.95))
            d = float(np.clip(cfg["d_fn"](x), 0.1, 25.0))
            return evaluate(w, d, best_stdp, arch, Xtr, ytr,
                           n_epochs=args.bo_epochs, n_folds=args.n_folds)

        best_arr, best_f1, hist = run_bo(
            wd_fit, cfg["bounds"],
            x0=cfg["x0"],
            n_calls=args.bo_calls_wd,
            n_initial=args.bo_init_wd,
            seed=ord(clbl), label=f"Case {clbl}")

        opt_p = cfg["save"](best_arr)
        wd_results[clbl] = {"params": opt_p, "f1": best_f1,
                             "hist": list(hist),
                             "n_evals": len(hist)}

        case_dir = os.path.join(BO_DIR, f"case_{clbl}")
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "opt_params.json"), "w") as f:
            json.dump(opt_p, f, indent=2)

        print(f"  Case {clbl}: CV F1={best_f1:.4f} "
              f"(in {len(hist)} evals) | {opt_p}")
        print(f"  Saved → {BO_DIR}/case_{clbl}/opt_params.json")

    # ── Convergence plot ───────────────────────────────────────────────────
    plot_bo_convergence(stdp_hist, wd_results, BO_DIR)

    # ── Save summary JSON ──────────────────────────────────────────────────
    total = time.time() - t_start
    summary = {
        "optimiser":       "BO",
        "dataset":         args.dataset,
        "stdp_f1":         best_stdp_f1,
        "stdp_n_evals":    len(stdp_hist),
        "runtime_hr":      total / 3600,
        "cases": {
            clbl: {
                "f1":      wd_results[clbl]["f1"],
                "n_evals": wd_results[clbl]["n_evals"],
                "params":  wd_results[clbl]["params"],
            }
            for clbl in CASE_CONFIGS
        }
    }
    with open(os.path.join(BO_DIR, "bo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BO COMPLETE — {args.dataset}  ({total/3600:.2f} hr)")
    print(f"{'='*60}")
    print(f"\n  STDP CV F1 = {best_stdp_f1:.4f} "
          f"({len(stdp_hist)} evals)")
    print(f"\n  {'Case':<6} {'CV F1':>8}  {'Evals':>6}  Parameters")
    print(f"  {'-'*55}")
    for clbl, res in wd_results.items():
        print(f"  {clbl:<6} {res['f1']:>8.4f}  "
              f"{res['n_evals']:>6}  {res['params']}")

    best_case = max(wd_results, key=lambda c: wd_results[c]["f1"])
    print(f"\n  Best case: {best_case} | "
          f"F1={wd_results[best_case]['f1']:.4f}")
    total_evals = len(stdp_hist) + sum(
        wd_results[c]["n_evals"] for c in wd_results)
    print(f"\n  Total evaluations: {total_evals} "
          f"({total/total_evals:.0f}s per eval avg)")

    print(f"\n  Next step — training + inference:")
    print(f"    python lsm_generic.py --dataset {args.dataset} "
          f"--n_res {args.n_res} --n_driven {args.n_driven} "
          f"--skip_es --es_dir {BO_DIR}")
