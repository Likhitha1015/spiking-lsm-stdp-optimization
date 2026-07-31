"""
Step 1 — Optimisation Only (ES / GA / BO)
Runs optimisation and saves best params to OPT_DIR/.
just finds best hyperparameters.

Output: OPT_DIR/
  stdp_params.json         — best STDP params + CV F1
  case_A/opt_params.json   — best w/d for Case A
  case_B/opt_params.json   — best w/d for Case B
  case_C/opt_params.json   — best w/d for Case C
  case_D/opt_params.json   — best w/d for Case D
  es_convergence.pdf/.svg  — convergence plots
  arch.json                — network architecture (for Step 2)

Inspect results then run Step 2:
  python lsm_train_infer.py --dataset ECG --opt_dir OPT_DIR
"""

import os, json, gc, time, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import nest

warnings.filterwarnings("ignore")
nest.set_verbosity("M_QUIET")

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
_MRK = {"A": "o", "B": "s", "C": "^", "D": "D"}

from lsm_generic import (
    load_dataset, CaseParams, NEURONS_PER_CLASS, CASE_CONFIGS,
    x_to_stdp, _build, make_input_map, inject_step, clear_input,
    reset_membrane, teach, clear_teacher, get_l2, get_l3,
    run_es, run_ga,
)


# STDP SEARCH SPACE
STDP_BOUNDS = [
    (1e-5, 0.05), (1e-5, 0.05), (5.0, 50.0),
    (1e-4, 0.05), (1e-4, 0.05), (20.0, 300.0), (100.0, 800.0),
]
# Expanded: Aplus_triplet/Aminus_triplet 0.02→0.05, Wmax 600→800
# Previous best was hitting upper bounds at 0.0142/0.0155 and w_fixed=555
STDP_X0 = [0.001, 0.001, 20.0, 0.005, 0.005, 100.0, 300.0]

# Dataset-specific optimizer hyperparameters
# sigma_init: ES step size (larger = more exploration)
# mut_rate:   GA mutation rate (larger = more diversity)
# cross_rate: GA crossover rate
# acq_func:   BO acquisition function
#   EI  = Expected Improvement (balanced)
#   LCB = Lower Confidence Bound (more exploration)
#   PI  = Probability of Improvement (exploitation)
DATASET_OPT_CONFIGS = {
    "ECG":   {"sigma_init":0.25, "mut_rate":0.15, "cross_rate":0.7, "acq_func":"EI"},
    "ECG96": {"sigma_init":0.25, "mut_rate":0.15, "cross_rate":0.7, "acq_func":"EI"},
    "PHAL":  {"sigma_init":0.20, "mut_rate":0.15, "cross_rate":0.8, "acq_func":"EI"},
    "WAF":   {"sigma_init":0.20, "mut_rate":0.10, "cross_rate":0.8, "acq_func":"EI"},
    "ROBOT": {"sigma_init":0.40, "mut_rate":0.30, "cross_rate":0.6, "acq_func":"LCB"},
    "JPVOW": {"sigma_init":0.35, "mut_rate":0.25, "cross_rate":0.6, "acq_func":"LCB"},
    "LIB":   {"sigma_init":0.40, "mut_rate":0.30, "cross_rate":0.6, "acq_func":"LCB"},
    "SWE":   {"sigma_init":0.30, "mut_rate":0.20, "cross_rate":0.7, "acq_func":"EI"},
    "CHLO":  {"sigma_init":0.25, "mut_rate":0.15, "cross_rate":0.7, "acq_func":"EI"},
}
_DEFAULT_OPT = {"sigma_init":0.30, "mut_rate":0.20, "cross_rate":0.7, "acq_func":"EI"}

# Dataset-specific STDP bounds for short-sequence datasets (T < 20ms)
STDP_BOUNDS_SHORT = [
    (1e-5, 0.05), (1e-5, 0.05), (0.5, 10.0),
    (1e-4, 0.02), (1e-4, 0.02), (1.0,  14.0), (100.0, 200.0),
]
STDP_X0_SHORT = [0.001, 0.001, 5.0, 0.005, 0.005, 8.0, 150.0]
SHORT_SEQ_DATASETS = ["ROBOT"]

# ROBOT-specific case configs: tight delays (d<5ms) for T=15ms sequences
CASE_CONFIGS_ROBOT = {
    "A": {
        "names":  ["w_fixed", "d_fixed"],
        "x0":     [150.0, 1.0],
        "bounds": [(20.0, 580.0), (0.1, 5.0)],
        "w_fn":   lambda x: x[0],
        "d_fn":   lambda x: x[1],
        "save":   lambda x: {"w_fixed": x[0], "d_fixed": x[1]},
        "load":   lambda d: __import__('lsm_generic', fromlist=['CaseParams']).CaseParams("A",
                            w_fixed=d.get("w_fixed"), d_fixed=d.get("d_fixed")),
    },
    "B": {
        "names":  ["w_fixed", "d_low", "d_range"],
        "x0":     [150.0, 0.1, 3.0],
        "bounds": [(20.0, 580.0), (0.1, 3.0), (0.1, 4.0)],
        "w_fn":   lambda x: x[0],
        "d_fn":   lambda x: x[1] + x[2] / 2,
        "save":   lambda x: {"w_fixed": x[0], "d_low": x[1], "d_high": x[1]+x[2]},
        "load":   lambda d: __import__('lsm_generic', fromlist=['CaseParams']).CaseParams("B",
                            w_fixed=d.get("w_fixed"), d_low=d.get("d_low"), d_high=d.get("d_high")),
    },
    "C": {
        "names":  ["w_low", "w_range", "d_fixed"],
        "x0":     [100.0, 200.0, 1.0],
        "bounds": [(10.0, 400.0), (50.0, 400.0), (0.1, 5.0)],
        "w_fn":   lambda x: x[0] + x[1] / 2,
        "d_fn":   lambda x: x[2],
        "save":   lambda x: {"w_low": x[0], "w_high": x[0]+x[1], "d_fixed": x[2]},
        "load":   lambda d: __import__('lsm_generic', fromlist=['CaseParams']).CaseParams("C",
                            w_low=d.get("w_low"), w_high=d.get("w_high"), d_fixed=d.get("d_fixed")),
    },
    "D": {
        "names":  ["w_low", "w_range", "d_low", "d_range"],
        "x0":     [100.0, 200.0, 0.1, 3.0],
        "bounds": [(10.0, 400.0), (50.0, 400.0), (0.1, 3.0), (0.1, 4.0)],
        "w_fn":   lambda x: x[0] + x[1] / 2,
        "d_fn":   lambda x: x[2] + x[3] / 2,
        "save":   lambda x: {"w_low": x[0], "w_high": x[0]+x[1],
                              "d_low": x[2], "d_high": x[2]+x[3]},
        "load":   lambda d: __import__('lsm_generic', fromlist=['CaseParams']).CaseParams("D",
                            w_low=d.get("w_low"), w_high=d.get("w_high"),
                            d_low=d.get("d_low"), d_high=d.get("d_high")),
    },
}



# 5-FOLD CV FITNESS

_eval_n = [0]


def _run_one_fold(w_init, d_mean, stdp, arch,
                  X_tr, y_tr, X_val, y_val, n_epochs, seed):
    n_res    = arch["n_res"]
    n_cls    = arch["n_classes"]
    n_driven = arch["n_driven"]
    n_ch     = arch["n_features"]
    I_scale  = arch["I_scale"]
    T        = X_tr.shape[1]
    N        = len(X_tr)

    rng = np.random.default_rng(seed)
    nest.ResetKernel()
    nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time": False, "overwrite_files": True, "rng_seed": max(1, seed)})

    p   = CaseParams("_", w_fixed=float(w_init), d_fixed=float(d_mean))
    net = _build(p, stdp, arch, "train", rng)
    L2, L3 = net["L2"], net["L3"]
    tg      = net["tg"]
    rec_l2  = net["rec_l2"]
    rec_l3  = net["rec_l3"]
    L2_ids  = net["L2_ids"]

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
                Ie_h[k] += (-2.0 if l3f[k] > 10 else (1.0 if l3f[k] == 0 else 0.0))
                Ie_h[k]  = float(np.clip(Ie_h[k], -100.0, 30.0))
                nest.SetStatus(L3[k:k+1], {"I_e": Ie_h[k]})
            if last:
                Ftr[si] = l2r

    w22 = np.array([d["weight"] for d in nest.GetStatus(net["c22"])])

    rng2 = np.random.default_rng(seed + 100)
    nest.ResetKernel()
    nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time": False, "overwrite_files": True, "rng_seed": max(1, seed + 100)})
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
        return 0.0, 1.0

    sc   = StandardScaler()
    clf  = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(sc.fit_transform(Ftr), y_tr)
    pred = clf.predict(sc.transform(Fval))
    f1   = f1_score(y_val, pred, average="macro", zero_division=0)
    cer  = float(np.mean(pred != y_val))
    gc.collect()
    return float(f1), float(cer)


def evaluate(w_init, d_mean, stdp, arch, Xtr, ytr,
             n_epochs=3, seed=42, n_folds=5) -> float:
    """5-fold CV fitness — test set NEVER used here."""
    global _eval_n
    _eval_n[0] += 1
    try:
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        f1s, cers = [], []
        for fi, (tri, vali) in enumerate(kf.split(Xtr)):
            f1, cer = _run_one_fold(w_init, d_mean, stdp, arch,
                                    Xtr[tri], ytr[tri],
                                    Xtr[vali], ytr[vali],
                                    n_epochs, seed + fi * 1000)
            f1s.append(f1)
            cers.append(cer)
            gc.collect()
        mean_f1  = float(np.mean(f1s))
        mean_cer = float(np.mean(cers))
        fitness  = 0.7 * mean_f1 + 0.3 * (1.0 - mean_cer)
        print(f"  eval#{_eval_n[0]:04d}: CV_F1={mean_f1:.4f} "
              f"CER={mean_cer:.4f} fitness={fitness:.4f} "
              f"folds={[f'{f:.3f}' for f in f1s]} "
              f"w={w_init:.1f} d={d_mean:.2f}")
        return fitness
    except Exception as e:
        import traceback
        print(f"  [WARN] eval#{_eval_n[0]} failed: {e}")
        traceback.print_exc()
        return 0.0



# BO OPTIMISER 

def run_bo(fitness_fn, bounds, x0, n_calls=50, n_initial=10,
           seed=0, label="", acq_func="EI"):
    """Bayesian Optimisation with GP surrogate + EI acquisition."""
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        raise ImportError("pip install scikit-optimize --break-system-packages")

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    dimensions = [Real(lo[i], hi[i], name=f"x{i}") for i in range(len(bounds))]
    history = []

    def neg_f(x):
        f = fitness_fn(np.array(x))
        history.append(f)
        return -f

    print(f"\n  [BO] {label} | n_calls={n_calls} n_initial={n_initial}")
    result = gp_minimize(neg_f, dimensions=dimensions, x0=[list(x0)],
                         n_calls=n_calls, n_initial_points=n_initial,
                         acq_func=acq_func, noise=1e-6, random_state=seed,
                         verbose=False)
    best_x = np.array(result.x)
    best_f = -result.fun
    running = []
    cur = -np.inf
    for f in history:
        cur = max(cur, f)
        running.append(cur)
    print(f"  [BO] Done | best F1={best_f:.4f} | evals={len(history)}")
    return best_x, best_f, running




# POST-OPTIMISATION PLOTS

def plot_cv_f1_bars(wd_results, best_stdp_f1, opt_dir, optimiser):
    """
    Bar chart of CV F1 per case after optimisation.
    Shows which case the optimiser found best BEFORE actual training.
    """
    cases  = list(wd_results.keys())
    f1s    = [wd_results[c]["f1"] for c in cases]
    colors = [_COL[c] for c in cases]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cases, f1s, color=colors, alpha=0.85,
                  edgecolor="black", lw=0.8, zorder=3)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(best_stdp_f1, ls="--", color="navy", lw=1.5,
               label=f"STDP baseline F1={best_stdp_f1:.4f}")
    ax.set_xlabel("Case")
    ax.set_ylabel("5-fold CV F1 (training set only)")
    ax.set_title(f"{optimiser} Optimisation — CV F1 per Case\n"
                 f"(estimated performance before full training)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_xticklabels([f"Case {c}\n{_CASE_SHORT[c]}"
                        for c in cases], fontsize=9)
    plt.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(opt_dir, f"cv_f1_bars.{ext}"), format=ext)
    plt.close()
    print("  Saved → cv_f1_bars.svg/.pdf")


def plot_param_table(wd_results, best_stdp, opt_dir, optimiser):
    """
    Table of optimised parameters for all 4 cases.
    Useful to inspect before running training.
    """
    cols = ["Case", "Description",
            "Weight", "Delay", "CV F1"]
    rows = []
    for clbl, res in wd_results.items():
        p = res["params"]
        if "w_fixed" in p and "d_fixed" in p:
            w_str = f"fixed {p['w_fixed']:.1f}pA"
            d_str = f"fixed {p['d_fixed']:.2f}ms"
        elif "w_fixed" in p:
            w_str = f"fixed {p['w_fixed']:.1f}pA"
            d_str = f"U[{p['d_low']:.1f},{p['d_high']:.1f}]ms"
        elif "d_fixed" in p:
            w_str = f"U[{p['w_low']:.0f},{p['w_high']:.0f}]pA"
            d_str = f"fixed {p['d_fixed']:.2f}ms"
        else:
            w_str = f"U[{p['w_low']:.0f},{p['w_high']:.0f}]pA"
            d_str = f"U[{p['d_low']:.1f},{p['d_high']:.1f}]ms"
        rows.append([
            f"Case {clbl}", _CASE_SHORT[clbl],
            w_str, d_str, f"{res['f1']:.4f}"
        ])

    # Add STDP row
    stdp_str = (f"Aplus={best_stdp['Aplus']:.4f}  "
                f"Aminus={best_stdp['Aminus']:.4f}  "
                f"Wmax={best_stdp['Wmax']:.0f}pA")
    rows.append(["STDP", stdp_str, "—", "—",
                 f"{max(r[4] for r in rows)}"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 2.5)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#1e3a5f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, clbl in enumerate(list(wd_results.keys())):
        for j in range(len(cols)):
            tbl[(i+1, j)].set_facecolor(_COL[clbl] + "22")
    # Highlight best F1
    best_clbl = max(wd_results, key=lambda c: wd_results[c]["f1"])
    best_row  = list(wd_results.keys()).index(best_clbl) + 1
    tbl[(best_row, 4)].set_facecolor("#d4edda")
    ax.set_title(f"{optimiser} Optimisation Results — Inspect Before Training",
                 fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(opt_dir, f"opt_params_table.{ext}"),
                    format=ext)
    plt.close()
    print("  Saved → opt_params_table.svg/.pdf")


def plot_stdp_params(best_stdp, opt_dir, optimiser):
    param_bounds = {
        "Aplus":            (1e-5,  0.05),
        "Aminus":           (1e-5,  0.05),
        "tau_plus":         (5.0,   50.0),
        "Aplus_triplet":    (1e-4,  0.02),
        "Aminus_triplet":   (1e-4,  0.02),
        "tau_plus_triplet": (20.0,  300.0),
        "Wmax":             (100.0, 600.0),
    }
    keys   = list(param_bounds.keys())
    values = [best_stdp.get(k, 0) for k in keys]
    # Normalise each to [0,1] within its search range
    norm_vals = []
    for k, v in zip(keys, values):
        lo, hi = param_bounds[k]
        norm_vals.append((v - lo) / (hi - lo + 1e-12))

    labels = ["A+", "A−", "τ+", "A+_t", "A−_t", "τ+_t", "Wmax"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Optimised STDP Parameters — {optimiser}",
                 fontsize=13, fontweight="bold")

    # Left: normalised position in search range
    ax = axes[0]
    cols = ["#2563EB" if v > 0.5 else "#DC2626" for v in norm_vals]
    bars = ax.bar(labels, norm_vals, color=cols, alpha=0.85,
                  edgecolor="black", lw=0.5, zorder=3)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.6,
               label="Midpoint of search range")
    for bar, v, rv in zip(bars, norm_vals, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{rv:.4f}", ha="center", fontsize=8, rotation=45)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("Normalised position in search range")
    ax.set_title("Position in Search Space\n(blue=high end, red=low end)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

    # Right: actual values
    ax = axes[1]
    log_vals = [np.log10(v) if v > 0 else 0 for v in values]
    ax.bar(labels, log_vals, color="#2563EB", alpha=0.75,
           edgecolor="black", lw=0.5, zorder=3)
    for i, (lv, rv) in enumerate(zip(log_vals, values)):
        ax.text(i, lv + 0.05, f"{rv:.4f}", ha="center",
                fontsize=8, rotation=45)
    ax.set_ylabel("log10(value)")
    ax.set_title("Actual Values (log scale)")
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

    plt.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(opt_dir, f"stdp_params_viz.{ext}"),
                    format=ext)
    plt.close()
    print("  Saved → stdp_params_viz.svg/.pdf")


def plot_expected_distributions(wd_results, opt_dir):
    rng = np.random.default_rng(42)
    n_sample = 1600  # typical n_res × indeg22 = 200 × 8

    for param_type, label in [("weight", "Weight (pA)"),
                               ("delay",  "Delay (ms)")]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"Expected {param_type.capitalize()} Distribution "
                     f"(Before Training) — Based on Optimised Parameters",
                     fontsize=12, fontweight="bold")

        for ax, (clbl, res) in zip(axes.flat, wd_results.items()):
            p = res["params"]

            if param_type == "weight":
                if "w_fixed" in p:
                    vals = np.full(n_sample, p["w_fixed"])
                    is_fixed = True
                else:
                    vals = rng.uniform(p["w_low"], p["w_high"], n_sample)
                    is_fixed = False
            else:  # delay
                if "d_fixed" in p:
                    vals = np.full(n_sample, p["d_fixed"])
                    is_fixed = True
                else:
                    vals = rng.uniform(p["d_low"], p["d_high"], n_sample)
                    is_fixed = False

            bins = 5 if is_fixed else 40
            ax.hist(vals, bins=bins, color=_COL[clbl],
                    alpha=0.82, edgecolor="white", lw=0.4, zorder=3)
            ax.axvline(vals.mean(), color="black", ls="--", lw=1.5,
                       label=f"Mean={vals.mean():.2f}")

            if not is_fixed:
                lo = p.get("w_low" if param_type == "weight" else "d_low")
                hi = p.get("w_high" if param_type == "weight" else "d_high")
                ax.axvline(lo, color="red", ls=":", lw=1.2,
                           label=f"low={lo:.2f}")
                ax.axvline(hi, color="blue", ls=":", lw=1.2,
                           label=f"high={hi:.2f}")

            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            dist_type = "Fixed" if is_fixed else "Random U[low,high]"
            ax.set_title(f"Case {clbl}: {_CASE_SHORT[clbl]} | {dist_type} | "
                         f"mean={vals.mean():.2f}  std={vals.std():.2f}",
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

        plt.tight_layout()
        fname = f"expected_{param_type}_dist"
        for ext in ("svg", "pdf"):
            fig.savefig(os.path.join(opt_dir, f"{fname}.{ext}"), format=ext)
        plt.close()
        print(f"  Saved → {fname}.svg/.pdf")


_CASE_SHORT = {
    "A": "Fixed w + Fixed d",
    "B": "Fixed w + Random d",
    "C": "Random w + Fixed d",
    "D": "Random w + Random d",
}


def plot_convergence(stdp_hist, wd_results, opt_dir, optimiser):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{optimiser} Optimisation Convergence — 5-fold CV F1",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    if stdp_hist:
        ax.plot(stdp_hist, color="#2563EB", lw=2, marker="o", markersize=5,
                label=f"Best={max(stdp_hist):.4f}")
        ax.fill_between(range(len(stdp_hist)), stdp_hist,
                        alpha=0.12, color="#2563EB")
    ax.set_xlabel("Generation / Evaluation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1a — STDP Optimisation")
    ax.set_ylim(0, 1)
    if stdp_hist:
        ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    for clbl, res in wd_results.items():
        hist = res.get("hist", [])
        if hist:
            ax.plot(hist, color=_COL[clbl], marker=_MRK[clbl],
                    lw=2, markersize=5,
                    label=f"Case {clbl} (best={max(hist):.3f})")
    ax.set_xlabel("Generation / Evaluation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1b — Weight/Delay Optimisation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(opt_dir, f"convergence_{optimiser.lower()}.{ext}"),
                    format=ext)
    plt.close()
    print(f"  Saved → convergence_{optimiser.lower()}.svg/.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LSM Optimisation — ES / GA / BO")
    ap.add_argument("--dataset",      default="ECG")
    ap.add_argument("--n_res",        type=int,   default=200)
    ap.add_argument("--n_driven",     type=int,   default=20)
    ap.add_argument("--burst_ms",     type=float, default=1.0,
                    help="Simulation ms per input timestep (>1 stretches T for short sequences)")
    ap.add_argument("--indeg22",      type=int,   default=None)
    ap.add_argument("--indeg23",      type=int,   default=None)
    ap.add_argument("--I_scale",      type=float, default=3000.0)
    ap.add_argument("--n_folds",      type=int,   default=3)
    ap.add_argument("--es_epochs",    type=int,   default=3,
                    help="Epochs per fold during optimisation")
    ap.add_argument("--optimiser",    default="ES",
                    choices=["ES", "GA", "BO"],
                    help="Optimiser to use (default: ES)")
    # ES settings
    ap.add_argument("--es_pop",       type=int,   default=8)
    ap.add_argument("--es_gen_stdp",  type=int,   default=15)
    ap.add_argument("--es_gen_wd",    type=int,   default=10)
    # GA settings
    ap.add_argument("--ga_pop",       type=int,   default=10)
    ap.add_argument("--ga_gen_stdp",  type=int,   default=15)
    ap.add_argument("--ga_gen_wd",    type=int,   default=10)
    ap.add_argument("--ga_mut",       type=float, default=0.2)
    ap.add_argument("--ga_cross",     type=float, default=0.5)
    # BO settings
    ap.add_argument("--bo_calls_stdp",type=int,   default=50)
    ap.add_argument("--bo_init_stdp", type=int,   default=10)
    ap.add_argument("--bo_calls_wd",  type=int,   default=30)
    ap.add_argument("--bo_init_wd",   type=int,   default=8)
    # Output
    ap.add_argument("--opt_dir",      default=None,
                    help="Output dir (default: OPT_DATASET_OPTIMISER/)")
    ap.add_argument("--cache_dir",    default="datasets_cache")
    args = ap.parse_args()

    OPT_DIR = args.opt_dir or f"OPT_{args.dataset}_{args.optimiser}"
    os.makedirs(OPT_DIR, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  LSM Optimisation — {args.optimiser}")
    print(f"  Dataset: {args.dataset} | n_res={args.n_res} "
          f"n_driven={args.n_driven} I_scale={args.I_scale}")
    print(f"  {args.n_folds}-fold CV | es_epochs={args.es_epochs}")
    print(f"  Output: {OPT_DIR}/")
    print(f"{'='*60}")

    # Load training data ONLY — test set not used during optimisation
    print(f"\n--- Loading {args.dataset} (train only for optimisation) ---")
    Xtr, ytr, Xte, yte, n_cls = load_dataset(
        args.dataset, cache_dir=args.cache_dir)
    n_ch = Xtr.shape[2]
    print(f"  Train: {Xtr.shape} | Classes: {n_cls}")

    arch = {
        "n_res":      args.n_res,
        "n_classes":  n_cls,
        "n_features": n_ch,
        "indeg22":    args.indeg22 if args.indeg22 is not None else min(15, args.n_res - 1),
        "indeg23":    args.indeg23 if args.indeg23 is not None else 30,
        "n_driven":   args.n_driven,
        "burst_ms":   args.burst_ms,
        "I_scale":    args.I_scale,
    }

    # Save arch for lsm_train_infer.py to read
    with open(os.path.join(OPT_DIR, "arch.json"), "w") as f:
        json.dump({**arch, "dataset": args.dataset}, f, indent=2)

    _eval_n[0] = 0

    #Phase 1a: STDP optimisation
    print(f"\n{'='*60}")
    print(f"  Phase 1a — STDP ({args.optimiser})")
    print(f"{'='*60}")

    def stdp_fit(x):
        return evaluate(150.0, 1.0, x_to_stdp(x), arch, Xtr, ytr,
                        n_epochs=args.es_epochs, n_folds=args.n_folds)

    if args.optimiser == "ES":
        _oc = DATASET_OPT_CONFIGS.get(args.dataset, _DEFAULT_OPT)
        best_arr, best_f1, stdp_hist = run_es(
            stdp_fit, STDP_BOUNDS_SHORT if args.dataset in SHORT_SEQ_DATASETS else STDP_BOUNDS, STDP_X0_SHORT if args.dataset in SHORT_SEQ_DATASETS else STDP_X0,
            popsize=args.es_pop, n_gen=args.es_gen_stdp,
            seed=0, label="STDP [ES]", sigma_init=_oc["sigma_init"])
    elif args.optimiser == "GA":
        _oc = DATASET_OPT_CONFIGS.get(args.dataset, _DEFAULT_OPT)
        best_arr, best_f1, stdp_hist = run_ga(
            stdp_fit, STDP_BOUNDS_SHORT if args.dataset in SHORT_SEQ_DATASETS else STDP_BOUNDS,
            popsize=args.ga_pop, n_gen=args.ga_gen_stdp,
            mutation_rate=_oc["mut_rate"], crossover_rate=_oc["cross_rate"],
            seed=0, label="STDP [GA]")
    else:  # BO
        _oc = DATASET_OPT_CONFIGS.get(args.dataset, _DEFAULT_OPT)
        best_arr, best_f1, stdp_hist = run_bo(
            stdp_fit, STDP_BOUNDS_SHORT if args.dataset in SHORT_SEQ_DATASETS else STDP_BOUNDS, STDP_X0_SHORT if args.dataset in SHORT_SEQ_DATASETS else STDP_X0,
            n_calls=args.bo_calls_stdp, n_initial=args.bo_init_stdp,
            seed=0, label="STDP [BO]", acq_func=_oc["acq_func"])

    best_stdp = x_to_stdp(best_arr)
    with open(os.path.join(OPT_DIR, "stdp_params.json"), "w") as f:
        json.dump({"stdp": best_stdp, "f1": best_f1,
                   "optimiser": args.optimiser}, f, indent=2)
    print(f"\n  STDP best CV F1 = {best_f1:.4f}")
    print(f"  Saved → {OPT_DIR}/stdp_params.json")

    #  Phase 1b: weight/delay per case
    wd_results = {}

    for clbl, cfg in (CASE_CONFIGS_ROBOT if args.dataset in SHORT_SEQ_DATASETS else CASE_CONFIGS).items():
        print(f"\n{'='*60}")
        print(f"  Phase 1b — Case {clbl} ({args.optimiser})")
        print(f"{'='*60}")

        def wd_fit(x, cfg=cfg):
            w = float(np.clip(cfg["w_fn"](x), 10.0,
                              best_stdp["Wmax"] * 0.95))
            d = float(np.clip(cfg["d_fn"](x), 0.1, 25.0))
            return evaluate(w, d, best_stdp, arch, Xtr, ytr,
                           n_epochs=args.es_epochs, n_folds=args.n_folds)

        if args.optimiser == "ES":
            arr, f1, hist = run_es(
                wd_fit, cfg["bounds"], cfg["x0"],
                popsize=args.es_pop, n_gen=args.es_gen_wd,
                seed=ord(clbl), label=f"Case {clbl} [ES]")
        elif args.optimiser == "GA":
            arr, f1, hist = run_ga(
                wd_fit, cfg["bounds"],
                popsize=args.ga_pop, n_gen=args.ga_gen_wd,
                mutation_rate=args.ga_mut, crossover_rate=args.ga_cross,
                seed=ord(clbl), label=f"Case {clbl} [GA]")
        else:  # BO
            arr, f1, hist = run_bo(
                wd_fit, cfg["bounds"], cfg["x0"],
                n_calls=args.bo_calls_wd, n_initial=args.bo_init_wd,
                seed=ord(clbl), label=f"Case {clbl} [BO]")

        opt_p = cfg["save"](arr)
        wd_results[clbl] = {"params": opt_p, "f1": f1, "hist": list(hist)}

        case_dir = os.path.join(OPT_DIR, f"case_{clbl}")
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "opt_params.json"), "w") as f:
            json.dump(opt_p, f, indent=2)
        print(f"  Case {clbl}: CV F1={f1:.4f} | {opt_p}")


    print(f"\n--- Generating plots → {OPT_DIR}/ ---")
    plot_convergence(stdp_hist, wd_results, OPT_DIR, args.optimiser)
    plot_cv_f1_bars(wd_results, best_f1, OPT_DIR, args.optimiser)
    plot_param_table(wd_results, best_stdp, OPT_DIR, args.optimiser)
    plot_stdp_params(best_stdp, OPT_DIR, args.optimiser)
    plot_expected_distributions(wd_results, OPT_DIR)
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  OPTIMISATION COMPLETE — {args.dataset} "
          f"({total/3600:.2f} hr)")
    print(f"{'='*60}")
    print(f"\n  Optimiser:  {args.optimiser}")
    print(f"  STDP CV F1: {best_f1:.4f}")
    print(f"\n  {'Case':<6} {'CV F1':>8}  Parameters")
    print(f"  {'-'*50}")
    for clbl, res in wd_results.items():
        print(f"  {clbl:<6} {res['f1']:>8.4f}  {res['params']}")

    best_case = max(wd_results, key=lambda c: wd_results[c]["f1"])
    print(f"\n  Best case: {best_case} | "
          f"F1={wd_results[best_case]['f1']:.4f}")

    print(f"\n  Inspect results in: {OPT_DIR}/")
    print(f"\n  Next step — training + inference:")
    print(f"    python lsm_train_infer.py "
          f"--dataset {args.dataset} "
          f"--n_res {args.n_res} --n_driven {args.n_driven} "
          f"--opt_dir {OPT_DIR}")
