"""
LSM — Genetic Algorithm Optimisation
"""

import os, json, gc, time, argparse, warnings, random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
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


# STDP SEARCH SPACE


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


# 5-FOLD CV FITNESS (same as lsm_generic)


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
                Ie_h[k] += (-2.0 if l3f[k] > 10 else (1.0 if l3f[k] == 0 else 0.0))
                Ie_h[k]  = float(np.clip(Ie_h[k], -100.0, 30.0))
                nest.SetStatus(L3[k:k+1], {"I_e": Ie_h[k]})
            if last:
                Ftr[si] = l2r

    # Save synapses before ResetKernel
    from lsm_generic import get_syn_tuples
    syn22 = get_syn_tuples(net["c22"]) if hasattr(nest, 'GetStatus') else None

    w22 = np.array([d["weight"] for d in nest.GetStatus(net["c22"])])

    rng2 = np.random.default_rng(seed + 100)
    nest.ResetKernel()
    nest.set_verbosity(0)
    nest.SetKernelStatus({"print_time": False, "rng_seed": max(1, seed + 100)})

    p2   = CaseParams("_", w_fixed=float(w_init), d_fixed=float(d_mean))
    net2 = _build(p2, stdp, arch, "inference", rng2,
                  w_l2l2=w22, w_l2l3=None)
    L2_2    = net2["L2"]
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
        kf   = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        f1s  = []
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


# GENETIC ALGORITHM

def run_ga(fitness_fn, bounds, popsize=10, n_gen=20,
           mutation_rate=0.2, crossover_rate=0.5,
           seed=0, label=""):
    """
    Genetic Algorithm optimiser.

    Mirrors reference GA code structure:
      - Random initialisation across full parameter space
      - Tournament selection: top 50% survive each generation
      - Uniform crossover: each gene randomly from parent1 or parent2
      - Uniform mutation: randomly redraw gene within bounds
      - Elitism: best individual always preserved

    Fitness: 5-fold CV F1 (higher = better)
    Parameters normalised to [0,1] internally.

    Args:
        fitness_fn:     callable(x) → float
        bounds:         list of (low, high) per parameter
        popsize:        population size (default 10)
        n_gen:          number of generations (default 20)
        mutation_rate:  probability per gene to mutate (default 0.2)
        crossover_rate: probability of taking gene from parent1 (default 0.5)
        seed:           random seed
        label:          label for print output
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    lo       = np.array([b[0] for b in bounds])
    hi       = np.array([b[1] for b in bounds])
    n_params = len(bounds)

    # Normalise helpers
    norm   = lambda x: (x - lo) / (hi - lo + 1e-12)
    denorm = lambda x: lo + x * (hi - lo)


    def random_individual():
        """Random individual uniformly across [0,1] (normalised space)."""
        return rng.random(n_params)

    def crossover(p1, p2):
        """
        Uniform crossover — each gene randomly from p1 or p2.
        Mirrors: child[k] = random.choice([parent1[k], parent2[k]])
        """
        mask  = rng.random(n_params) < crossover_rate
        return np.where(mask, p1, p2)

    def mutate(ind):
        """
        Uniform mutation — randomly redraw gene within [0,1].
        Mirrors: if random() < mutation_rate: new_ind[k] = random.uniform(...)
        """
        new_ind = ind.copy()
        for i in range(n_params):
            if rng.random() < mutation_rate:
                new_ind[i] = rng.random()
        return np.clip(new_ind, 0.0, 1.0)

    population = [random_individual() for _ in range(popsize)]
    scores     = [fitness_fn(denorm(ind)) for ind in population]

    best_idx = int(np.argmax(scores))
    best_x   = population[best_idx].copy()
    best_f   = scores[best_idx]
    hist     = [best_f]

    print(f"\n  [GA] {label} | gen={n_gen} pop={popsize} "
          f"mut={mutation_rate} cross={crossover_rate}")
    print(f"  [GA] Init: best F1={best_f:.4f}  "
          f"mean={np.mean(scores):.4f}  "
          f"std={np.std(scores):.4f}")

    for gen in range(1, n_gen + 1):

        ranked    = sorted(zip(scores, population),
                           key=lambda x: x[0], reverse=True)
        n_keep    = max(2, popsize // 2)
        survivors = [ind for _, ind in ranked[:n_keep]]
        # Scores of survivors (already evaluated)
        surv_scores = [s for s, _ in ranked[:n_keep]]


        children = []
        while len(children) < popsize - n_keep:
            p1, p2 = random.sample(survivors, 2)
            child  = crossover(p1, p2)
            child  = mutate(child)
            children.append(child)

        child_scores = []
        for child in children:
            f = fitness_fn(denorm(child))
            child_scores.append(f)

        population = survivors + children
        scores     = surv_scores + child_scores

        gen_best = int(np.argmax(scores))
        if scores[gen_best] > best_f:
            best_f = scores[gen_best]
            best_x = population[gen_best].copy()

        hist.append(best_f)
        print(f"  [GA] Gen {gen:3d}/{n_gen} | "
              f"best F1={best_f:.4f} | "
              f"gen_best={scores[gen_best]:.4f} | "
              f"mean={np.mean(scores):.4f}")

    print(f"  [GA] Done | best F1={best_f:.4f}")
    return denorm(best_x), best_f, hist


# CONVERGENCE PLOT

def plot_ga_convergence(stdp_hist, wd_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GA Optimisation Convergence — 5-fold CV F1",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    if stdp_hist:
        ax.plot(stdp_hist, color="#2563EB", lw=2, marker="o", markersize=5,
                label=f"Best={max(stdp_hist):.4f}")
        ax.fill_between(range(len(stdp_hist)), stdp_hist,
                        alpha=0.12, color="#2563EB")
    ax.set_xlabel("Generation")
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
                    lw=2, markersize=5,
                    label=f"Case {clbl} (best={max(hist):.3f})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1b — Weight/Delay Optimisation\n"
                 "(per-case w/d scalar or distribution)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"ga_convergence.{ext}"),
                    format=ext)
    plt.close()
    print(f"  Saved → ga_convergence.svg/.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LSM Genetic Algorithm Optimisation")
    ap.add_argument("--dataset",      default="ECG")
    ap.add_argument("--n_res",        type=int,   default=200)
    ap.add_argument("--n_driven",     type=int,   default=80)
    ap.add_argument("--I_scale",      type=float, default=3000.0)
    ap.add_argument("--n_folds",      type=int,   default=5,
                    help="CV folds for fitness (default 5, Paper 3 method)")
    ap.add_argument("--ga_epochs",    type=int,   default=3,
                    help="Training epochs per CV fold during GA")
    ap.add_argument("--ga_gen_stdp",  type=int,   default=15,
                    help="GA generations for STDP phase (default 15)")
    ap.add_argument("--ga_gen_wd",    type=int,   default=10,
                    help="GA generations for w/d phase (default 10)")
    ap.add_argument("--ga_pop",       type=int,   default=10,
                    help="Population size (default 10)")
    ap.add_argument("--ga_mut",       type=float, default=0.2,
                    help="Mutation rate per gene (default 0.2)")
    ap.add_argument("--ga_cross",     type=float, default=0.5,
                    help="Crossover rate (default 0.5)")
    ap.add_argument("--ga_dir",       default=None,
                    help="Output dir (default: GA_DATASET/)")
    ap.add_argument("--cache_dir",    default="datasets_cache")
    args = ap.parse_args()

    GA_DIR = args.ga_dir or f"GA_{args.dataset}"
    os.makedirs(GA_DIR, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  LSM — Genetic Algorithm Optimisation")
    print(f"  Dataset: {args.dataset} | n_res={args.n_res} "
          f"n_driven={args.n_driven}")
    print(f"  pop={args.ga_pop} | mut={args.ga_mut} | "
          f"cross={args.ga_cross}")
    print(f"  gen_stdp={args.ga_gen_stdp} | gen_wd={args.ga_gen_wd}")
    print(f"  {args.n_folds}-fold CV | epochs={args.ga_epochs}")
    print(f"  Output: {GA_DIR}/")
    print(f"{'='*60}")

    # Load training data only — test set not used during GA
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

    #Phase 1a: STDP optimisation
    print(f"\n{'='*60}")
    print(f"  Phase 1a — STDP Optimisation (GA)")
    print(f"  7 params: {STDP_KEYS}")
    print(f"  Fixed: w=150pA d=1ms | {args.n_folds}-fold CV on Xtr only")
    print(f"{'='*60}")

    def stdp_fit(x):
        return evaluate(150.0, 1.0, x_to_stdp(x), arch, Xtr, ytr,
                        n_epochs=args.ga_epochs, n_folds=args.n_folds)

    best_stdp_arr, best_stdp_f1, stdp_hist = run_ga(
        stdp_fit, STDP_BOUNDS,
        popsize=args.ga_pop, n_gen=args.ga_gen_stdp,
        mutation_rate=args.ga_mut, crossover_rate=args.ga_cross,
        seed=0, label="STDP")

    best_stdp = x_to_stdp(best_stdp_arr)

    with open(os.path.join(GA_DIR, "stdp_params.json"), "w") as f:
        json.dump({"stdp": best_stdp, "f1": best_stdp_f1,
                   "optimiser": "GA"}, f, indent=2)

    print(f"\n  STDP best CV F1 = {best_stdp_f1:.4f}")
    print(f"  Saved → {GA_DIR}/stdp_params.json")
    for k, v in best_stdp.items():
        if k != "synapse_model":
            print(f"    {k}: {v:.6f}")

    # Phase 1b: weight/delay per case 
    wd_results = {}

    for clbl, cfg in CASE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Phase 1b — Case {clbl}: {cfg['names']} (GA)")
        print(f"{'='*60}")

        def wd_fit(x, cfg=cfg):
            w = float(np.clip(cfg["w_fn"](x), 10.0,
                              best_stdp["Wmax"] * 0.95))
            d = float(np.clip(cfg["d_fn"](x), 0.1, 25.0))
            return evaluate(w, d, best_stdp, arch, Xtr, ytr,
                           n_epochs=args.ga_epochs, n_folds=args.n_folds)

        best_arr, best_f1, hist = run_ga(
            wd_fit, cfg["bounds"],
            popsize=args.ga_pop, n_gen=args.ga_gen_wd,
            mutation_rate=args.ga_mut, crossover_rate=args.ga_cross,
            seed=ord(clbl), label=f"Case {clbl}")

        opt_p = cfg["save"](best_arr)
        wd_results[clbl] = {"params": opt_p, "f1": best_f1,
                             "hist": list(hist)}

        case_dir = os.path.join(GA_DIR, f"case_{clbl}")
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "opt_params.json"), "w") as f:
            json.dump(opt_p, f, indent=2)

        print(f"  Case {clbl}: CV F1={best_f1:.4f} | {opt_p}")
        print(f"  Saved → {GA_DIR}/case_{clbl}/opt_params.json")


    plot_ga_convergence(stdp_hist, wd_results, GA_DIR)

    #Save full comparison JSON
    summary = {
        "optimiser":    "GA",
        "dataset":      args.dataset,
        "stdp_f1":      best_stdp_f1,
        "runtime_hr":   (time.time() - t_start) / 3600,
        "cases": {
            clbl: {"f1": wd_results[clbl]["f1"],
                   "params": wd_results[clbl]["params"]}
            for clbl in CASE_CONFIGS
        }
    }
    with open(os.path.join(GA_DIR, "ga_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  GA COMPLETE — {args.dataset}  "
          f"({total/3600:.2f} hr)")
    print(f"{'='*60}")
    print(f"\n  STDP CV F1 = {best_stdp_f1:.4f}")
    print(f"\n  {'Case':<6} {'CV F1':>8}  Parameters")
    print(f"  {'-'*50}")
    for clbl, res in wd_results.items():
        print(f"  {clbl:<6} {res['f1']:>8.4f}  {res['params']}")

    best_case = max(wd_results, key=lambda c: wd_results[c]["f1"])
    print(f"\n  Best case: {best_case} | "
          f"F1={wd_results[best_case]['f1']:.4f}")

    print(f"\n  Next step — training + inference:")
    print(f"    python lsm_generic.py --dataset {args.dataset} "
          f"--n_res {args.n_res} --n_driven {args.n_driven} "
          f"--skip_es --es_dir {GA_DIR}")
