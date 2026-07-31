"""
Step 2 — Training + Inference + Plots
Requires lsm_optimise.py to have completed:
  OPT_DIR/stdp_params.json
  OPT_DIR/case_A/opt_params.json  (B, C, D also)
  OPT_DIR/arch.json

Output: RESULTS_DIR/
  case_A/  case_B/  case_C/  case_D/
    w_l2l2.npy, d_l2l2.npy, Ftr_l2.npy, Ftr_l3.npy ...
"""

import os, json, gc, time, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score,
)
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
CASE_NAMES = {
    "A": "Fixed weight + Fixed delay",
    "B": "Fixed weight + Random delay",
    "C": "Random weight + Fixed delay",
    "D": "Random weight + Random delay",
}

from lsm_generic import (
    load_dataset, CaseParams, NEURONS_PER_CLASS, CASE_CONFIGS,
    _build, make_input_map, inject_step, clear_input,
    reset_membrane, teach, clear_teacher, get_l2, get_l3, l3_summed,
)



# LOAD OPTIMISED PARAMS


def load_opt_params(opt_dir, case_label):
    stdp_path = os.path.join(opt_dir, "stdp_params.json")
    case_path = os.path.join(opt_dir, f"case_{case_label}", "opt_params.json")

    if not os.path.exists(stdp_path):
        raise FileNotFoundError(
            f"Not found: {stdp_path}\n"
            f"Run lsm_optimise.py first.")
    if not os.path.exists(case_path):
        raise FileNotFoundError(
            f"Not found: {case_path}\n"
            f"Run lsm_optimise.py first.")

    with open(stdp_path) as f:
        stdp = json.load(f)["stdp"]
    with open(case_path) as f:
        wd = json.load(f)

    params = CaseParams(
        label   = case_label,
        w_fixed = wd.get("w_fixed"),
        d_fixed = wd.get("d_fixed"),
        w_low   = wd.get("w_low"),
        w_high  = wd.get("w_high"),
        d_low   = wd.get("d_low"),
        d_high  = wd.get("d_high"),
    )
    return stdp, params


# TRAINING

def train_case(params, stdp, arch, Xtr, ytr,
               n_epochs=10, seed=10, out_dir=""):
    n_res  = arch["n_res"]
    n_cls  = arch["n_classes"]
    n_ch   = arch["n_features"]
    npc    = NEURONS_PER_CLASS
    T      = Xtr.shape[1]
    N      = len(Xtr)

    rng = np.random.default_rng(seed)
    nest.ResetKernel()
    nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time": False, "overwrite_files": True, "rng_seed": max(1, seed)})

    net    = _build(params, stdp, arch, "train", rng)
    L2, L3 = net["L2"], net["L3"]
    tg     = net["tg"]
    rec_l2 = net["rec_l2"]
    rec_l3 = net["rec_l3"]
    L2_ids = net["L2_ids"]
    L3_ids = net["L3_ids"]

    input_maps = [make_input_map(n_res, arch["n_driven"], rng)
                  for _ in range(n_ch)]
    Ie_h   = np.zeros(n_cls * npc, dtype=np.float32)
    Ftr_l2 = np.zeros((N, n_res),       dtype=np.float32)
    Ftr_l3 = np.zeros((N, n_cls * npc), dtype=np.float32)
    t_now  = 0.0
    accs   = []

    print(f"\n{'='*60}")
    print(f"  Training Case {params.label} — {CASE_NAMES[params.label]}")
    print(f"  {N} samples × {n_epochs} epochs | {params.summary()}")
    print(f"{'='*60}")
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        last    = (ep == n_epochs)
        correct = 0
        # Weighted shuffle: minority class appears more often
        from collections import Counter
        counts = Counter(ytr.tolist())
        weights = np.array([1.0/counts[int(y)] for y in ytr])
        weights /= weights.sum()
        rng_ep = np.random.default_rng(seed + ep)
        idx = rng_ep.choice(N, size=N, replace=True, p=weights)
        for pos, si in enumerate(idx):
            x, lbl = Xtr[si], int(ytr[si])
            reset_membrane(L2)
            reset_membrane(L3)
            nest.SetStatus(rec_l2, {"n_events": 0})
            nest.SetStatus(rec_l3, {"n_events": 0})
            # Progressive teacher: strong early epochs, weak later
            t_w = max(0.0, 600.0 * (1.0 - ep/n_epochs))
            if t_w > 0:
                for k in range(n_cls):
                    for nn in range(npc):
                        tc = nest.GetConnections(tg[k:k+1], L3[k*npc+nn:k*npc+nn+1])
                        if len(tc): tc.set({"weight": t_w})
            teach(tg, lbl, t_now, T, n_cls)
            for t in range(T):
                inject_step(L2, x[t], input_maps, arch["I_scale"])
                nest.Simulate(1.0)
                t_now += 1.0
            clear_input(L2, input_maps)
            clear_teacher(tg, n_cls)
            nest.Simulate(30.0)
            t_now += 30.0
            l2r = get_l2(rec_l2, L2_ids)
            l3f = get_l3(rec_l3, L3_ids)
            pred = int(np.argmax(l3_summed(l3f, n_cls)))
            if pred == lbl: correct += 1
            for k in range(n_cls * npc):
                Ie_h[k] += (-2.0 if l3f[k] > 10 else (1.0 if l3f[k] == 0 else 0.0))
                Ie_h[k]  = float(np.clip(Ie_h[k], -100.0, 30.0))
                nest.SetStatus(L3[k:k+1], {"I_e": Ie_h[k]})
            if last:
                Ftr_l2[si] = l2r
                Ftr_l3[si] = l3f
            if (pos + 1) % 50 == 0 or (pos + 1) == N:
                print(f"    Ep {ep}/{n_epochs} | {pos+1}/{N} | "
                      f"acc={correct/(pos+1)*100:.1f}%")
        acc = correct / N * 100
        accs.append(acc)
        w23 = np.array([d["weight"] for d in nest.GetStatus(net["c23"])])
        print(f"    Ep {ep} done acc={acc:.1f}%  "
              f"L2→L3 mean={w23.mean():.1f} std={w23.std():.1f}")

    # Diagnostics
    print(f"\n  Diagnostics:")
    print(f"    L2 train: mean={Ftr_l2.mean():.2f} "
          f"max={Ftr_l2.max():.0f} std={Ftr_l2.std():.2f}")
    if Ftr_l2.mean() < 0.1:
        print(f"    ⚠ L2 mean ≈ 0 — check I_scale ({arch['I_scale']}pA)")

    # Save everything before any ResetKernel
    w22 = np.array([d["weight"] for d in nest.GetStatus(net["c22"])])
    w23 = np.array([d["weight"] for d in nest.GetStatus(net["c23"])])
    d22 = np.array([d["delay"]  for d in nest.GetStatus(net["c22"])])
    d23 = np.array([d["delay"]  for d in nest.GetStatus(net["c23"])])

    np.save(os.path.join(out_dir, "w_l2l2.npy"),     w22)
    np.save(os.path.join(out_dir, "w_l2l3.npy"),     w23)
    np.save(os.path.join(out_dir, "d_l2l2.npy"),     d22)
    np.save(os.path.join(out_dir, "d_l2l3.npy"),     d23)
    np.save(os.path.join(out_dir, "Ftr_l2.npy"),     Ftr_l2)
    np.save(os.path.join(out_dir, "Ftr_l3.npy"),     Ftr_l3)
    np.save(os.path.join(out_dir, "Ie_h.npy"),       Ie_h)
    np.save(os.path.join(out_dir, "input_maps.npy"), np.array(input_maps))
    np.save(os.path.join(out_dir, "epoch_accs.npy"), np.array(accs))
    np.save(os.path.join(out_dir, "train_labels.npy"), ytr)

    print(f"    Done in {(time.time()-t0)/60:.1f} min")
    gc.collect()
    return Ftr_l2, Ftr_l3, accs



# INFERENCE

def infer_case(params, stdp, arch, Xte, out_dir, seed=10):
    n_res = arch["n_res"]
    n_cls = arch["n_classes"]
    n_ch  = arch["n_features"]
    T     = Xte.shape[1]

    w_l2l2 = np.load(os.path.join(out_dir, "w_l2l2.npy"))
    w_l2l3 = np.load(os.path.join(out_dir, "w_l2l3.npy"))
    d_l2l2 = np.load(os.path.join(out_dir, "d_l2l2.npy"))
    d_l2l3 = np.load(os.path.join(out_dir, "d_l2l3.npy"))
    imap   = list(np.load(os.path.join(out_dir, "input_maps.npy")))
    Ie_h_p = os.path.join(out_dir, "Ie_h.npy")
    Ie_h   = np.load(Ie_h_p) if os.path.exists(Ie_h_p) else None

    rng = np.random.default_rng(seed)
    nest.ResetKernel()
    nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time": False, "overwrite_files": True, "rng_seed": max(1, seed)})

    net = _build(params, stdp, arch, "inference", rng,
                 w_l2l2=w_l2l2, w_l2l3=w_l2l3)
    L2, L3 = net["L2"], net["L3"]
    rec_l2 = net["rec_l2"]
    rec_l3 = net["rec_l3"]
    L2_ids = net["L2_ids"]
    L3_ids = net["L3_ids"]

    # Apply trained delays
    res = nest.GetKernelStatus("resolution")
    for c, d in zip(net["c22"], d_l2l2):
        c.set({"delay": float(np.clip(d, res, None))})
    for c, d in zip(net["c23"], d_l2l3):
        c.set({"delay": float(np.clip(d, res, None))})

    # Apply homeostatic state
    if Ie_h is not None:
        for k in range(n_cls * NEURONS_PER_CLASS):
            nest.SetStatus(L3[k:k+1], {"I_e": float(Ie_h[k])})

    Nte    = len(Xte)
    Fte_l2 = np.zeros((Nte, n_res),                 dtype=np.float32)
    Fte_l3 = np.zeros((Nte, n_cls*NEURONS_PER_CLASS), dtype=np.float32)
    t_now  = 0.0

    print(f"  Inference Case {params.label} | {Nte} samples")
    for i in range(Nte):
        reset_membrane(L2)
        reset_membrane(L3)
        nest.SetStatus(rec_l2, {"n_events": 0})
        nest.SetStatus(rec_l3, {"n_events": 0})
        for t in range(T):
            inject_step(L2, Xte[i][t], imap, arch["I_scale"])
            nest.Simulate(1.0)
            t_now += 1.0
        clear_input(L2, imap)
        nest.Simulate(30.0)
        t_now += 30.0
        Fte_l2[i] = get_l2(rec_l2, L2_ids)
        Fte_l3[i] = get_l3(rec_l3, L3_ids)
        if (i+1) % 20 == 0 or (i+1) == Nte:
            print(f"    {i+1}/{Nte}  "
                  f"L2_mean={Fte_l2[:i+1].mean():.2f}  "
                  f"L3_mean={Fte_l3[:i+1].mean():.2f}")

    np.save(os.path.join(out_dir, "Fte_l2.npy"), Fte_l2)
    np.save(os.path.join(out_dir, "Fte_l3.npy"), Fte_l3)
    gc.collect()
    return Fte_l2, Fte_l3




def classify(Ftr_l2, Ftr_l3, ytr, Fte_l2, Fte_l3, yte,
             label, n_cls, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(n_cls)]
    npc = NEURONS_PER_CLASS

    # SampleNorm: normalize each sample by its max spike count
    # Fixes train/test activity mismatch caused by IH suppression during training
    # During training: teacher->L3->IH->L2 suppresses L2 (low activity)
    # During inference: no teacher -> no IH -> L2 fires freely (high activity)
    # SampleNorm makes features scale-invariant, fixing this mismatch
    def _snorm(F):
        mx = F.max(axis=1, keepdims=True) + 1e-6
        return F / mx
    def _znorm(F):
        m = F.mean(axis=1, keepdims=True)
        s = F.std(axis=1, keepdims=True) + 1e-6
        return (F - m) / s
    def _l2norm(F):
        from sklearn.preprocessing import normalize as sk_normalize
        return sk_normalize(F, norm='l2')
    def _apply_norm(F):
        nm = getattr(args, 'norm', 'sample')
        if nm == 'zscore': return _znorm(F)
        elif nm == 'l2':   return _l2norm(F)
        else:              return _snorm(F)

    def _score(y_true, y_pred, name):
        n_err  = int(np.sum(y_true != y_pred))
        f1_per = f1_score(y_true, y_pred, average=None,
                          labels=list(range(n_cls)),
                          zero_division=0).tolist()
        return {
            "decoder":      name,
            "y_pred":       y_pred,
            "cm":           confusion_matrix(y_true, y_pred,
                                             labels=list(range(n_cls))),
            "accuracy":     accuracy_score(y_true, y_pred),
            "cer":          n_err / len(y_true),
            "n_error":      n_err,
            "n_seq":        len(y_true),
            "f1":           f1_score(y_true, y_pred, average="macro",
                                     zero_division=0),
            "f1_per_class": f1_per,
            "precision":    precision_score(y_true, y_pred, average="macro",
                                            zero_division=0),
            "recall":       recall_score(y_true, y_pred, average="macro",
                                         zero_division=0),
        }

    # Argmax(L3)
    Fte_sum = Fte_l3.reshape(-1, n_cls, npc).sum(axis=2)
    Ftr_sum = Ftr_l3.reshape(-1, n_cls, npc).sum(axis=2)
    res_ax  = _score(yte, np.argmax(Fte_sum, axis=1), "Argmax(L3)")

    # Ridge(L3) — full L3 features
    sc1     = StandardScaler()
    clf_l3  = RidgeClassifier(alpha=args.ridge_alpha, class_weight="balanced")
    clf_l3.fit(sc1.fit_transform(Ftr_l3), ytr)
    res_rl3 = _score(yte, clf_l3.predict(sc1.transform(Fte_l3)), "Ridge(L3)")

    # Ridge(L2) — reservoir states (Paper comparison metric)
    # Use SampleNorm instead of StandardScaler to fix train/test activity mismatch
    clf_l2  = RidgeClassifier(alpha=args.ridge_alpha, class_weight="balanced")
    clf_l2.fit(_apply_norm(Ftr_l2), ytr)
    res_rl2 = _score(yte, clf_l2.predict(_apply_norm(Fte_l2)),
                     "Ridge(L2) [primary]")

    print(f"\n{'='*60}")
    print(f"  Case {label} — {CASE_NAMES[label]}")
    print(f"{'='*60}")
    for res in [res_ax, res_rl3, res_rl2]:
        print(f"\n  [{res['decoder']}]")
        print(f"  Acc={res['accuracy']:.4f}  "
              f"CER={res['cer']:.4f}  "
              f"({res['n_error']}/{res['n_seq']})  "
              f"F1={res['f1']:.4f}")
        print(classification_report(yte, res["y_pred"],
                                    target_names=class_names[:n_cls],
                                    labels=list(range(n_cls)),
                                    zero_division=0))
    return {"case": label, "y_true": yte,
            "argmax": res_ax, "ridge_l3": res_rl3, "ridge_l2": res_rl2}



def _savefig(fig, name, out_dir):
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), format=ext)
    plt.close()
    print(f"  Saved → {name}.svg/.pdf")


def plot_metrics(all_m, all_p, out_dir):
    x = np.arange(len(all_m)); w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Results — All Decoders × All Cases", fontweight="bold")
    for ax, metric in zip(axes, ["cer", "f1"]):
        lmap = {"cer": "CER ↓", "f1": "Macro F1 ↑"}
        for i, (key, lbl, alpha) in enumerate([
                ("argmax",   "Argmax(L3)",  0.45),
                ("ridge_l3", "Ridge(L3)",   0.70),
                ("ridge_l2", "Ridge(L2) ★", 1.00)]):
            vals = [m[key][metric] for m in all_m]
            cols = [_COL[m["case"]] for m in all_m]
            bars = ax.bar(x + (i-1)*w, vals, w, label=lbl,
                          color=cols, alpha=alpha, zorder=3,
                          edgecolor="black" if alpha == 1.0 else "none",
                          lw=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.01,
                        f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Case {m['case']}" for m in all_m])
        ax.set_ylim(0, 1.18); ax.set_ylabel(lmap[metric])
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    plt.tight_layout()
    _savefig(fig, "metrics_3decoders", out_dir)


def plot_epoch_accuracy(all_accs, all_params, n_cls, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    for p, accs in zip(all_params, all_accs):
        ax.plot(range(1, len(accs)+1), accs,
                color=_COL[p.label], marker=_MRK[p.label],
                lw=2, markersize=6,
                label=f"Case {p.label}: {p.summary()}")
    ax.axhline(100/n_cls, ls="--", color="gray", lw=1, alpha=0.6,
               label=f"Random ({100/n_cls:.0f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Accuracy (%)")
    ax.set_title("STDP Training Convergence — Argmax(L3)")
    ax.set_ylim(0, 105); ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig, "epoch_accuracy", out_dir)


def plot_confusion_matrices(all_m, all_p, n_cls, class_names, out_dir):
    for dec_key, dec_title in [
        ("argmax",   "Argmax(L3)"),
        ("ridge_l3", "Ridge(L3)"),
        ("ridge_l2", "Ridge(L2) [Primary]"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Confusion Matrices — {dec_title}",
                     fontsize=13, fontweight="bold")
        for ax, m in zip(axes.flat, all_m):
            res = m[dec_key]
            cm  = res["cm"]
            im  = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ticks = list(range(n_cls))
            ax.set_xticks(ticks); ax.set_xticklabels(
                class_names[:n_cls], rotation=35, ha="right", fontsize=8)
            ax.set_yticks(ticks); ax.set_yticklabels(
                class_names[:n_cls], fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            p = next(q for q in all_p if q.label == m["case"])
            ax.set_title(f"Case {m['case']}: {CASE_NAMES[m['case']]}\n"
                         f"{p.summary()}\n"
                         f"CER={res['cer']:.3f}  F1={res['f1']:.3f}",
                         fontsize=9)
            th = cm.max() / 2
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, str(cm[i, j]), ha="center",
                            va="center", fontsize=9,
                            color="white" if cm[i, j] > th else "black")
        plt.tight_layout()
        _savefig(fig, f"cm_{dec_key}", out_dir)


def plot_weight_delay_histograms(all_p, out_dir):
    for arr_name, ylabel, title_suffix in [
        ("w_l2l2", "Synaptic Weight (pA)", "Weight — L2→L2 Recurrent"),
        ("w_l2l3", "Synaptic Weight (pA)", "Weight — L2→L3 Readout"),
        ("d_l2l2", "Delay (ms)",           "Delay — L2→L2 Recurrent"),
        ("d_l2l3", "Delay (ms)",           "Delay — L2→L3 Readout"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"Synaptic {title_suffix} After STDP Training",
                     fontsize=12, fontweight="bold")
        for ax, p in zip(axes.flat, all_p):
            fpath = os.path.join(out_dir, f"case_{p.label}", f"{arr_name}.npy")
            if not os.path.exists(fpath):
                ax.text(0.5, 0.5, "Not found", ha="center",
                        va="center", transform=ax.transAxes)
                continue
            vals = np.load(fpath)
            vals = vals[vals > 0.01]
            ax.hist(vals, bins=40, color=_COL[p.label],
                    alpha=0.82, edgecolor="white", lw=0.4, zorder=3)
            ax.axvline(vals.mean(), color="black", ls="--", lw=1.5,
                       label=f"Mean={vals.mean():.2f}")
            ax.set_xlabel(ylabel); ax.set_ylabel("Count")
            ax.set_title(f"Case {p.label}: {CASE_NAMES[p.label]}\n"
                         f"{p.summary()}\n"
                         f"mean={vals.mean():.2f}  std={vals.std():.2f}",
                         fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
        plt.tight_layout()
        _savefig(fig, f"hist_{arr_name}", out_dir)


def plot_per_class_f1(all_m, all_p, n_cls, class_names, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle("Per-Class F1 — All Cases and Decoders",
                 fontsize=13, fontweight="bold")
    x = np.arange(n_cls); w = 0.25
    for ax, m in zip(axes.flat, all_m):
        p = next(q for q in all_p if q.label == m["case"])
        for i, (key, lbl, alpha) in enumerate([
                ("argmax",   "Argmax(L3)",  0.45),
                ("ridge_l3", "Ridge(L3)",   0.70),
                ("ridge_l2", "Ridge(L2) ★", 1.00)]):
            ax.bar(x + (i-1)*w, m[key]["f1_per_class"], w,
                   label=lbl, color=_COL[m["case"]], alpha=alpha,
                   zorder=3, edgecolor="black" if alpha == 1.0 else "none",
                   lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names[:n_cls], rotation=35,
                           ha="right", fontsize=8)
        ax.set_ylim(0, 1.2); ax.set_ylabel("F1")
        ax.set_title(f"Case {m['case']}: {CASE_NAMES[m['case']]}\n"
                     f"{p.summary()}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
        ax.axhline(1/n_cls, ls="--", color="gray", lw=1, alpha=0.5)
    plt.tight_layout()
    _savefig(fig, "per_class_f1", out_dir)


def plot_l2_activity(all_m, all_p, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Reservoir L2 Activity Distribution",
                 fontsize=13, fontweight="bold")
    for ax, split in zip(axes, ["train", "test"]):
        for m in all_m:
            p    = next(q for q in all_p if q.label == m["case"])
            cd   = os.path.join(out_dir, f"case_{m['case']}")
            fname = "Ftr_l2.npy" if split == "train" else "Fte_l2.npy"
            fp   = os.path.join(cd, fname)
            if not os.path.exists(fp): continue
            F    = np.load(fp)
            ax.plot(sorted(F.mean(axis=0), reverse=True),
                    color=_COL[m["case"]], lw=1.5,
                    label=f"Case {m['case']}: {p.summary()}")
        ax.set_xlabel("Neuron (sorted)"); ax.set_ylabel("Mean Spikes")
        ax.set_title(f"L2 Activity — {split.capitalize()}")
        ax.legend(fontsize=7); ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    _savefig(fig, "l2_activity", out_dir)


def plot_summary_table(all_m, all_p, out_dir):
    cols = ["Case", "Parameters",
            "Argmax CER↓", "Argmax F1↑",
            "Ridge(L3) CER↓", "Ridge(L3) F1↑",
            "Ridge(L2) CER↓★", "Ridge(L2) F1↑★"]
    rows = []
    for m in all_m:
        p = next(q for q in all_p if q.label == m["case"])
        rows.append([
            f"Case {m['case']}", p.summary(),
            f"{m['argmax']['cer']:.4f}",   f"{m['argmax']['f1']:.4f}",
            f"{m['ridge_l3']['cer']:.4f}", f"{m['ridge_l3']['f1']:.4f}",
            f"{m['ridge_l2']['cer']:.4f}", f"{m['ridge_l2']['f1']:.4f}",
        ])
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2, 2.2)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#1e3a5f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, m in enumerate(all_m):
        for j in range(len(cols)):
            tbl[(i+1, j)].set_facecolor(_COL[m["case"]] + "22")
    plt.tight_layout()
    _savefig(fig, "summary_table", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LSM Training + Inference + Plots")
    ap.add_argument("--dataset",     default="ECG")
    ap.add_argument("--n_res",       type=int,   default=200)
    ap.add_argument("--n_driven",    type=int,   default=20)
    ap.add_argument("--seed",         type=int,   default=42,
                    help="Random seed for NEST kernel and numpy RNG")
    ap.add_argument("--eval_metric",  type=str,   default="argmax",
                    choices=["argmax","ridge"],
                    help="Metric for best-epoch selection: argmax or ridge CV")
    ap.add_argument("--ridge_alpha",  type=float, default=1.0,
                    help="Ridge classifier regularization (default=1.0)")
    ap.add_argument("--norm",         type=str,   default="sample",
                    choices=["sample","zscore","l2","none"],
                    help="Feature normalization method")
    ap.add_argument("--burst_ms",     type=float, default=1.0,
                    help="Simulation ms per input timestep")
    ap.add_argument("--indeg22",     type=int,   default=None)
    ap.add_argument("--indeg23",     type=int,   default=None)
    ap.add_argument("--I_scale",     type=float, default=3000.0)
    ap.add_argument("--n_epochs",    type=int,   default=10)
    ap.add_argument("--opt_dir",     required=True,
                    help="Optimisation results dir from lsm_optimise.py")
    ap.add_argument("--results_dir", default=None,
                    help="Output dir (default: RESULTS_DATASET/)")
    ap.add_argument("--cases",       nargs="+",
                    default=["A", "B", "C", "D"])
    ap.add_argument("--skip_infer",  action="store_true",
                    help="Skip inference, just re-plot from saved features")
    ap.add_argument("--cache_dir",   default="datasets_cache")
    args = ap.parse_args()

    # Load arch from optimisation run if available
    arch_path = os.path.join(args.opt_dir, "arch.json")
    if os.path.exists(arch_path):
        with open(arch_path) as f:
            saved_arch = json.load(f)
        dataset = saved_arch.get("dataset", args.dataset)
    else:
        dataset = args.dataset

    RESULTS_DIR = args.results_dir or f"RESULTS_{dataset}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_total = time.time()
    print(f"\n{'='*60}")
    print(f"  LSM Training + Inference")
    print(f"  Dataset: {dataset} | Cases: {args.cases}")
    print(f"  Optimisation params from: {args.opt_dir}/")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*60}")

    # Load full dataset — test set used HERE for the first time
    print(f"\n--- Loading {dataset} ---")
    Xtr, ytr, Xte, yte, n_cls = load_dataset(
        dataset, cache_dir=args.cache_dir)
    n_ch = Xtr.shape[2]
    class_names = [str(i) for i in range(n_cls)]

    # Oversample minority class to balance STDP updates
    from collections import Counter
    counts = Counter(ytr.tolist())
    max_c  = max(counts.values())
    X_bal, y_bal = list(Xtr), list(ytr)
    import numpy as np
    rng_bal = np.random.default_rng(42)
    for cls, cnt in counts.items():
        if cnt < max_c:
            idx = np.where(ytr == cls)[0]
            extra = rng_bal.choice(idx, max_c - cnt, replace=True)
            X_bal.extend([Xtr[i] for i in extra])
            y_bal.extend([cls] * (max_c - cnt))
    Xtr = np.array(X_bal); ytr = np.array(y_bal)
    print(f'  After balancing: {dict(Counter(ytr.tolist()))}')
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

    # Save arch for reference
    with open(os.path.join(RESULTS_DIR, "arch.json"), "w") as f:
        json.dump(arch, f, indent=2)

    # Train + infer each case
    all_params  = []
    all_accs    = []
    all_metrics = []

    for clbl in args.cases:
        stdp, params = load_opt_params(args.opt_dir, clbl)
        all_params.append(params)
        case_dir = os.path.join(RESULTS_DIR, f"case_{clbl}")
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n  Case {clbl}: {params.summary()}")

        if args.skip_infer:
            print(f"  Skipping training+inference (--skip_infer)")
            Ftr_l2 = np.load(os.path.join(case_dir, "Ftr_l2.npy"))
            Ftr_l3 = np.load(os.path.join(case_dir, "Ftr_l3.npy"))
            Fte_l2 = np.load(os.path.join(case_dir, "Fte_l2.npy"))
            Fte_l3 = np.load(os.path.join(case_dir, "Fte_l3.npy"))
            accs   = list(np.load(os.path.join(case_dir, "epoch_accs.npy")))
        else:
            Ftr_l2, Ftr_l3, accs = train_case(
                params, stdp, arch, Xtr, ytr,
                n_epochs=args.n_epochs, seed=10, out_dir=case_dir)
            Fte_l2, Fte_l3 = infer_case(
                params, stdp, arch, Xte, case_dir, seed=10)

        all_accs.append(accs)
        m = classify(Ftr_l2, Ftr_l3, ytr,
                     Fte_l2, Fte_l3, yte,
                     clbl, n_cls, class_names)
        all_metrics.append(m)

    # Generate all plots
    print(f"\n{'='*60}")
    print(f"  Generating plots → {RESULTS_DIR}/")
    print(f"{'='*60}")
    plot_metrics(all_metrics, all_params, RESULTS_DIR)
    plot_epoch_accuracy(all_accs, all_params, n_cls, RESULTS_DIR)
    plot_confusion_matrices(all_metrics, all_params, n_cls,
                            class_names, RESULTS_DIR)
    plot_weight_delay_histograms(all_params, RESULTS_DIR)
    plot_per_class_f1(all_metrics, all_params, n_cls,
                      class_names, RESULTS_DIR)
    plot_l2_activity(all_metrics, all_params, RESULTS_DIR)
    plot_summary_table(all_metrics, all_params, RESULTS_DIR)

    # Save JSON
    out = {}
    for m in all_metrics:
        p = next(q for q in all_params if q.label == m["case"])
        out[m["case"]] = {
            "params":       p.summary(),
            "argmax_f1":    m["argmax"]["f1"],
            "argmax_cer":   m["argmax"]["cer"],
            "ridge_l3_f1":  m["ridge_l3"]["f1"],
            "ridge_l3_cer": m["ridge_l3"]["cer"],
            "ridge_l2_f1":  m["ridge_l2"]["f1"],
            "ridge_l2_cer": m["ridge_l2"]["cer"],
        }
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Final summary
    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {dataset}  ({total/3600:.2f} hr)")
    print(f"{'='*60}")
    print(f"\n  {'Case':<6} {'Params':<34} "
          f"{'Argmax':>10} {'':>8} {'R(L2)★':>9} {'':>8}")
    print(f"  {'':6} {'':34} "
          f"{'CER↓':>10} {'F1↑':>8} {'CER↓':>9} {'F1↑':>8}")
    print(f"  {'-'*78}")
    for m in all_metrics:
        p = next(q for q in all_params if q.label == m["case"])
        print(f"  {m['case']:<6} {p.summary():<34} "
              f"{m['argmax']['cer']:>10.4f} "
              f"{m['argmax']['f1']:>8.4f} "
              f"{m['ridge_l2']['cer']:>9.4f} "
              f"{m['ridge_l2']['f1']:>8.4f}")

    best = max(all_metrics, key=lambda m: m["argmax"]["f1"])
    bp   = next(p for p in all_params if p.label == best["case"])
    print(f"\n  Best Argmax(L3): Case {best['case']} | {bp.summary()}")
    print(f"  F1={best['argmax']['f1']:.4f}  "
          f"CER={best['argmax']['cer']:.4f}")
    print(f"\n  All results: {RESULTS_DIR}/")
