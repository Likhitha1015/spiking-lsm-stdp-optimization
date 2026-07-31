"""
Generic LSM — ES Optimisation + Training + Inference
======================================================
Handles any multivariate time series dataset from:
  https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing

Supports all Paper 3 datasets: ROBOT, JPVOW, WAF, LIB, SWE, CHLO,
PHAL, CHAR, UWAV, NET, KICK, WALK, PEMS, CMU, AUS

Usage:
  python lsm_generic.py --dataset ROBOT
  python lsm_generic.py --dataset JPVOW
  python lsm_generic.py --dataset WAF

Download datasets:
  git clone https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing
  # datasets are in the datasets/ folder as .pkl files

Three decoders reported:
  1. Argmax(L3)  — pure spiking readout (PRIMARY)
  2. Ridge(L3)   — linear on L3 population counts
  3. Ridge(L2)   — linear on reservoir states

CER metric used (Steiner et al. TNNLS 2023).
"""

import os, json, gc, time, argparse, warnings
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
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score,
)
import scipy.optimize
import nest

warnings.filterwarnings("ignore")

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

_COL = {"A":"#2563EB","B":"#16A34A","C":"#D97706","D":"#DC2626"}
_MRK = {"A":"o",     "B":"s",     "C":"^",     "D":"D"}

CASE_NAMES = {
    "A": "Fixed weight + Fixed delay",
    "B": "Fixed weight + Random delay",
    "C": "Random weight + Fixed delay",
    "D": "Random weight + Random delay",
}

NEURONS_PER_CLASS = 10


# ===========================================================================
# 1. DATA LOADING — FilippoMB pkl format
# ===========================================================================

# Zenodo dataset URLs (from FilippoMB reservoir_computing/datasets.py)
ZENODO_URLS = {
    "ECG":   "https://zenodo.org/records/10839881/files/ECG_2D.npz?download=1",
    "ROBOT": "https://zenodo.org/records/10852893/files/Robot.npz?download=1",
    "JPVOW": "https://zenodo.org/records/10837602/files/Japanese_Vowels.npz?download=1",
    "WAF":   "https://zenodo.org/records/10839966/files/Wafer.npz?download=1",
    "LIB":   "https://zenodo.org/records/10852531/files/LIB.npz?download=1",
    "CHAR":  "https://zenodo.org/records/10852786/files/CHAR.npz?download=1",
    "UWAV":  "https://zenodo.org/records/10852667/files/UWAVE.npz?download=1",
    "NET":   "https://zenodo.org/records/10840246/files/NET.npz?download=1",
    "CHLO":  "https://zenodo.org/records/10840284/files/CHLO.npz?download=1",
    "PHAL":  "https://zenodo.org/records/10852613/files/PHAL.npz?download=1",
    "SWE":   "https://zenodo.org/records/10840000/files/SwedishLeaf.npz?download=1",
    "ARAB":  "https://zenodo.org/records/10852747/files/ARAB.npz?download=1",
}


def load_dataset(name: str,
                 data_dir: str = "datasets",
                 cache_dir: str = "datasets_cache",
                 ) -> Tuple[np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, int]:
    """
    Load dataset from Zenodo (auto-download and cache as .npz).
    No login required — direct Zenodo URLs.

    Returns:
      X_train: (N_train, T, n_features)  normalised to [0,1]
      y_train: (N_train,)  integer labels 0..N_classes-1
      X_test:  (N_test, T, n_features)
      y_test:  (N_test,)
      n_classes: int
    """
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}.npz")

    # Download if not cached
    if not os.path.exists(cache_path):
        if name not in ZENODO_URLS:
            raise ValueError(
                f"Unknown dataset: {name}\n"
                f"Available: {list(ZENODO_URLS.keys())}"
            )
        url = ZENODO_URLS[name]
        print(f"  Downloading {name} from Zenodo...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"  Saved to: {cache_path}")
    else:
        print(f"  Loading cached {name} from {cache_path}")

    # Load npz
    data = np.load(cache_path, allow_pickle=True)
    print(f"  NPZ keys: {list(data.keys())}")

    # Get arrays — try multiple key name conventions
    def _get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(
            f"None of {keys} found. Available keys: {list(d.keys())}")

    X_tr = _get(data, "X_train", "Xtr", "x_train", "X_tr")
    X_te = _get(data, "X_test",  "Xte", "x_test",  "X_te")
    y_tr = _get(data, "Y_train", "Ytr", "ytr", "y_train", "Y_tr")
    y_te = _get(data, "Y_test",  "Yte", "yte", "y_test",  "Y_te")

    # Handle shape: ensure (N, T, F)
    X_tr = np.array(X_tr, dtype=np.float32)
    X_te = np.array(X_te, dtype=np.float32)

    if X_tr.ndim == 2:
        # (N, T) univariate → (N, T, 1)
        X_tr = X_tr[:, :, np.newaxis]
        X_te = X_te[:, :, np.newaxis]
    elif X_tr.ndim == 3 and X_tr.shape[1] < X_tr.shape[2]:
        # (N, F, T) → (N, T, F)
        X_tr = X_tr.transpose(0, 2, 1)
        X_te = X_te.transpose(0, 2, 1)

    # Normalise per-sample per-feature to [0,1] (original pipeline)
    # Note: z-score produces negative values which hyperpolarize LIF neurons
    # Per-sample MinMax ensures all injected currents are excitatory
    for X in [X_tr, X_te]:
        for i in range(len(X)):
            for f in range(X.shape[2]):
                lo, hi = X[i, :, f].min(), X[i, :, f].max()
                if hi - lo > 1e-8:
                    X[i, :, f] = (X[i, :, f] - lo) / (hi - lo)
                else:
                    X[i, :, f] = 0.0

    # Encode labels to integers 0..N-1
    le    = LabelEncoder()
    y_tr  = le.fit_transform(np.array(y_tr).ravel()).astype(int)
    y_te  = le.transform(np.array(y_te).ravel()).astype(int)
    n_cls = len(le.classes_)

    print(f"  {name}: X_train={X_tr.shape} X_test={X_te.shape} "
          f"classes={n_cls}")
    print(f"  Train: {dict(zip(*np.unique(y_tr,return_counts=True)))}")
    print(f"  Test:  {dict(zip(*np.unique(y_te,return_counts=True)))}")
    return X_tr, y_tr, X_te, y_te, n_cls


# ===========================================================================
# 2. CASE PARAMS
# ===========================================================================

@dataclass
class CaseParams:
    label:   str
    w_fixed: Optional[float] = None
    d_fixed: Optional[float] = None
    w_low:   Optional[float] = None
    w_high:  Optional[float] = None
    d_low:   Optional[float] = None
    d_high:  Optional[float] = None

    def sample_weights(self, rng, n):
        return (np.full(n, self.w_fixed) if self.w_fixed is not None
                else rng.uniform(self.w_low, self.w_high, n))

    def sample_delays(self, rng, n):
        return (np.full(n, self.d_fixed) if self.d_fixed is not None
                else rng.uniform(self.d_low, self.d_high, n))

    def summary(self):
        w = (f"w={self.w_fixed:.1f}pA" if self.w_fixed is not None
             else f"w~U[{self.w_low:.0f},{self.w_high:.0f}]pA")
        d = (f"d={self.d_fixed:.2f}ms" if self.d_fixed is not None
             else f"d~U[{self.d_low:.1f},{self.d_high:.1f}]ms")
        return f"{w}, {d}"


CASE_CONFIGS = {
    "A": {
        "names":  ["w_fixed", "d_fixed"],
        "x0":     [150.0, 1.0],
        "bounds": [(20.0, 580.0), (0.1, 20.0)],
        "w_fn":   lambda x: x[0],
        "d_fn":   lambda x: x[1],
        "save":   lambda x: {"w_fixed": x[0], "d_fixed": x[1]},
        "load":   lambda d: CaseParams("A",
                            w_fixed=d.get("w_fixed"),
                            d_fixed=d.get("d_fixed")),
    },
    "B": {
        "names":  ["w_fixed", "d_low", "d_range"],
        "x0":     [150.0, 1.0, 10.0],
        "bounds": [(20.0, 580.0), (0.1, 15.0), (3.0, 20.0)],
        "w_fn":   lambda x: x[0],
        "d_fn":   lambda x: x[1] + x[2] / 2,
        "save":   lambda x: {"w_fixed": x[0],
                              "d_low":   x[1],
                              "d_high":  x[1] + x[2]},
        "load":   lambda d: CaseParams("B",
                            w_fixed=d.get("w_fixed"),
                            d_low=d.get("d_low"),
                            d_high=d.get("d_high")),
    },
    "C": {
        "names":  ["w_low", "w_range", "d_fixed"],
        "x0":     [100.0, 200.0, 1.0],
        "bounds": [(10.0, 400.0), (50.0, 400.0), (0.1, 20.0)],
        "w_fn":   lambda x: x[0] + x[1] / 2,
        "d_fn":   lambda x: x[2],
        "save":   lambda x: {"w_low":   x[0],
                              "w_high":  x[0] + x[1],
                              "d_fixed": x[2]},
        "load":   lambda d: CaseParams("C",
                            w_low=d.get("w_low"),
                            w_high=d.get("w_high"),
                            d_fixed=d.get("d_fixed")),
    },
    "D": {
        "names":  ["w_low", "w_range", "d_low", "d_range"],
        "x0":     [100.0, 200.0, 1.0, 10.0],
        "bounds": [(10.0, 400.0), (50.0, 400.0),
                   (0.1,  15.0),  (3.0,  20.0)],
        "w_fn":   lambda x: x[0] + x[1] / 2,
        "d_fn":   lambda x: x[2] + x[3] / 2,
        "save":   lambda x: {"w_low":   x[0],
                              "w_high":  x[0] + x[1],
                              "d_low":   x[2],
                              "d_high":  x[2] + x[3]},
        "load":   lambda d: CaseParams("D",
                            w_low=d.get("w_low"),
                            w_high=d.get("w_high"),
                            d_low=d.get("d_low"),
                            d_high=d.get("d_high")),
    },
}


def x_to_stdp(x) -> Dict:
    return {
        "synapse_model":    "stdp_triplet_synapse",
        "Aplus":            float(x[0]),
        "Aminus":           float(x[1]),
        "tau_plus":         float(x[2]),
        "Aplus_triplet":    float(x[3]),
        "Aminus_triplet":   float(x[4]),
        "tau_plus_triplet": float(x[5]),
        "Wmax":             float(x[6]),
    }


# ===========================================================================
# 3. NETWORK BUILDER
# ===========================================================================

def _build(params, stdp, arch, mode, rng,
           w_l2l2=None, w_l2l3=None):
    n_res = arch["n_res"]
    n_cls = arch["n_classes"]
    npc   = NEURONS_PER_CLASS
    n_l3  = n_cls * npc

    nest.SetDefaults("iaf_psc_alpha", {
        "I_e": 0.0, "tau_m": 20.0,
        "V_th": -55.0, "V_reset": -70.0,
        "E_L": -70.0, "C_m": 250.0,
    })

    L2 = nest.Create("iaf_psc_alpha", n_res)
    L3 = nest.Create("iaf_psc_alpha", n_l3)
    NZ = nest.Create("poisson_generator", 1)
    IH = nest.Create("iaf_psc_alpha", max(10, n_res // 10))


    rec_l2 = nest.Create("spike_recorder")
    rec_l3 = nest.Create("spike_recorder")
    nest.Connect(L2, rec_l2)
    nest.Connect(L3, rec_l3)
    tg = nest.Create("spike_generator", n_cls)

    min_d  = nest.GetKernelStatus("resolution")
    # Delays sampled per-synapse after connections created
    # (not collapsed to mean — preserves true heterogeneity for Cases B/D)

    wmax   = stdp.get("Wmax", 580.0)
    w_init = float(np.clip(
        params.w_fixed if params.w_fixed is not None
        else (params.w_low + params.w_high) / 2,
        1.0, wmax))

    stdp_res = {k: v for k, v in stdp.items()}
    stdp_rdo = {
        "synapse_model":    "stdp_triplet_synapse",
        "Aplus":            0.01,
        "Aplus_triplet":    0.005,
        "Aminus":           0.02,
        "Aminus_triplet":   0.008,
        "tau_plus":         20.0,
        "tau_plus_triplet": 100.0,
        "Wmax":             wmax,
    }

    indeg22 = min(arch["indeg22"], n_res - 1)
    indeg23 = arch["indeg23"]

    if mode == "train":
        # Noise
        nest.SetStatus(NZ, {"rate": 10.0})
        for t in range(min(max(2, n_res // 20), n_res)):
            nest.Connect(NZ, L2[t:t+1],
                         syn_spec={"synapse_model":"static_synapse",
                                   "weight": 80.0})
        # IH: gentle -500pA (not -3000pA which kills L2 activity)
        nest.Connect(L3, IH, syn_spec={"weight": 500.0})
        nest.Connect(IH, L2, syn_spec={"weight": -50.0})

        # Lateral inhibition between classes
        for ci in range(n_cls):
            for cj in range(n_cls):
                if ci == cj: continue
                for ni in range(npc):
                    for nj in range(npc):
                        nest.Connect(
                            L3[ci*npc+ni:ci*npc+ni+1],
                            L3[cj*npc+nj:cj*npc+nj+1],
                            syn_spec={"synapse_model":"static_synapse",
                                      "weight": -2000.0})

        # L2→L2 STDP — placeholder delay, set per-synapse AFTER all connects
        nest.Connect(L2, L2,
                     conn_spec={"rule":"fixed_indegree",
                                "indegree":indeg22,
                                "allow_autapses":False},
                     syn_spec={**stdp_res,
                               "weight": w_init,
                               "delay":  min_d})
        # L2→L3 STDP — placeholder delay
        nest.Connect(L2, L3,
                     conn_spec={"rule":"fixed_indegree",
                                "indegree":indeg23},
                     syn_spec={**stdp_rdo,
                               "weight": 150.0,
                               "delay":  min_d})
        # Teacher
        for k in range(n_cls):
            for n in range(npc):
                nest.Connect(tg[k:k+1],
                             L3[k*npc+n:k*npc+n+1],
                             syn_spec={"synapse_model":"static_synapse",
                                       "weight": 600.0})
    else:
        # Same fixed_indegree structure as training
        # Same rng_seed → same topology → weights loaded by position
        nest.Connect(L2, L2,
                     conn_spec={"rule":"fixed_indegree",
                                "indegree":indeg22,
                                "allow_autapses":False},
                     syn_spec={"synapse_model":"static_synapse",
                               "delay": min_d})
        nest.Connect(L2, L3,
                     conn_spec={"rule":"fixed_indegree",
                                "indegree":indeg23},
                     syn_spec={"synapse_model":"static_synapse",
                               "delay": min_d})

    # GetConnections AFTER all connects — avoids descriptor invalidation
    c22 = nest.GetConnections(L2, L2)
    c23 = nest.GetConnections(L2, L3)

    def _sw(conns, arr, wm):
        for c, w in zip(conns, arr):
            c.set({"weight": float(np.clip(w, 0.0, wm))})

    def _sd(conns, arr):
        """Set per-synapse delays."""
        for c, d in zip(conns, arr):
            c.set({"delay": float(np.clip(d, min_d, None))})

    if mode == "train":
        # Weights
        _sw(c22, params.sample_weights(rng, len(c22)), wmax)
        _sw(c23, np.full(len(c23), 150.0), wmax)
        # Per-synapse delays — true heterogeneity for Cases B/D
        _sd(c22, params.sample_delays(rng, len(c22)))
        _sd(c23, params.sample_delays(rng, len(c23)))
    else:
        # Inference: load trained weights by position (same seed = same topology)
        if w_l2l2 is not None: _sw(c22, w_l2l2, wmax)
        if w_l2l3 is not None: _sw(c23, w_l2l3, wmax)
        # Load trained delays
        if w_l2l2 is not None and hasattr(w_l2l2, '__len__'):
            # delays saved separately
            pass  # delays already set via d_l2l2/d_l2l3 in infer_case

    return {"L2":L2,"L3":L3,"NZ":NZ,"IH":IH,"tg":tg,
            "rec_l2":rec_l2,"rec_l3":rec_l3,
            "c22":c22,"c23":c23,
            "L2_ids":L2.tolist(),"L3_ids":L3.tolist()}


# ===========================================================================
# 4. INPUT INJECTION + HELPERS
# ===========================================================================

def make_input_map(n_res, n_driven, rng):
    return rng.choice(n_res, size=n_driven, replace=False)


def inject_step(L2, vals, input_maps, I_scale=800.0):
    """vals: (n_features,) array for one time step."""
    for ch, imap in enumerate(input_maps):
        I_e = float(vals[ch]) * I_scale
        for idx in imap:
            nest.SetStatus(L2[int(idx):int(idx)+1], {"I_e": I_e})


def clear_input(L2, input_maps):
    for imap in input_maps:
        for idx in imap:
            nest.SetStatus(L2[int(idx):int(idx)+1], {"I_e": 0.0})

# ===========================================================================
# TEMPORAL ENCODING (place coding — one spike generator per timestep×channel)
# Matches Verilog implementation: input neuron k fires at timestep k
# ===========================================================================
def build_temporal_input(X_sample, n_res, n_driven, rng, I_scale=3000.0,
                          threshold=0.0, t_offset=0.5):
    """
    Create spike generators for temporal (place) encoding.
    Each (ch, t) pair gets a spike generator that fires at time t+t_offset ms
    if amplitude > threshold. Amplitude scales the injected current weight.
    
    Returns: (spike_gens, sg_maps) where sg_maps maps each generator to
             reservoir neurons via make_input_map.
    """
    T, n_ch = X_sample.shape
    spike_gens = []
    sg_maps = []
    sg_weights = []
    
    for ch in range(n_ch):
        for t in range(T):
            amp = float(X_sample[t, ch])
            sg = nest.Create("spike_generator", 1)
            if amp > threshold:
                spike_time = float(t) + t_offset
                nest.SetStatus(sg, {"spike_times": [spike_time]})
            else:
                nest.SetStatus(sg, {"spike_times": []})
            imap = rng.choice(n_res, size=n_driven, replace=False)
            weight = max(amp * I_scale, 1.0)  # amplitude → synaptic weight
            spike_gens.append(sg)
            sg_maps.append(imap)
            sg_weights.append(weight)
    
    return spike_gens, sg_maps, sg_weights

def connect_temporal_input(spike_gens, sg_maps, sg_weights, L2):
    """Connect spike generators to reservoir neurons."""
    for sg, imap, w in zip(spike_gens, sg_maps, sg_weights):
        for idx in imap:
            nest.Connect(sg, L2[int(idx):int(idx)+1],
                        syn_spec={"synapse_model": "static_synapse",
                                  "weight": float(w),
                                  "delay": 0.1})

def run_temporal_sample(L2, rec_l2, X_sample, n_res, n_driven, rng,
                         I_scale=3000.0, t_offset=0.5):
    """
    Full temporal encoding pass for one sample.
    Creates spike generators, connects them, simulates T+2ms, returns spike count.
    NOTE: Must call nest.ResetNetwork() between samples to clear spike generators.
    """
    T, n_ch = X_sample.shape
    spike_gens, sg_maps, sg_weights = build_temporal_input(
        X_sample, n_res, n_driven, rng, I_scale, t_offset)
    connect_temporal_input(spike_gens, sg_maps, sg_weights, L2)
    nest.SetStatus(rec_l2, {"n_events": 0})
    nest.Simulate(float(T) + 2.0)  # simulate full sequence + buffer
    n_spikes = nest.GetStatus(rec_l2, "n_events")[0]
    return n_spikes


def reset_membrane(L2):
    nest.SetStatus(L2, {"V_m": -70.0})


def teach(tg, lbl, t_now, T, n_cls, dt=1.0):
    """
    Spike-count based teacher — class k gets (k+1) teacher spikes.

    KEY IDEA (user-proposed):
      Class 0 → 1 spike  → L3[0] fires 1 time
      Class 1 → 2 spikes → L3[1] fires 2 times
      Class 2 → 3 spikes → L3[2] fires 3 times
      ...
      Class k → (k+1) spikes → L3[k] fires (k+1) times

    Why this works:
      More teacher spikes = more STDP updates on L2→L3[k]
      = stronger L2→L3[k] weights after training
      At test time (no teacher):
        L3[k] with strongest weights fires most → Argmax picks k

    Works for ANY sequence length T — no timing dependency.
    Spikes evenly distributed across window for each class.

    All non-target class teachers remain silent.
    """
    # Silence all teachers
    for k in range(n_cls):
        nest.SetStatus(tg[k:k+1], {"spike_times": []})

    dur        = T * dt
    n_spikes   = 3  # uniform for all classes

    # Distribute spikes evenly across the stimulus window
    # Use (n_spikes+1) intervals to avoid edge effects
    if n_spikes == 1:
        # Single spike at midpoint
        spike_times = [round((t_now + dur / 2.0) * 10) / 10]
    else:
        # Evenly spaced across window
        interval   = dur / (n_spikes + 1)
        spike_times = []
        for i in range(1, n_spikes + 1):
            t = round((t_now + i * interval) * 10) / 10
            spike_times.append(t)

    # Ensure all spikes within valid window
    spike_times = [t for t in spike_times
                   if t > t_now + 0.1 and t < t_now + dur - 0.1]

    # Fallback if window too short for all spikes
    if len(spike_times) == 0:
        spike_times = [round((t_now + dur / 2.0) * 10) / 10]

    # Ensure minimum spacing of 2ms between spikes (refractory)
    final_times = [spike_times[0]]
    for t in spike_times[1:]:
        if t - final_times[-1] >= 2.0:
            final_times.append(t)

    nest.SetStatus(tg[lbl:lbl+1],
                   {"spike_times": final_times,
                    "allow_offgrid_times": True})


def clear_teacher(tg, n_cls):
    for k in range(n_cls):
        nest.SetStatus(tg[k:k+1], {"spike_times": []})


def get_l2(rec_l2, L2_ids):
    ev = nest.GetStatus(rec_l2,"events")[0]
    return np.array([np.sum(ev["senders"]==nid)
                     for nid in L2_ids], dtype=np.float32)


def get_l3(rec_l3, L3_ids):
    ev = nest.GetStatus(rec_l3,"events")[0]
    return np.array([np.sum(ev["senders"]==nid)
                     for nid in L3_ids], dtype=np.float32)


def l3_summed(full, n_cls):
    return full.reshape(n_cls, NEURONS_PER_CLASS).sum(axis=1)


# ===========================================================================
# 5. FITNESS FUNCTION — 5-fold cross-validation (matching Paper 3)
# ===========================================================================
# Paper 3 (Steiner et al. 2023): 5-fold CV on training set, minimise MSE
# Our adaptation:  5-fold CV on training set, maximise macro F1
# Test set is NEVER seen during optimisation — only used for final report.

_eval_n = [0]


def _run_one_fold(w_init, d_mean, stdp, arch,
                  X_fold_tr, y_fold_tr,
                  X_fold_val, y_fold_val,
                  n_epochs, seed):
    """Train LSM on one fold, evaluate Ridge(L2) F1 on validation fold."""
    n_res    = arch["n_res"]
    n_cls    = arch["n_classes"]
    n_driven = arch["n_driven"]
    n_ch     = arch["n_features"]
    I_scale  = arch["I_scale"]
    T        = X_fold_tr.shape[1]
    N        = len(X_fold_tr)

    rng = np.random.default_rng(seed)
    nest.ResetKernel()
    nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time": False, "overwrite_files": True, "rng_seed": max(1, seed)})

    # Build network with exact paper architecture:
    # L1(input) → L2(reservoir) → L3(readout)
    # IH(inhibitory feedback), NZ(Poisson noise), Teacher(supervisory)
    p   = CaseParams("_", w_fixed=float(w_init), d_fixed=float(d_mean))
    net = _build(p, stdp, arch, "train", rng)

    L2, L3  = net["L2"], net["L3"]
    tg      = net["tg"]
    rec_l2  = net["rec_l2"]
    rec_l3  = net["rec_l3"]
    L2_ids  = net["L2_ids"]
    L3_ids  = net["L3_ids"]

    input_maps = [make_input_map(n_res, n_driven, rng)
                  for _ in range(n_ch)]
    Ftr   = np.zeros((N, n_res), dtype=np.float32)
    t_now = 0.0
    Ie_h  = np.zeros(n_cls * NEURONS_PER_CLASS, dtype=np.float32)

    # Training phase: teacher active, STDP enabled
    for ep in range(n_epochs):
        last = (ep == n_epochs - 1)
        idx  = np.random.default_rng(seed + ep).permutation(N)
        for si in idx:
            x, lbl = X_fold_tr[si], int(y_fold_tr[si])
            reset_membrane(L2)
            nest.SetStatus(rec_l2, {"n_events": 0})
            nest.SetStatus(rec_l3, {"n_events": 0})
            teach(tg, lbl, t_now, T, n_cls)

            # Input injection step-by-step (preserves temporal morphology)
            for t in range(T):
                inject_step(L2, x[t], input_maps, I_scale)
                nest.Simulate(float(arch.get("burst_ms", 1.0)))
                t_now += 1.0

            clear_input(L2, input_maps)
            clear_teacher(tg, n_cls)

            # Eval window: teacher disconnected, measure clean L2 state
            nest.SetStatus(rec_l2, {"n_events": 0})
            nest.SetStatus(rec_l3, {"n_events": 0})
            nest.Simulate(30.0)
            t_now += 30.0

            l2r = get_l2(rec_l2, L2_ids)
            l3f = get_l3(rec_l3, L3_ids)

            # Homeostatic adaptation on L3
            for k in range(n_cls * NEURONS_PER_CLASS):
                Ie_h[k] += (-2.0 if l3f[k] > 10
                             else (1.0 if l3f[k] == 0 else 0.0))
                Ie_h[k]  = float(np.clip(Ie_h[k], -100.0, 30.0))
                nest.SetStatus(L3[k:k+1], {"I_e": Ie_h[k]})

            if last:
                Ftr[si] = l2r

    # Save trained weights
    w22 = np.array([d["weight"] for d in nest.GetStatus(net["c22"])])

    # Inference phase: teacher disconnected, weights frozen
    rng2 = np.random.default_rng(seed + 100)
    nest.ResetKernel()
    nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time": False, "overwrite_files": True, "rng_seed": max(1, seed+100)})

    p2   = CaseParams("_", w_fixed=float(w_init), d_fixed=float(d_mean))
    net2 = _build(p2, stdp, arch, "inference", rng2, w_l2l2=w22)
    L2_2, rec_l2_2 = net2["L2"], net2["rec_l2"]
    L2_ids2 = net2["L2_ids"]
    imap2   = [make_input_map(n_res, n_driven, rng2) for _ in range(n_ch)]

    Nval  = len(X_fold_val)
    Fval  = np.zeros((Nval, n_res), dtype=np.float32)
    t2    = 0.0

    for i in range(Nval):
        reset_membrane(L2_2)
        nest.SetStatus(rec_l2_2, {"n_events": 0})
        for t in range(T):
            inject_step(L2_2, X_fold_val[i][t], imap2, I_scale)
            nest.Simulate(float(arch.get("burst_ms", 1.0)))
            t2 += 1.0
        clear_input(L2_2, imap2)
        nest.SetStatus(rec_l2_2, {"n_events": 0})
        nest.Simulate(30.0)
        t2 += 30.0
        Fval[i] = get_l2(rec_l2_2, L2_ids2)

    if Ftr.max() < 0.1 or Fval.max() < 0.1:
        return 0.0

    sc  = StandardScaler()
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(sc.fit_transform(Ftr), y_fold_tr)
    y_pred = clf.predict(sc.transform(Fval))
    return f1_score(y_fold_val, y_pred, average="macro", zero_division=0)


def evaluate(w_init, d_mean, stdp, arch,
             Xtr, ytr,
             n_epochs=3, seed=42,
             n_folds=5) -> float:
    """
    5-fold cross-validation fitness (matching Paper 3 methodology).

    Following Steiner et al. [TNNLS 2023]:
      - Training set partitioned into n_folds folds
      - Train on (n_folds-1) folds, validate on held-out fold
      - Fitness = mean CV F1 across all folds
      - Test set NEVER passed here — used only for final report

    Args:
        w_init:   initial/mean recurrent weight (pA)
        d_mean:   mean propagation delay (ms)
        stdp:     STDP parameter dict
        arch:     network architecture dict
        Xtr/ytr:  training data ONLY — test set not used in CV
        n_folds:  number of CV folds (default 5, matching Paper 3)
    """
    global _eval_n
    _eval_n[0] += 1

    try:
        kf       = KFold(n_splits=n_folds, shuffle=True,
                         random_state=seed)
        fold_f1s = []

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(Xtr)):
            X_fold_tr  = Xtr[tr_idx]
            y_fold_tr  = ytr[tr_idx]
            X_fold_val = Xtr[val_idx]
            y_fold_val = ytr[val_idx]

            fold_seed = seed + fold_idx * 1000
            f1 = _run_one_fold(
                w_init, d_mean, stdp, arch,
                X_fold_tr, y_fold_tr,
                X_fold_val, y_fold_val,
                n_epochs, fold_seed
            )
            fold_f1s.append(f1)
            gc.collect()

        mean_f1 = float(np.mean(fold_f1s))
        print(f"  eval#{_eval_n[0]:04d}: CV_F1={mean_f1:.4f} "
              f"folds={[f'{f:.3f}' for f in fold_f1s]} "
              f"w={w_init:.1f} d={d_mean:.2f}")
        return mean_f1

    except Exception as e:
        import traceback
        print(f"  [WARN] eval#{_eval_n[0]} failed: {e}")
        traceback.print_exc()
        return 0.0

# ===========================================================================
# 6. ES OPTIMISER
# ===========================================================================



def run_ga(fitness_fn, bounds, popsize=10, n_gen=20,
           mutation_rate=0.2, crossover_rate=0.5,
           seed=0, label=""):
    """Genetic Algorithm — top 50% selection, uniform crossover+mutation."""
    import random as _random
    _random.seed(seed)
    rng    = np.random.default_rng(seed)
    lo     = np.array([b[0] for b in bounds])
    hi     = np.array([b[1] for b in bounds])
    norm   = lambda x: (x - lo) / (hi - lo + 1e-12)
    denorm = lambda x: lo + x * (hi - lo)
    n_p    = len(bounds)

    def crossover(p1, p2):
        mask = rng.random(n_p) < crossover_rate
        return np.where(mask, p1, p2)

    def mutate(ind):
        new = ind.copy()
        for i in range(n_p):
            if rng.random() < mutation_rate:
                new[i] = rng.random()
        return np.clip(new, 0.0, 1.0)

    population = [rng.random(n_p) for _ in range(popsize)]
    scores     = [fitness_fn(denorm(ind)) for ind in population]
    best_idx   = int(np.argmax(scores))
    best_x     = population[best_idx].copy()
    best_f     = scores[best_idx]
    hist       = [best_f]

    print(f"\n  [GA] {label} | gen={n_gen} pop={popsize}")
    for gen in range(1, n_gen + 1):
        ranked    = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        n_keep    = max(2, popsize // 2)
        survivors = [ind for _, ind in ranked[:n_keep]]
        surv_sc   = [s for s, _ in ranked[:n_keep]]
        children  = []
        while len(children) < popsize - n_keep:
            p1, p2 = _random.sample(survivors, 2)
            children.append(mutate(crossover(p1, p2)))
        child_sc   = [fitness_fn(denorm(c)) for c in children]
        population = survivors + children
        scores     = surv_sc + child_sc
        gb         = int(np.argmax(scores))
        if scores[gb] > best_f:
            best_f, best_x = scores[gb], population[gb].copy()
        hist.append(best_f)
        print(f"  [GA] Gen {gen:3d}/{n_gen} best F1={best_f:.4f}")
    return denorm(best_x), best_f, hist


def run_es(fitness_fn, bounds, x0,
           popsize=8, n_gen=15, seed=0, label="", sigma_init=0.3):
    rng    = np.random.default_rng(seed)
    lo     = np.array([b[0] for b in bounds])
    hi     = np.array([b[1] for b in bounds])
    norm   = lambda x: (x-lo)/(hi-lo+1e-12)
    denorm = lambda x: lo+x*(hi-lo)
    x_n    = norm(np.clip(np.array(x0),lo,hi))
    sigma  = np.full(len(x_n), sigma_init)
    best_x = x_n.copy()
    best_f = fitness_fn(denorm(x_n))
    hist   = [best_f]
    print(f"\n  [ES] {label} | gen={n_gen} pop={popsize}")
    for gen in range(1, n_gen+1):
        fits = []
        for _ in range(popsize):
            xm = np.clip(x_n+sigma*rng.standard_normal(len(x_n)),0,1)
            f  = fitness_fn(denorm(xm))
            fits.append((f,xm))
            if f>best_f: best_f,best_x = f,xm.copy()
        nb = sum(f>best_f-1e-6 for f,_ in fits)
        sigma *= 1.22 if nb/popsize>0.2 else 0.82
        sigma  = np.clip(sigma,0.01,0.5)
        x_n    = best_x.copy()
        hist.append(best_f)
        print(f"  [ES] Gen {gen:3d}/{n_gen} best F1={best_f:.4f} "
              f"σ={sigma.mean():.3f}")
    return denorm(best_x), best_f, hist


# ===========================================================================
# 7. FULL TRAINING
# ===========================================================================

def train_case(params, stdp, arch, Xtr, ytr,
               n_epochs=10, seed=1, out_dir=""):
    n_res = arch["n_res"]
    n_cls = arch["n_classes"]
    n_ch  = arch["n_features"]
    npc   = NEURONS_PER_CLASS
    T     = Xtr.shape[1]
    N     = len(Xtr)

    rng = np.random.default_rng(seed)
    nest.ResetKernel(); nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time":False,"rng_seed":max(1,seed)})

    net    = _build(params, stdp, arch, "train", rng)
    L2,L3  = net["L2"],net["L3"]
    tg     = net["tg"]
    rec_l2,rec_l3 = net["rec_l2"],net["rec_l3"]
    L2_ids,L3_ids = net["L2_ids"],net["L3_ids"]

    input_maps = [make_input_map(n_res, arch["n_driven"], rng)
                  for _ in range(n_ch)]
    Ie_h  = np.zeros(n_cls*npc, dtype=np.float32)
    Ftr_l2= np.zeros((N, n_res),      dtype=np.float32)
    Ftr_l3= np.zeros((N, n_cls*npc),  dtype=np.float32)
    t_now = 0.0
    accs  = []

    print(f"\n{'='*60}")
    print(f"  Training Case {params.label} | {N}×{n_epochs} epochs "
          f"| {params.summary()}")
    t0 = time.time()

    for ep in range(1, n_epochs+1):
        last    = (ep==n_epochs)
        correct = 0
        idx     = np.random.default_rng(seed+ep).permutation(N)

        for pos, si in enumerate(idx):
            x, lbl = Xtr[si], int(ytr[si])
            reset_membrane(L2)
            reset_membrane(L3)  # reset V_m; homeostatic I_e retained
            nest.SetStatus(rec_l2,{"n_events":0})
            nest.SetStatus(rec_l3,{"n_events":0})
            teach(tg, lbl, t_now, T, n_cls)

            for t in range(T):
                inject_step(L2, x[t], input_maps,
                            arch["I_scale"])
                nest.Simulate(float(arch.get("burst_ms", 1.0))); t_now += float(arch.get("burst_ms", 1.0))

            clear_input(L2, input_maps)
            clear_teacher(tg, n_cls)

            # Eval window
            nest.SetStatus(rec_l2,{"n_events":0})
            nest.SetStatus(rec_l3,{"n_events":0})
            nest.Simulate(30.0); t_now += 30.0

            l2r = get_l2(rec_l2, L2_ids)
            l3f = get_l3(rec_l3, L3_ids)
            pred = int(np.argmax(l3_summed(l3f, n_cls)))
            if pred==lbl: correct+=1

            for k in range(n_cls*npc):
                Ie_h[k]+=-2.0 if l3f[k]>10 else(1.0 if l3f[k]==0 else 0.0)
                Ie_h[k]=float(np.clip(Ie_h[k],-100.0,30.0))
                nest.SetStatus(L3[k:k+1],{"I_e":Ie_h[k]})

            if last:
                Ftr_l2[si]=l2r; Ftr_l3[si]=l3f

            if (pos+1)%50==0 or (pos+1)==N:
                print(f"  Ep {ep}/{n_epochs} | {pos+1}/{N} | "
                      f"acc={correct/(pos+1)*100:.1f}%")

        acc = correct/N*100; accs.append(acc)
        w23 = np.array([d["weight"]
                        for d in nest.GetStatus(net["c23"])])
        print(f"  Ep {ep} done — acc={acc:.1f}% "
              f"L2L3 mean={w23.mean():.1f} "
              f"std={w23.std():.1f}")

    w22=np.array([d["weight"] for d in nest.GetStatus(net["c22"])])
    w23=np.array([d["weight"] for d in nest.GetStatus(net["c23"])])
    d22=np.array([d["delay"]  for d in nest.GetStatus(net["c22"])])
    d23=np.array([d["delay"]  for d in nest.GetStatus(net["c23"])])
    np.save(os.path.join(out_dir,"Ie_h.npy"),   Ie_h)
    np.save(os.path.join(out_dir,"w_l2l2.npy"), w22)
    np.save(os.path.join(out_dir,"w_l2l3.npy"), w23)
    np.save(os.path.join(out_dir,"d_l2l2.npy"), d22)
    np.save(os.path.join(out_dir,"d_l2l3.npy"), d23)

    # Save delays for histogram plots
    d22 = np.array([d["delay"] for d in nest.GetStatus(net["c22"])])
    d23 = np.array([d["delay"] for d in nest.GetStatus(net["c23"])])
    np.save(os.path.join(out_dir,"d_l2l2.npy"), d22)
    np.save(os.path.join(out_dir,"d_l2l3.npy"), d23)
    np.save(os.path.join(out_dir,"Ftr_l2.npy"), Ftr_l2)
    np.save(os.path.join(out_dir,"Ftr_l3.npy"), Ftr_l3)
    np.save(os.path.join(out_dir,"train_labels.npy"), ytr)
    np.save(os.path.join(out_dir,"epoch_accs.npy"), np.array(accs))
    np.save(os.path.join(out_dir,"input_maps.npy"), np.array(input_maps))
    print(f"  [Case {params.label}] {(time.time()-t0)/60:.1f} min")
    gc.collect()
    return Ftr_l2, Ftr_l3


# ===========================================================================
# 8. INFERENCE
# ===========================================================================

def infer_case(params, stdp, arch, Xte, out_dir, seed=10):
    n_res = arch["n_res"]
    n_cls = arch["n_classes"]
    n_ch  = arch["n_features"]
    T     = Xte.shape[1]

    rng = np.random.default_rng(seed)
    nest.ResetKernel(); nest.set_verbosity("M_QUIET")
    nest.SetKernelStatus({"print_time":False,"rng_seed":max(1,seed)})

    net = _build(params, stdp, arch, "inference", rng,
                 w_l2l2=np.load(os.path.join(out_dir,"w_l2l2.npy")),
                 w_l2l3=np.load(os.path.join(out_dir,"w_l2l3.npy")))
    # Load and apply trained delays after build
    d22_path = os.path.join(out_dir, "d_l2l2.npy")
    d23_path = os.path.join(out_dir, "d_l2l3.npy")
    if os.path.exists(d22_path):
        d22 = np.load(d22_path)
        c22 = net["c22"]
        res = nest.GetKernelStatus("resolution")
        for c, d in zip(c22, d22):
            c.set({"delay": float(np.clip(d, res, None))})
    if os.path.exists(d23_path):
        d23 = np.load(d23_path)
        c23 = net["c23"]
        res = nest.GetKernelStatus("resolution")
        for c, d in zip(c23, d23):
            c.set({"delay": float(np.clip(d, res, None))})
    L2     = net["L2"]
    rec_l2,rec_l3 = net["rec_l2"],net["rec_l3"]
    L2_ids,L3_ids = net["L2_ids"],net["L3_ids"]
    imap   = list(np.load(os.path.join(out_dir,"input_maps.npy")))

    Nte    = len(Xte)
    Fte_l2 = np.zeros((Nte, n_res),      dtype=np.float32)
    Fte_l3 = np.zeros((Nte, n_cls*NEURONS_PER_CLASS), dtype=np.float32)
    t_now  = 0.0

    print(f"  Inference Case {params.label} | {Nte} samples")
    for i in range(Nte):
        reset_membrane(L2)
        nest.SetStatus(rec_l2,{"n_events":0})
        nest.SetStatus(rec_l3,{"n_events":0})
        for t in range(T):
            inject_step(L2, Xte[i][t], imap, arch["I_scale"])
            nest.Simulate(float(arch.get("burst_ms", 1.0))); t_now+=float(arch.get("burst_ms", 1.0))
        clear_input(L2, imap)
        nest.SetStatus(rec_l2,{"n_events":0})
        nest.SetStatus(rec_l3,{"n_events":0})
        nest.Simulate(30.0); t_now+=30.0
        Fte_l2[i]=get_l2(rec_l2,L2_ids)
        Fte_l3[i]=get_l3(rec_l3,L3_ids)
        if (i+1)%20==0 or (i+1)==Nte:
            print(f"    {i+1}/{Nte}")

    np.save(os.path.join(out_dir,"Fte_l2.npy"), Fte_l2)
    np.save(os.path.join(out_dir,"Fte_l3.npy"), Fte_l3)
    gc.collect()
    return Fte_l2, Fte_l3


# ===========================================================================
# 9. CLASSIFICATION
# ===========================================================================

def classify(Ftr_l2, Ftr_l3, ytr,
             Fte_l2, Fte_l3, yte,
             label, n_cls, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(n_cls)]
    npc = NEURONS_PER_CLASS

    def _score(y_true, y_pred, name):
        n_seq   = len(y_true)
        n_error = int(np.sum(np.array(y_true)!=np.array(y_pred)))
        acc     = accuracy_score(y_true, y_pred)
        # Per-class F1 (needed for per_class_f1 plot)
        f1_per  = f1_score(y_true, y_pred, average=None,
                           labels=list(range(n_cls)),
                           zero_division=0).tolist()
        return {
            "decoder":       name,
            "y_pred":        y_pred,
            "cm":            confusion_matrix(y_true, y_pred,
                                              labels=list(range(n_cls))),
            "accuracy":      acc,
            "cer":           n_error/n_seq,
            "n_error":       n_error,
            "n_seq":         n_seq,
            "f1":            f1_score(y_true, y_pred,
                                      average="macro", zero_division=0),
            "f1_per_class":  f1_per,       # list of F1 per class
            "precision":     precision_score(y_true, y_pred,
                                             average="macro", zero_division=0),
            "recall":        recall_score(y_true, y_pred,
                                          average="macro", zero_division=0),
        }

    # Argmax(L3)
    Fte_sum = Fte_l3.reshape(-1,n_cls,npc).sum(axis=2)
    Ftr_sum = Ftr_l3.reshape(-1,n_cls,npc).sum(axis=2)
    res_ax  = _score(yte, np.argmax(Fte_sum,axis=1), "Argmax(L3)")

    # Ridge(L3)
    sc1     = StandardScaler()
    clf_l3  = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf_l3.fit(sc1.fit_transform(Ftr_sum), ytr)
    res_rl3 = _score(yte, clf_l3.predict(sc1.transform(Fte_sum)), "Ridge(L3)")

    # Ridge(L2) — primary
    sc2     = StandardScaler()
    clf_l2  = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf_l2.fit(sc2.fit_transform(Ftr_l2), ytr)
    res_rl2 = _score(yte, clf_l2.predict(sc2.transform(Fte_l2)),
                     "Ridge(L2) [primary]")

    print(f"\n{'='*60}")
    print(f"  Case {label} — {CASE_NAMES[label]}")
    print(f"{'='*60}")
    for res in [res_ax, res_rl3, res_rl2]:
        print(f"\n  [{res['decoder']}]")
        print(f"  Acc={res['accuracy']:.4f}  CER={res['cer']:.4f}  "
              f"({res['n_error']}/{res['n_seq']})  F1={res['f1']:.4f}")
        print(classification_report(yte, res["y_pred"],
                                    target_names=class_names[:n_cls],
                                    labels=list(range(n_cls)),
                                    zero_division=0))
    return {"case":label,"y_true":yte,
            "argmax":res_ax,"ridge_l3":res_rl3,"ridge_l2":res_rl2}


# ===========================================================================
# 10. PLOTS
# ===========================================================================

def _savefig(fig, name, out_dir):
    for ext in ("svg","pdf"):
        fig.savefig(os.path.join(out_dir,f"{name}.{ext}"),format=ext)
    print(f"  Saved → {name}.svg/.pdf")



def plot_confusion_matrices(all_m, all_p, n_cls, class_names, out_dir):
    """
    Confusion matrices for all 3 decoders × 4 cases.
    Saved as: cm_argmax.pdf/.svg, cm_ridge_l3.pdf/.svg, cm_ridge_l2.pdf/.svg
    """
    for decoder_key, decoder_title in [
        ("argmax",   "Argmax(L3) — Pure Spiking Readout"),
        ("ridge_l3", "Ridge(L3) — Linear on L3 Population"),
        ("ridge_l2", "Ridge(L2) — Linear on Reservoir States [Primary]"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Confusion Matrices — {decoder_title}",
                     fontsize=13, fontweight="bold")
        for ax, m in zip(axes.flat, all_m):
            res = m[decoder_key]
            cm  = res["cm"]
            im  = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ticks = list(range(n_cls))
            ax.set_xticks(ticks)
            ax.set_xticklabels(class_names[:n_cls], rotation=35, ha="right",
                               fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels(class_names[:n_cls], fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            p = next(q for q in all_p if q.label == m["case"])
            ax.set_title(f"Case {m['case']}: {CASE_NAMES[m['case']]}\n"
                         f"{p.summary()}\n"
                         f"Acc={res['accuracy']:.3f}  "
                         f"CER={res['cer']:.3f}  "
                         f"F1={res['f1']:.3f}", fontsize=9)
            th = cm.max() / 2
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > th else "black",
                            fontsize=9)
        plt.tight_layout()
        _savefig(fig, f"cm_{decoder_key}", out_dir)
        plt.close()
    print("  [Plot] Confusion matrices (3 decoders × 4 cases)")


def plot_weight_histograms(all_p, out_dir):
    """
    L2→L2 weight distributions after training.
    Key plot: shows Cases A/B have spike at fixed weight,
              Cases C/D have broad distribution.
    Times New Roman, saved as SVG+PDF.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    fig.suptitle("Recurrent Synaptic Weight Distributions After STDP Training\n"
                 "(L2→L2 connections)", fontsize=13, fontweight="bold")

    for ax, p in zip(axes.flat, all_p):
        wpath = os.path.join(out_dir, f"case_{p.label}", "w_l2l2.npy")
        if not os.path.exists(wpath):
            ax.text(0.5, 0.5, "Weights not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        weights = np.load(wpath)
        weights = weights[weights > 0.1]  # exclude unconnected synapses

        ax.hist(weights, bins=40, color=_COL[p.label],
                alpha=0.82, edgecolor="white", lw=0.4, zorder=3)
        ax.axvline(weights.mean(), color="black", ls="--", lw=1.5,
                   label=f"Mean={weights.mean():.1f}pA")
        ax.set_xlabel("Synaptic Weight (pA)")
        ax.set_ylabel("Count")
        ax.set_title(f"Case {p.label}: {CASE_NAMES[p.label]}\n"
                     f"{p.summary()}\n"
                     f"mean={weights.mean():.1f}  "
                     f"std={weights.std():.1f}  "
                     f"max={weights.max():.1f}pA", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

    plt.tight_layout()
    _savefig(fig, "weight_histograms_l2l2", out_dir)
    plt.close()

    # Also L2→L3 weights
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    fig.suptitle("Readout Synaptic Weight Distributions After STDP Training\n"
                 "(L2→L3 connections)", fontsize=13, fontweight="bold")

    for ax, p in zip(axes.flat, all_p):
        wpath = os.path.join(out_dir, f"case_{p.label}", "w_l2l3.npy")
        if not os.path.exists(wpath):
            ax.text(0.5, 0.5, "Weights not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        weights = np.load(wpath)
        weights = weights[weights > 0.1]

        ax.hist(weights, bins=40, color=_COL[p.label],
                alpha=0.82, edgecolor="white", lw=0.4, zorder=3)
        ax.axvline(weights.mean(), color="black", ls="--", lw=1.5,
                   label=f"Mean={weights.mean():.1f}pA")
        ax.set_xlabel("Synaptic Weight (pA)")
        ax.set_ylabel("Count")
        ax.set_title(f"Case {p.label}: {CASE_NAMES[p.label]}\n"
                     f"{p.summary()}\n"
                     f"mean={weights.mean():.1f}  "
                     f"std={weights.std():.1f}  "
                     f"max={weights.max():.1f}pA", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

    plt.tight_layout()
    _savefig(fig, "weight_histograms_l2l3", out_dir)
    plt.close()
    print("  [Plot] Weight histograms (L2→L2 and L2→L3)")


def plot_delay_histograms(all_p, out_dir):
    """
    L2→L2 and L2→L3 delay distributions after training.
    Key plot: Cases A/C show spike at fixed delay d*
              Cases B/D show broad distribution U[d_low, d_high]
    Directly shows the effect of delay heterogeneity on reservoir dynamics.
    Times New Roman, SVG + PDF.
    """
    for conn, label_str in [("l2l2", "L2→L2 Recurrent"),
                             ("l2l3", "L2→L3 Readout")]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
        fig.suptitle(f"Propagation Delay Distributions — {label_str}\n"
                     f"(Cases A/C: fixed optimised delay  |  "
                     f"Cases B/D: random optimised distribution)",
                     fontsize=12, fontweight="bold")

        for ax, p in zip(axes.flat, all_p):
            dpath = os.path.join(out_dir, f"case_{p.label}",
                                 f"d_{conn}.npy")
            if not os.path.exists(dpath):
                ax.text(0.5, 0.5, "Delays not saved\n(rerun needed)",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f"Case {p.label}: {CASE_NAMES[p.label]}",
                             fontsize=9)
                continue

            delays = np.load(dpath)

            # Fixed delay cases (A, C): spike at single value
            # Random delay cases (B, D): broad distribution
            is_fixed = (p.d_fixed is not None)
            bins = 5 if is_fixed else 40

            ax.hist(delays, bins=bins, color=_COL[p.label],
                    alpha=0.82, edgecolor="white", lw=0.4, zorder=3)
            ax.axvline(delays.mean(), color="black", ls="--", lw=1.5,
                       label=f"Mean={delays.mean():.2f}ms")

            # For random cases, show the optimised bounds
            if not is_fixed and p.d_low is not None:
                ax.axvline(p.d_low,  color="red",  ls=":", lw=1.2,
                           label=f"d_low={p.d_low:.1f}ms")
                ax.axvline(p.d_high, color="blue", ls=":", lw=1.2,
                           label=f"d_high={p.d_high:.1f}ms")

            ax.set_xlabel("Propagation Delay (ms)")
            ax.set_ylabel("Count")
            ax.set_title(f"Case {p.label}: {CASE_NAMES[p.label]}\n"
                         f"{p.summary()}\n"
                         f"mean={delays.mean():.2f}  "
                         f"std={delays.std():.2f}  "
                         f"range=[{delays.min():.2f}, "
                         f"{delays.max():.2f}]ms",
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

        plt.tight_layout()
        _savefig(fig, f"delay_histograms_{conn}", out_dir)
        plt.close()

    print("  [Plot] Delay histograms (L2→L2 and L2→L3)")


def plot_per_class_f1(all_m, all_p, n_cls, class_names, out_dir):
    """
    Per-class F1 breakdown for all 4 cases.
    Shows which classes each configuration handles well.
    One subplot per case, bars per class, 3 decoders overlaid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle("Per-Class F1 Score — All Cases and Decoders",
                 fontsize=13, fontweight="bold")

    x = np.arange(n_cls)
    w = 0.25

    for ax, m in zip(axes.flat, all_m):
        p = next(q for q in all_p if q.label == m["case"])

        for i, (key, lbl, alpha) in enumerate([
            ("argmax",   "Argmax(L3)",  0.45),
            ("ridge_l3", "Ridge(L3)",   0.70),
            ("ridge_l2", "Ridge(L2) ★", 1.00),
        ]):
            f1_per = m[key]["f1_per_class"]
            bars = ax.bar(x + (i-1)*w, f1_per, w,
                          label=lbl, color=_COL[m["case"]],
                          alpha=alpha, zorder=3,
                          edgecolor="black" if alpha == 1.0 else "none",
                          lw=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names[:n_cls], rotation=35,
                           ha="right", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Case {m['case']}: {CASE_NAMES[m['case']]}\n"
                     f"{p.summary()}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
        ax.axhline(1/n_cls, ls="--", color="gray", lw=1,
                   alpha=0.5, label="Random")

    plt.tight_layout()
    _savefig(fig, "per_class_f1", out_dir)
    plt.close()
    print("  [Plot] Per-class F1 breakdown")


def plot_es_convergence(stdp_hist, wd_histories, out_dir):
    """
    ES convergence curves:
    Left:  STDP parameter optimisation (Phase 1a)
    Right: Weight/delay optimisation per case (Phase 1b)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ES Optimisation Convergence",
                 fontsize=13, fontweight="bold")

    # Phase 1a — STDP
    ax = axes[0]
    if stdp_hist:
        ax.plot(stdp_hist, "b-o", lw=2, markersize=5,
                label=f"Best F1={max(stdp_hist):.4f}")
        ax.fill_between(range(len(stdp_hist)), stdp_hist,
                        alpha=0.12, color="blue")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1a — STDP Parameter Optimisation\n"
                 "(Aplus, Aminus, tau, Wmax, triplet params)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    # Phase 1b — per case
    ax = axes[1]
    for lbl, hist in wd_histories.items():
        ax.plot(hist, color=_COL[lbl], marker=_MRK[lbl],
                lw=2, markersize=5,
                label=f"Case {lbl} (best={max(hist):.3f})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best 5-fold CV F1")
    ax.set_title("Phase 1b — Weight/Delay Optimisation\n"
                 "(per-case: w_fixed/range, d_fixed/range)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    _savefig(fig, "es_convergence", out_dir)
    plt.close()
    print("  [Plot] ES convergence curves")


def plot_l2_activity(all_m, all_p, out_dir):
    """
    L2 reservoir spike rate comparison across cases.
    Shows mean spike rate per reservoir neuron for each case.
    Reveals how weight/delay configuration affects reservoir dynamics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Reservoir (L2) Activity — Train vs Test",
                 fontsize=13, fontweight="bold")

    for ax, split in zip(axes, ["train", "test"]):
        for m in all_m:
            p    = next(q for q in all_p if q.label == m["case"])
            cd   = os.path.join(out_dir, f"case_{m['case']}")
            fname = f"Ftr_l2.npy" if split == "train" else f"Fte_l2.npy"
            fpath = os.path.join(cd, fname)
            if not os.path.exists(fpath):
                continue
            F    = np.load(fpath)
            # Mean spike rate per neuron across all samples
            mean_rates = F.mean(axis=0)
            ax.plot(sorted(mean_rates, reverse=True),
                    color=_COL[m["case"]], lw=1.5,
                    label=f"Case {m['case']}: {p.summary()}")

        ax.set_xlabel("Reservoir Neuron (sorted by activity)")
        ax.set_ylabel("Mean Spike Count")
        ax.set_title(f"L2 Activity Distribution ({split.capitalize()})")
        ax.legend(fontsize=7)
        ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    _savefig(fig, "l2_activity", out_dir)
    plt.close()
    print("  [Plot] L2 reservoir activity")


def plot_all(all_m, all_p, n_cls, out_dir):
    # Metrics bar chart
    x = np.arange(len(all_m)); w = 0.25
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(f"Results — All Decoders × All Cases",
                 fontweight="bold")
    for ax, metric in zip(axes, ["cer","f1"]):
        lmap = {"cer":"CER ↓ (lower better)",
                "f1": "Macro F1 ↑ (higher better)"}
        ax_v  = [m["argmax"][metric]   for m in all_m]
        rl3_v = [m["ridge_l3"][metric] for m in all_m]
        rl2_v = [m["ridge_l2"][metric] for m in all_m]
        cols  = [_COL[m["case"]]       for m in all_m]
        for i,(vals,lbl,alpha) in enumerate([
                (ax_v, "Argmax(L3)",  0.45),
                (rl3_v,"Ridge(L3)",   0.70),
                (rl2_v,"Ridge(L2) ★", 1.00)]):
            bars = ax.bar(x+(i-1)*w, vals, w, label=lbl,
                          color=cols, alpha=alpha, zorder=3,
                          edgecolor="black" if alpha==1.0 else "none",
                          lw=0.5)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.01,
                        f"{v:.3f}",ha="center",fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Case {m['case']}" for m in all_m])
        ax.set_ylim(0,1.18)
        ax.set_ylabel(lmap[metric])
        ax.legend(fontsize=8)
        ax.grid(True,axis="y",ls=":",alpha=0.4,zorder=0)
    plt.tight_layout()
    _savefig(fig,"metrics_3decoders",out_dir)
    plt.close()

    # Epoch accuracy
    fig,ax = plt.subplots(figsize=(9,4))
    for p in all_p:
        f = os.path.join(out_dir,f"case_{p.label}","epoch_accs.npy")
        if os.path.exists(f):
            accs = np.load(f)
            ax.plot(range(1,len(accs)+1),accs,
                    color=_COL[p.label],marker=_MRK[p.label],
                    lw=2,markersize=6,
                    label=f"Case {p.label}: {p.summary()}")
    ax.axhline(100/n_cls,ls="--",color="gray",lw=1,alpha=0.6,
               label=f"Random ({100/n_cls:.0f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Accuracy (%)")
    ax.set_title("STDP Training Convergence")
    ax.set_ylim(0,105); ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig,"epoch_accuracy",out_dir)
    plt.close()

    # Summary table
    fig,ax = plt.subplots(figsize=(16,4)); ax.axis("off")
    cols = ["Case","Parameters",
            "Argmax CER↓","Argmax F1↑",
            "Ridge(L3) CER↓","Ridge(L3) F1↑",
            "Ridge(L2) CER↓★","Ridge(L2) F1↑★"]
    rows = [[f"Case {m['case']}", next(p for p in all_p
                                       if p.label==m["case"]).summary(),
             f"{m['argmax']['cer']:.4f}",
             f"{m['argmax']['f1']:.4f}",
             f"{m['ridge_l3']['cer']:.4f}",
             f"{m['ridge_l3']['f1']:.4f}",
             f"{m['ridge_l2']['cer']:.4f}",
             f"{m['ridge_l2']['f1']:.4f}"]
            for m in all_m]
    tbl = ax.table(cellText=rows,colLabels=cols,
                   loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2,2.2)
    for j in range(len(cols)):
        tbl[(0,j)].set_facecolor("#1e3a5f")
        tbl[(0,j)].set_text_props(color="white",fontweight="bold")
    for i,m in enumerate(all_m):
        for j in range(len(cols)):
            tbl[(i+1,j)].set_facecolor(_COL[m["case"]]+"22")
    plt.tight_layout()
    _savefig(fig,"summary_table",out_dir)
    plt.close()

    # Additional priority 1 plots
    class_names = [str(i) for i in range(n_cls)]
    plot_confusion_matrices(all_m, all_p, n_cls, class_names, out_dir)
    plot_weight_histograms(all_p, out_dir)
    plot_delay_histograms(all_p, out_dir)
    plot_per_class_f1(all_m, all_p, n_cls, class_names, out_dir)
    plot_l2_activity(all_m, all_p, out_dir)


# ===========================================================================
# 11. MAIN
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="ROBOT",
                    help="Dataset name (ROBOT, JPVOW, WAF, LIB, ...)")
    ap.add_argument("--data_dir", default=(
        "Time-series-classification-and-clustering-"
        "with-Reservoir-Computing/datasets"),
                    help="Path to FilippoMB datasets folder")
    ap.add_argument("--n_res",    type=int, default=200)
    ap.add_argument("--n_driven", type=int, default=80)
    ap.add_argument("--I_scale",  type=float, default=8000.0)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--es_gen_stdp", type=int, default=15)
    ap.add_argument("--es_gen_wd",   type=int, default=10)
    ap.add_argument("--es_pop",      type=int, default=8)
    ap.add_argument("--skip_es", action="store_true",
                    help="Skip ES, use default params")
    args = ap.parse_args()

    DATASET    = args.dataset
    OUT_DIR    = f"results_{DATASET}"
    ES_DIR     = f"Results_{DATASET}_ES"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ES_DIR,  exist_ok=True)

    t_total = time.time()
    print(f"\n{'='*60}")
    print(f"  LSM — {DATASET}")
    print(f"  n_res={args.n_res} n_driven={args.n_driven} "
          f"I_scale={args.I_scale} epochs={args.n_epochs}")
    print(f"{'='*60}")

    # Load data
    print(f"\n--- Loading {DATASET} ---")
    Xtr,ytr,Xte,yte,n_cls = load_dataset(DATASET, args.data_dir)
    n_ch = Xtr.shape[2]

    arch = {
        "n_res":     args.n_res,
        "n_classes": n_cls,
        "n_features":n_ch,
        "indeg22":   min(15, args.n_res-1),
        "indeg23":   30,
        "n_driven":  args.n_driven,
        "I_scale":   args.I_scale,
    }

    # ES Optimisation
    if not args.skip_es:
        print(f"\n{'='*60}")
        print(f"  ES OPTIMISATION — {DATASET}")
        print(f"{'='*60}")
        _eval_n[0] = 0

        STDP_BOUNDS = [(1e-5,0.05),(1e-5,0.05),(5.0,50.0),
                       (1e-4,0.02),(1e-4,0.02),(20.0,300.0),(100.0,600.0)]
        STDP_X0     = [0.001,0.001,20.0,0.005,0.005,100.0,300.0]

        def stdp_fit(x):
            # CV fitness — test set NOT passed (data leakage prevention)
            return evaluate(150.0, 1.0, x_to_stdp(x), arch,
                           Xtr, ytr, n_epochs=3)

        best_stdp_arr, best_stdp_f1, stdp_hist = run_es(
            stdp_fit, STDP_BOUNDS, STDP_X0,
            popsize=args.es_pop, n_gen=args.es_gen_stdp,
            seed=0, label="STDP")
        best_stdp = x_to_stdp(best_stdp_arr)

        with open(os.path.join(ES_DIR,"stdp_params.json"),"w") as f:
            json.dump({"stdp":best_stdp,"f1":best_stdp_f1},f,indent=2)
        print(f"  STDP F1={best_stdp_f1:.4f}")
        # Save STDP convergence history for later plotting
        stdp_convergence_hist = list(stdp_hist)

        # Per-case w/d
        wd_results = {}
        for clbl, cfg in CASE_CONFIGS.items():
            def wd_fit(x, cfg=cfg):
                w=float(np.clip(cfg["w_fn"](x),10.0,
                                best_stdp["Wmax"]*0.95))
                d=float(np.clip(cfg["d_fn"](x),0.1,25.0))
                # CV fitness — test set NOT passed (data leakage prevention)
                return evaluate(w,d,best_stdp,arch,
                               Xtr,ytr,n_epochs=3)
            arr,f1,hist = run_es(wd_fit,cfg["bounds"],cfg["x0"],
                                  popsize=args.es_pop,
                                  n_gen=args.es_gen_wd,
                                  seed=ord(clbl),
                                  label=f"Case {clbl}")
            opt_p = cfg["save"](arr)
            wd_results[clbl] = {"params":opt_p,"f1":f1,"hist":list(hist)}
            cd = os.path.join(ES_DIR,f"case_{clbl}")
            os.makedirs(cd,exist_ok=True)
            with open(os.path.join(cd,"opt_params.json"),"w") as f:
                json.dump(opt_p,f,indent=2)
            print(f"  Case {clbl}: F1={f1:.4f} {opt_p}")
    else:
        print("  Skipping ES — using default params")
        best_stdp = x_to_stdp(
            [0.001,0.001,20.0,0.005,0.005,100.0,300.0])
        wd_results = {
            "A": {"params":{"w_fixed":150.0,"d_fixed":1.0}},
            "B": {"params":{"w_fixed":150.0,"d_low":1.0,"d_high":10.0}},
            "C": {"params":{"w_low":100.0,"w_high":300.0,"d_fixed":1.0}},
            "D": {"params":{"w_low":100.0,"w_high":300.0,
                            "d_low":1.0,"d_high":10.0}},
        }

    # ES convergence plot
    if not args.skip_es:
        wd_hist_dict = {clbl: list(wd_results[clbl].get("hist", []))
                        for clbl in CASE_CONFIGS}
        stdp_h = locals().get("stdp_convergence_hist", [])
        plot_es_convergence(stdp_h, wd_hist_dict,
                           os.path.join(OUT_DIR))

    # Load params as CaseParams
    all_params = []
    for clbl, cfg in CASE_CONFIGS.items():
        if args.skip_es:
            p = cfg["load"](wd_results[clbl]["params"])
        else:
            p = cfg["load"](wd_results[clbl]["params"])
        p.label = clbl
        all_params.append(p)
        print(f"  Case {clbl}: {p.summary()}")

    # Full training
    print(f"\n{'='*60}")
    print(f"  TRAINING — {DATASET}")
    print(f"{'='*60}")
    tr_l2,tr_l3 = {},{}
    for p in all_params:
        cd = os.path.join(OUT_DIR,f"case_{p.label}")
        os.makedirs(cd,exist_ok=True)
        fl2,fl3 = train_case(p, best_stdp, arch, Xtr, ytr,
                             n_epochs=args.n_epochs, seed=10,
                             out_dir=cd)
        tr_l2[p.label],tr_l3[p.label] = fl2,fl3

    # Inference + classification
    print(f"\n{'='*60}")
    print(f"  INFERENCE — {DATASET}")
    print(f"{'='*60}")
    all_metrics = []
    for p in all_params:
        cd   = os.path.join(OUT_DIR,f"case_{p.label}")
        fl2,fl3 = infer_case(p, best_stdp, arch, Xte, cd, seed=10)
        print(f"\n  Diagnostics Case {p.label}:")
        print(f"    L2 train: mean={tr_l2[p.label].mean():.2f} "
              f"max={tr_l2[p.label].max():.0f}")
        print(f"    L2 test:  mean={fl2.mean():.2f} "
              f"max={fl2.max():.0f}")
        print(f"    L3 test:  mean={fl3.mean():.2f} "
              f"max={fl3.max():.0f}")
        m = classify(tr_l2[p.label],tr_l3[p.label],ytr,
                     fl2,fl3,yte,p.label,n_cls)
        all_metrics.append(m)

    # Plots
    plot_all(all_metrics, all_params, n_cls, OUT_DIR)

    # Final summary
    total = time.time()-t_total
    print(f"\n{'='*60}")
    print(f"  FINAL — {DATASET}  ({total/3600:.2f} hr)")
    print(f"{'='*60}")
    print(f"\n  {'Case':<6} {'Params':<32} "
          f"{'Argmax':>10} {'':>8} {'R(L2)★':>9} {'':>8}")
    print(f"  {'':6} {'':32} "
          f"{'CER↓':>10} {'F1↑':>8} {'CER↓':>9} {'F1↑':>8}")
    print(f"  {'-'*75}")
    for m in all_metrics:
        p = next(q for q in all_params if q.label==m["case"])
        print(f"  {m['case']:<6} {p.summary():<32} "
              f"{m['argmax']['cer']:>10.4f} "
              f"{m['argmax']['f1']:>8.4f} "
              f"{m['ridge_l2']['cer']:>9.4f} "
              f"{m['ridge_l2']['f1']:>8.4f}")

    best = max(all_metrics, key=lambda m: m["argmax"]["f1"])
    bp   = next(p for p in all_params if p.label==best["case"])
    print(f"\n  Best Argmax(L3): Case {best['case']} "
          f"| {bp.summary()}")
    print(f"  F1={best['argmax']['f1']:.4f}  "
          f"CER={best['argmax']['cer']:.4f}")
    print(f"\n  Results: {OUT_DIR}/")

    # Save JSON
    out = {m["case"]: {
        "params": next(p for p in all_params
                       if p.label==m["case"]).summary(),
        "argmax_f1":    m["argmax"]["f1"],
        "argmax_cer":   m["argmax"]["cer"],
        "ridge_l2_f1":  m["ridge_l2"]["f1"],
        "ridge_l2_cer": m["ridge_l2"]["cer"],
    } for m in all_metrics}
    with open(os.path.join(OUT_DIR,"results.json"),"w") as f:
        json.dump(out,f,indent=2)
