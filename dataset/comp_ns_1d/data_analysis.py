"""
PDEBench 1D Compressible CFD Dataset Analyser
==============================================
Dataset : 1D_CFD_Sod1.hdf5  (Sod shock-tube, compressible Navier-Stokes)
Source  : https://darus.uni-stuttgart.de

Usage
-----
    python analyse_pdebench_1d_cfd.py

Outputs
-------
  • Console report  – structure, shapes, statistics, NaN/Inf audit
  • plots/           – all figures saved as high-res PNGs

What this covers (in order)
----------------------------
  1.  HDF5 tree inspection
  2.  Shape / dtype summary
  3.  Descriptive statistics per field
  4.  NaN / Inf / near-zero-std audit (DL-readiness check)
  5.  Field snapshots at t=0, t_mid, t_end  (per trajectory)
  6.  Space-time heatmaps (density / velocity / pressure)
  7.  Time-evolution of spatial mean & std per field
  8.  Distribution (histogram + KDE) per field
  9.  Field-to-field correlation heatmap
  10. Multi-trajectory spread (variability across ICs)
  11. Power spectral density per field (spatial frequency content)
  12. Normalisation recommendation for DL training
"""

import os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – safe on any machine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  –  change FILE_PATH if needed
# ─────────────────────────────────────────────
FILE_PATH   = "/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d/1D_CFD_Sod5.hdf5"
PLOT_DIR    = "plots"
N_TRAJ_SHOW = 5        # trajectories to overlay in spread plots
SEED        = 42

os.makedirs(PLOT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────
# COLOUR PALETTE  (colourblind-friendly)
# ─────────────────────────────────────────────
FIELD_COLORS = {"Vx": "#2196F3", "rho": "#FF5722", "pressure": "#4CAF50"}
CMAP_SPACE_TIME = "RdBu_r"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def banner(title: str):
    w = 70
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)

def section(title: str):
    print(f"\n── {title} {'─'*(65-len(title))}")

def save(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

def fmt_shape(arr):
    return " × ".join(str(d) for d in arr.shape)

def describe(arr, name=""):
    flat = arr.flatten()
    return {
        "field"  : name,
        "shape"  : arr.shape,
        "dtype"  : arr.dtype,
        "min"    : float(np.nanmin(flat)),
        "max"    : float(np.nanmax(flat)),
        "mean"   : float(np.nanmean(flat)),
        "std"    : float(np.nanstd(flat)),
        "median" : float(np.nanmedian(flat)),
        "nan"    : int(np.sum(np.isnan(flat))),
        "inf"    : int(np.sum(np.isinf(flat))),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 1. OPEN FILE & TREE INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════

banner("PDEBench 1D CFD – Dataset Analysis")
print(f"\nFile : {FILE_PATH}")

if not os.path.exists(FILE_PATH):
    sys.exit(f"\n[ERROR] File not found:\n  {FILE_PATH}\n"
             "  Update FILE_PATH at the top of this script.")

fsize_mb = os.path.getsize(FILE_PATH) / 1e6
print(f"Size : {fsize_mb:.1f} MB")

section("HDF5 Tree")
def print_tree(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}[DS]  {name}  |  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"{indent}[GRP] {name}/")

with h5py.File(FILE_PATH, "r") as f:
    f.visititems(print_tree)
    top_keys = list(f.keys())
    print(f"\nTop-level keys: {top_keys}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATA  –  handle both common PDEBench layouts
# ═══════════════════════════════════════════════════════════════════════════════

section("Loading Arrays")

with h5py.File(FILE_PATH, "r") as f:

    # PDEBench compressible NS 1D stores data under keys: Vx, rho, pressure  (or p)
    # Coordinates: t (scalar or 1-D), x (1-D)
    # Shape convention: (n_samples, n_t, n_x)

    def load_key(f, candidates):
        for k in candidates:
            if k in f:
                return k, f[k][:]
        return None, None

    t_key, t_data = load_key(f, ["t", "time", "T"])
    x_key, x_data = load_key(f, ["x", "X", "coords"])

    vx_key,  Vx   = load_key(f, ["Vx", "vx", "u", "velocity"])
    rho_key, rho  = load_key(f, ["rho", "density", "Rho"])
    p_key,   pres = load_key(f, ["pressure", "p", "P", "press"])

    # Fallback: list all remaining datasets
    all_ds = {}
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_ds[name] = obj[:]
    f.visititems(collect)

# Build fields dict with whatever we found
fields: dict[str, np.ndarray] = {}
field_labels = {}

for key, arr in [("Vx", Vx), ("rho", rho), ("pressure", pres)]:
    if arr is not None:
        fields[key] = arr

if not fields:
    print("[WARN] Standard keys not found – using all datasets as fields.")
    fields = {k: v for k, v in all_ds.items()
              if k not in ("t", "x", "time", "X", "coords")}

print(f"Fields loaded   : {list(fields.keys())}")

# Infer dimensions  (samples, timesteps, spatial_pts)
first = next(iter(fields.values()))
if first.ndim == 3:
    n_samples, n_t, n_x = first.shape
elif first.ndim == 2:
    n_samples, n_x = first.shape
    n_t = 1
    fields = {k: v[:, np.newaxis, :] for k, v in fields.items()}
else:
    sys.exit("[ERROR] Unexpected array dimensionality.")

print(f"Samples         : {n_samples}")
print(f"Timesteps       : {n_t}")
print(f"Spatial pts     : {n_x}")

# Build coordinate arrays
if t_data is not None:
    t = np.asarray(t_data).flatten()
    if len(t) == 1:
        t = np.linspace(0, float(t[0]), n_t)
    elif len(t) != n_t:
        t = np.linspace(0, 1, n_t)
else:
    t = np.linspace(0, 1, n_t)

if x_data is not None:
    x = np.asarray(x_data).flatten()
    if len(x) != n_x:
        x = np.linspace(0, 1, n_x)
else:
    x = np.linspace(0, 1, n_x)

dt_str = f"{t[1]-t[0]:.5f}" if len(t) > 1 else "N/A (single-step)"
dx_str = f"{x[1]-x[0]:.5f}" if len(x) > 1 else "N/A (single-point)"
print(f"t ∈ [{t[0]:.4f}, {t[-1]:.4f}]  (dt = {dt_str})")
print(f"x ∈ [{x[0]:.4f}, {x[-1]:.4f}]  (dx = {dx_str})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

banner("Descriptive Statistics")
stats_rows = []
for name, arr in fields.items():
    d = describe(arr, name)
    stats_rows.append(d)
    print(f"\n  Field : {name}  |  shape = {fmt_shape(arr)}")
    print(f"    min={d['min']:.6g}  max={d['max']:.6g}  "
          f"mean={d['mean']:.6g}  std={d['std']:.6g}  "
          f"median={d['median']:.6g}")
    print(f"    NaN={d['nan']}  Inf={d['inf']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DL-READINESS AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

banner("DL-Readiness Audit")
issues = []
for d in stats_rows:
    if d["nan"] > 0:
        issues.append(f"  ⚠  {d['field']}: {d['nan']} NaN values detected!")
    if d["inf"] > 0:
        issues.append(f"  ⚠  {d['field']}: {d['inf']} Inf values detected!")
    if d["std"] < 1e-8:
        issues.append(f"  ⚠  {d['field']}: near-zero std ({d['std']:.2e}) – constant field?")

if issues:
    print("\n".join(issues))
else:
    print("  ✓  No NaN / Inf / zero-std issues found.")

print("\nNormalisation recommendations (global, per field):")
for d in stats_rows:
    rng_val = d["max"] - d["min"]
    print(f"  {d['field']:12s}  min-max range = {rng_val:.4g}  "
          f"→  z-score: subtract {d['mean']:.4g}, divide by {d['std']:.4g}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIELD SNAPSHOTS AT t=0, t_mid, t_end
# ═══════════════════════════════════════════════════════════════════════════════

banner("Plotting…")
n_fields = len(fields)
t_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
t_labels  = ["t=0", "t=T/4", "t=T/2", "t=3T/4", "t=T"]
traj_idx  = 0   # first trajectory for snapshot plots

fig, axes = plt.subplots(n_fields, len(t_indices),
                          figsize=(4 * len(t_indices), 3 * n_fields),
                          squeeze=False)
fig.suptitle(f"Field Snapshots – Trajectory #{traj_idx}", fontsize=14, fontweight="bold")

for row, (fname, farr) in enumerate(fields.items()):
    color = FIELD_COLORS.get(fname, "#9C27B0")
    for col, (ti, tlbl) in enumerate(zip(t_indices, t_labels)):
        ax = axes[row][col]
        ax.plot(x, farr[traj_idx, ti, :], color=color, lw=1.8)
        ax.set_title(tlbl, fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        if col == 0:
            ax.set_ylabel(fname, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

plt.tight_layout()
save(fig, "01_field_snapshots.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPACE-TIME HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 5), squeeze=False)
fig.suptitle(f"Space-Time Heatmaps – Trajectory #{traj_idx}", fontsize=14, fontweight="bold")

for col, (fname, farr) in enumerate(fields.items()):
    ax = axes[0][col]
    data2d = farr[traj_idx]            # (n_t, n_x)
    vabs   = np.nanpercentile(np.abs(data2d), 98)
    vmin   = data2d.mean() - vabs
    vmax   = data2d.mean() + vabs
    im = ax.imshow(data2d, aspect="auto", origin="lower",
                   extent=[x[0], x[-1], t[0], t[-1]],
                   cmap=CMAP_SPACE_TIME, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("t", fontsize=10)
    ax.set_title(fname, fontsize=11, fontweight="bold")

plt.tight_layout()
save(fig, "02_spacetime_heatmaps.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TIME-EVOLUTION OF SPATIAL MEAN & STD
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(n_fields, 2, figsize=(12, 3.5 * n_fields), squeeze=False)
fig.suptitle("Spatial Mean & Std Over Time  (all trajectories)", fontsize=13, fontweight="bold")

for row, (fname, farr) in enumerate(fields.items()):
    color  = FIELD_COLORS.get(fname, "#9C27B0")
    sp_mean = farr.mean(axis=-1)    # (n_samples, n_t)
    sp_std  = farr.std(axis=-1)     # (n_samples, n_t)

    ax_m, ax_s = axes[row][0], axes[row][1]

    # Plot each trajectory lightly, then ensemble mean
    for s in range(min(n_samples, 30)):
        ax_m.plot(t, sp_mean[s], color=color, alpha=0.15, lw=0.8)
        ax_s.plot(t, sp_std[s],  color=color, alpha=0.15, lw=0.8)

    ax_m.plot(t, sp_mean.mean(0), color=color, lw=2.2, label="ensemble mean")
    ax_s.plot(t, sp_std.mean(0),  color=color, lw=2.2, label="ensemble mean")

    for ax, lbl in [(ax_m, "Spatial Mean"), (ax_s, "Spatial Std")]:
        ax.set_ylabel(f"{fname} – {lbl}", fontsize=9)
        ax.set_xlabel("t", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

plt.tight_layout()
save(fig, "03_time_evolution_mean_std.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. DISTRIBUTION HISTOGRAMS + KDE
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 4), squeeze=False)
fig.suptitle("Value Distributions (all samples, all times)", fontsize=13, fontweight="bold")

for col, (fname, farr) in enumerate(fields.items()):
    ax    = axes[0][col]
    color = FIELD_COLORS.get(fname, "#9C27B0")
    flat  = farr.flatten()

    # Use a subsample if huge
    if len(flat) > 500_000:
        flat = rng.choice(flat, 500_000, replace=False)

    ax.hist(flat, bins=120, density=True, color=color, alpha=0.45, edgecolor="none")

    # KDE
    kde_x = np.linspace(flat.min(), flat.max(), 500)
    kde   = stats.gaussian_kde(flat, bw_method="scott")
    ax.plot(kde_x, kde(kde_x), color=color, lw=2.2)

    ax.axvline(float(np.mean(flat)), color="black", ls="--", lw=1.2, label=f"mean={np.mean(flat):.3g}")
    ax.axvline(float(np.median(flat)), color="gray", ls=":", lw=1.2, label=f"median={np.median(flat):.3g}")
    ax.set_xlabel(fname, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(fname, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "04_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FIELD CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

if len(fields) > 1:
    fnames = list(fields.keys())
    n_f    = len(fnames)
    corr   = np.ones((n_f, n_f))

    # Sample up to 100k points for speed
    flat_arrays = []
    for fn in fnames:
        fl = fields[fn].flatten()
        flat_arrays.append(fl)

    n_pts = min(100_000, len(flat_arrays[0]))
    idx   = rng.choice(len(flat_arrays[0]), n_pts, replace=False)

    for i in range(n_f):
        for j in range(i + 1, n_f):
            r, _ = stats.pearsonr(flat_arrays[i][idx], flat_arrays[j][idx])
            corr[i, j] = corr[j, i] = r

    fig, ax = plt.subplots(figsize=(4 + n_f, 3.5 + n_f * 0.4))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n_f)); ax.set_xticklabels(fnames, fontsize=11)
    ax.set_yticks(range(n_f)); ax.set_yticklabels(fnames, fontsize=11)
    ax.set_title("Field Correlation Matrix", fontsize=13, fontweight="bold")

    for i in range(n_f):
        for j in range(n_f):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color="black" if abs(corr[i, j]) < 0.7 else "white")

    plt.tight_layout()
    save(fig, "05_field_correlations.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. MULTI-TRAJECTORY SPREAD (variability across ICs)
# ═══════════════════════════════════════════════════════════════════════════════

traj_show = min(N_TRAJ_SHOW, n_samples)
chosen    = rng.choice(n_samples, traj_show, replace=False)

ti_show   = n_t // 2   # snapshot at t=T/2
fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 4), squeeze=False)
fig.suptitle(f"Trajectory Spread at t=T/2  ({traj_show} random trajectories)",
             fontsize=13, fontweight="bold")

for col, (fname, farr) in enumerate(fields.items()):
    ax    = axes[0][col]
    cmap  = plt.cm.viridis
    for k, s in enumerate(chosen):
        clr = cmap(k / max(traj_show - 1, 1))
        ax.plot(x, farr[s, ti_show, :], color=clr, lw=1.5, alpha=0.85, label=f"traj {s}")

    # Ensemble mean ± std band
    mu  = farr[:, ti_show, :].mean(0)
    sig = farr[:, ti_show, :].std(0)
    ax.plot(x, mu, "k--", lw=2.0, label="ensemble mean")
    ax.fill_between(x, mu - sig, mu + sig, color="gray", alpha=0.2, label="±1σ")

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel(fname, fontsize=10)
    ax.set_title(fname, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "06_trajectory_spread.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. POWER SPECTRAL DENSITY (spatial)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 4), squeeze=False)
fig.suptitle("Spatial Power Spectral Density", fontsize=13, fontweight="bold")

dx = float(x[1] - x[0]) if len(x) > 1 else 1.0

for col, (fname, farr) in enumerate(fields.items()):
    ax    = axes[0][col]
    color = FIELD_COLORS.get(fname, "#9C27B0")

    # Average PSD over all samples and timesteps
    psds = []
    for s in range(n_samples):
        for ti in range(0, n_t, max(1, n_t // 8)):   # every 8th timestep
            signal = farr[s, ti, :]
            f_freq, pxx = welch(signal, fs=1.0 / dx, nperseg=min(256, n_x))
            psds.append(pxx)

    psd_arr = np.stack(psds)
    ax.semilogy(f_freq, psd_arr.mean(0), color=color, lw=2.0, label="mean PSD")
    ax.fill_between(f_freq,
                    psd_arr.mean(0) - psd_arr.std(0),
                    psd_arr.mean(0) + psd_arr.std(0),
                    color=color, alpha=0.2)

    ax.set_xlabel("Spatial frequency", fontsize=10)
    ax.set_ylabel("PSD", fontsize=10)
    ax.set_title(fname, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
save(fig, "07_spatial_psd.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. INPUT–OUTPUT PAIR PREVIEW (what the operator model will see)
# ═══════════════════════════════════════════════════════════════════════════════
# Convention: give first T_in timesteps as input → predict remaining T_out
# If n_t == 1 (snapshot-only dataset), show field profiles across samples instead.

T_IN  = max(1, n_t // 5)
T_OUT = n_t - T_IN
s_demo = 0

if n_t > 1 and T_OUT > 0:
    # ── Normal case: multi-timestep data ──────────────────────────────────────
    fig = plt.figure(figsize=(14, 3.5 * n_fields))
    gs  = gridspec.GridSpec(n_fields, 2, figure=fig, hspace=0.45, wspace=0.3)
    fig.suptitle(f"Operator Model I/O Split  (T_in={T_IN}, T_out={T_OUT}) – Traj #0",
                 fontsize=13, fontweight="bold")

    for row, (fname, farr) in enumerate(fields.items()):
        ax_in  = fig.add_subplot(gs[row, 0])
        ax_out = fig.add_subplot(gs[row, 1])

        t_in_end  = t[min(T_IN - 1, len(t) - 1)]
        t_out_start = t[min(T_IN, len(t) - 1)]
        t_out_end   = t[-1]

        im1 = ax_in.imshow(farr[s_demo, :T_IN, :], aspect="auto", origin="lower",
                           extent=[x[0], x[-1], t[0], t_in_end],
                           cmap=CMAP_SPACE_TIME)
        plt.colorbar(im1, ax=ax_in, fraction=0.04)
        ax_in.set_title(f"{fname}  [INPUT  t=0…{t_in_end:.3f}]", fontsize=9)
        ax_in.set_xlabel("x"); ax_in.set_ylabel("t")

        im2 = ax_out.imshow(farr[s_demo, T_IN:, :], aspect="auto", origin="lower",
                            extent=[x[0], x[-1], t_out_start, t_out_end],
                            cmap=CMAP_SPACE_TIME)
        plt.colorbar(im2, ax=ax_out, fraction=0.04)
        ax_out.set_title(f"{fname}  [TARGET t={t_out_start:.3f}…{t_out_end:.3f}]", fontsize=9)
        ax_out.set_xlabel("x"); ax_out.set_ylabel("t")

    save(fig, "08_operator_io_split.png")

else:
    # ── Single-timestep dataset: show IC → field profile across samples ───────
    print("  [NOTE] n_t=1 – dataset contains a single snapshot per sample.")
    print("         Plot 08 shows field profiles for multiple ICs instead.")

    n_show = min(10, n_samples)
    chosen_ic = rng.choice(n_samples, n_show, replace=False)
    cmap_ic   = plt.cm.tab10

    fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 4), squeeze=False)
    fig.suptitle(f"Single-Snapshot Dataset – Field Profiles for {n_show} ICs",
                 fontsize=13, fontweight="bold")

    for col, (fname, farr) in enumerate(fields.items()):
        ax = axes[0][col]
        for k, s in enumerate(chosen_ic):
            ax.plot(x, farr[s, 0, :], color=cmap_ic(k / n_show),
                    lw=1.5, alpha=0.85, label=f"IC {s}")
        mu  = farr[:, 0, :].mean(0)
        sig = farr[:, 0, :].std(0)
        ax.plot(x, mu, "k--", lw=2.0, label="ensemble mean")
        ax.fill_between(x, mu - sig, mu + sig, color="gray", alpha=0.2, label="±1σ")
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel(fname, fontsize=10)
        ax.set_title(fname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    save(fig, "08_ic_profiles_single_snapshot.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

banner("Summary Report")

print(f"""
  ┌─ Dataset ──────────────────────────────────────────────────────────┐
  │  File       : {os.path.basename(FILE_PATH)}
  │  Samples    : {n_samples}
  │  Timesteps  : {n_t}       (T_in recommended ≥ {max(1, n_t//5)})
  │  Spatial pts: {n_x}
  │  Fields     : {', '.join(fields.keys())}
  └────────────────────────────────────────────────────────────────────┘

  ┌─ Per-Field Stats ───────────────────────────────────────────────────┐""")

for d in stats_rows:
    print(f"  │  {d['field']:10s}  range=[{d['min']:.4g}, {d['max']:.4g}]"
          f"  μ={d['mean']:.4g}  σ={d['std']:.4g}")

print(f"""  └────────────────────────────────────────────────────────────────────┘

  ┌─ Recommended Normalisation for DL ─────────────────────────────────┐""")

for d in stats_rows:
    print(f"  │  {d['field']:10s}  z-score → (x - {d['mean']:.5g}) / {d['std']:.5g}")

print(f"""  └────────────────────────────────────────────────────────────────────┘

  Plots saved → {os.path.abspath(PLOT_DIR)}/
    01_field_snapshots.png
    02_spacetime_heatmaps.png
    03_time_evolution_mean_std.png
    04_distributions.png
    05_field_correlations.png
    06_trajectory_spread.png
    07_spatial_psd.png
    08_operator_io_split.png
""")