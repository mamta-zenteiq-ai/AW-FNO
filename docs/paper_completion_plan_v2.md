# Paper Completion Plan v2 — Strong Multi-Dataset Submission

Supersedes [paper_completion_plan.md](paper_completion_plan.md) (the original
ICCFD13-tight roadmap). Created 2026-06-03 after three decisions:

1. **Rich gate validated.** The 2-layer Conv (k=5) gate produces *decisive,
   spatially-routed* α (Burgers Phase 2.5: ρ(1−α, |∇u|) up to +0.32, beats
   FNO ~10%). The 1×1 gate collapses to α≈0.5 (H≈ln2) — reproduced again on
   the SR run now in flight. The rich gate is the paper's headline mechanism.
2. **Scope = one strong paper, expanded with more datasets** (user, 2026-06-03):
   strength prioritised over the June-12 ICCFD13 date. Add PDEBench NS Riemann,
   PDEBench Burgers ν=1e-4, and a turbulence benchmark.
3. **Turbulence dataset = JHTDB isotropic4096, already on disk** at
   `/media/HDD/anjali/gazania_transolver/jhtdb_datasets` — no access friction.

The spine (SR + Burgers + NS regime story) stays the submittable core; new
datasets are layered on to widen the gate-routing evidence across regimes.

---

## Where we are (2026-06-03)

### Done / in flight
| Dataset | Models | Status | Headline number |
|---|---|---|---|
| NS 64² time-stepping (Phase 1) | 6-way ablation | ✅ done | rel_l2 1.4e-4; fixed α wins on homogeneous data |
| PDEBench Burgers ν=1e-3 (Phase 2/2.5) | FNO/WNO/AW-FNO 1×1/rich | ✅ done | rich gate 3.3e-2, ρ up to +0.32 |
| **NS-forcing 128² SR (Phase A — title task)** | 6-run queue | 🔄 **4/6** | FNO 0.0193 / WNO 0.0179 / fixed-α 0.0210 |

**SR queue ([scripts/run_sr_queue.sh](../scripts/run_sr_queue.sh)) live on GPU 0
— DO NOT relaunch.** Order: FNO✅ · WNO✅ · fixed-α✅ · **1×1 gate🔄(ep~140, H≈0.679 collapsed)** · fat-FNO⏳ · **rich-gate SR⏳ (headline)**. ~9h/AW-FNO run → rich-gate SR lands ~2026-06-04 AM.

### Paper state ([paper.tex](../paper.tex))
- Experiments + Results sections written. **5 open TODOs:** table rows for
  FNO-fat (L353) and AW-FNO rich gate (L357); headline paragraph (L378);
  E(k) figure (L393); gate-routing ρ figure + discussion (L408).
- **Does NOT compile:** missing `biblio.bib`, `figures/arch.pdf`, banner.

### Reusable infrastructure (don't rebuild)
- Entry point: `python experiments/train.py --config configs/experiment/<x>.yaml --data_path … --output_dir …`
- Queue pattern: `run_one` helper in [scripts/run_sr_queue.sh](../scripts/run_sr_queue.sh)
- Aggregation: [scripts/aggregate_{sr,burgers,phase1}_results.py](../scripts/)
- Gate routing: [scripts/analyze_gate_{sr,burgers}.py](../scripts/)
- Energy spectrum: [scripts/energy_spectrum_sr.py](../scripts/)
- Rich-gate patch (`_patch_rich_gate`, 1D Conv1d + 2D Conv2d): [experiments/train.py](../experiments/train.py)
- Loaders: [awfno/data/](../awfno/data/) — `pdebench_burgers.py`, **`pdebench_compns.py` (compressible NS — Riemann/Sod, already written)**, `nsforcing_sr.py`, `ns2d.py`, `burgers1d.py`
- Burgers ν=1e-3 HDF5 on disk: `/media/HDD/mamta_backup/datasets/PDEBench/burgers/1D_Burgers_Sols_Nu0.001.hdf5`

**GPU reality:** single GPU, ~9h per 128² AW-FNO run, 1D Burgers ~1h. Runs are
serialised. CPU/writing work (loaders, configs, paper, figures from finished
runs) parallelises against the GPU queue.

---

## Per-new-dataset model set (cost control)

The full 6-way ablation is already established on SR/Burgers/NS. New datasets
run a **focused 4-run set** (reuse the ablation narrative, don't repeat it):

1. FNO (standard) · 2. WNO · 3. **AW-FNO rich gate (k=5, 2L)** — the win ·
4. AW-FNO fixed α=0.5 — the regime control.
Add fat-FNO only where a reviewer-fairness question is live (high-param WNO).

---

## STAGE 0 — Finish the in-flight SR headline (no new GPU)
*Gated on the running queue; ~24h wall, ~2h hands-on after it completes.*

0.1 Let queue finish 1×1 gate → fat-FNO → rich-gate SR. **Do not touch GPU 0.**
0.2 `python scripts/aggregate_sr_results.py --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes` → fills `outputs/tables/sr_main_table.{csv,tex}`. Paste the 2 missing rows into [paper.tex](../paper.tex) L353, L357.
0.3 `python scripts/analyze_gate_sr.py` on `awfno_nsforcing_sr_richgate/best.pt` → gate-map fig + ρ(1−α,|∇ω|). Fill L408 + Fig 4.
0.4 `python scripts/energy_spectrum_sr.py` → E(k) fig (Fig 3, L393).
0.5 Write the headline paragraph at L378 from the measured rich-gate SR row.

**Deliverable:** SR section complete (Table 1, Figs 2–4). The paper is
*self-contained and submittable* at the end of Stage 0 + Stage 1.

---

## STAGE 1 — Make the paper build (parallel, CPU/writing, ~half day)

1.1 Create `biblio.bib`: PDEBench (Takamoto 2022), DiffFNO (Liu 2025), MoE
    (Jacobs 1991, Shazeer 2017), Fukami 2019 (turbulence SR), AFNO (Guibas
    2022), UNO (Rahman 2022), WNO (Tripura 2023), FNO (Li 2021), JHTDB
    (Li 2008 / Perlman 2007) for the isotropic data.
1.2 Produce `figures/arch.pdf` (architecture diagram) + resolve the banner include.
1.3 `pdflatex` clean compile end-to-end. **This unblocks every later figure/table.**

---

## STAGE 2 — DROPPED (Burgers viscosity sweep)
**Resolved 2026-06-03 (user):** PDEBench has no Burgers ν=1e-4 (min is ν=1e-3,
already trained). The planned sweep is **dropped** — the 1D CFD shock-tube
(Stage 3) is the genuine-shock case ("Burgers with real shocks") and is a
stronger shock story than a Burgers viscosity sweep. The Burgers ν=0.001 result
(done) stays as the canonical viscous-shock point; the shock-tube adds the
genuine multi-wave shock case. The ν=0.01 configs remain parked but unused.

2.1 Download `1D_Burgers_Sols_Nu0.0001.hdf5` (DARUS doi:10.18419/darus-2986)
    into `…/PDEBench/burgers/`.
2.2 Copy `configs/dataset/pdebench_burgers.yaml` → `pdebench_burgers_nu1e-4.yaml`,
    set `viscosity_tag: "0.0001"`.
2.3 4 experiment configs (clone the existing `*_pdebench_burgers*` set). Reuse
    `train_awfno_pdebench_burgers_richgate.yaml`.
2.4 Queue (1D = fast). Aggregate with `aggregate_burgers_results.py`; gate
    analysis with `analyze_gate_burgers.py`.

**Deliverable:** ν-sweep showing routing strengthens as the shock sharpens —
ablation depth + a second Burgers column in the cross-regime table.

---

## STAGE 3 — PDEBench compressible NS Riemann (strongest gate target)
*~2–3 days; loader exists, multi-wave analysis is the new work.*

Why: shock + contact discontinuity + rarefaction fan in a single sample —
the richest spatial routing target. `pdebench_compns.py` loader already exists
(used for Sod); extend/validate it for the Riemann HDF5.

3.1 Download PDEBench 1D (then 2D if budget) compressible NS HDF5
    (doi:10.18419/darus-2986); confirm field layout `[N,T,X,(Y),V]` (ρ,u,p…).
3.2 Validate `awfno/data/pdebench_compns.py` against the Riemann file; add
    `configs/dataset/pdebench_riemann.yaml`. Gate signal = |∇ρ|.
3.3 Model/experiment configs (4-run set). 1D first (fast), 2D if budget allows.
3.4 **New analysis:** `scripts/analyze_gate_riemann.py` — segment the domain
    into shock / contact / rarefaction bands and report per-wave
    ρ(1−α, |∇ρ|). The multi-wave extension of the gate story.

**Deliverable:** multi-wave routing figure + per-wave correlation table
(DATASET 1, ★★★★★).

---

## STAGE 4 — JHTDB isotropic turbulence SR (turbulence benchmark, data ready)
*~3 days; data on disk. New: loader + filament gate analysis.*

**Data:** `/media/HDD/anjali/gazania_transolver/jhtdb_datasets` — 23 cutout
chunks `velocity_cutout_x{a}_{b}.npy`, each `(z=1024, y=1024, x=4, V=3)`
float32 (3-component velocity), dataset attr `isotropic4096`. → ~92 planes of
1024² 3-component velocity. Use `.npy` (fast; `.nc` is the same data + coords).

**Why it strengthens the paper:** high-Reλ forced isotropic turbulence is
genuinely *intermittent* (velocity-gradient flatness ≫ 3) — vortex filaments
are localised 1D structures in a smooth background, the textbook adaptive-
routing target. It is **2D-sliceable**, so it fits the paper's SR framing
without 3D cost, and its clean k^{−5/3} inertial range makes the E(k) figure
especially compelling.

**NOTE the gate signal is filament/enstrophy-based, NOT wall-distance** — this
is the survey's DATASET 5 (isotropic), not DATASET 4 (channel). Paper framing
must say "high-enstrophy / high-strain vortex filaments → WNO; smooth bulk →
FNO."

4.1 `awfno/data/jhtdb_iso.py` loader: read `.npy` cutouts → 2D (z,y) planes
    per x-slice; **crop to 256² patches** (WNO filter banks scale with
    resolution — 1024² is infeasible, 256² keeps WNO ~ the 128² regime).
    SR task: HR = 256² patch, LR = 4× bicubic-downsampled input. 3 velocity
    channels. Train/test split *across chunks* (e.g. x1–x72 train, x73–x92
    test) to avoid leakage.
4.2 `configs/dataset/jhtdb_iso_sr.yaml` + model configs reusing `awfno_ns128`,
    `fno_ns128`, `wno_ns128` (256² ≈ same param regime as 128²; bump n_modes).
4.3 Queue the 4-run set on the GPU after Stages 2–3 (each ~several h).
4.4 **New analysis:** `scripts/analyze_gate_jhtdb.py` — compute per-plane
    enstrophy / strain-rate magnitude (in-plane gradients of all 3 velocity
    components), correlate (1−α) with high-enstrophy filament regions; overlay
    gate map on enstrophy. Extend `energy_spectrum_sr.py` to plot E(k) vs the
    k^{−5/3} reference.

**Deliverable:** turbulence-SR table column + filament gate-map figure +
isotropic E(k) figure — the paper's "beyond canonical shocks" breadth.

**Optional/stretch turbulence (only if runway remains):** CFDBench Kármán
(geometric gate, downloadable) or compressible RT LES (3D, author-contact
data, removed 3D model path would need restoration). Not required given JHTDB.

---

## STAGE 5 — Cross-dataset synthesis & polish (parallel + final week)

5.1 **Regime-characterization section (C6):** the paper's strongest insight —
    adaptive routing helps on spatially heterogeneous data (Burgers/Riemann
    shocks, JHTDB filaments); fixed mixing suffices on statistically
    homogeneous data (NS 64² turbulence). Now backed by 5–6 datasets.
5.2 Unified cross-regime table: per-dataset {FNO, WNO, AW-FNO rich, fixed-α}
    rel_l2 + ρ(1−α, gate signal) + gate entropy. One table tells the story.
5.3 Cheap reviewer-strengthening ablations: kernel sweep k∈{3,5,7,9} on Burgers
    (B3); FLOPs/params table via `fvcore` (B5); per-channel α analysis (B6).
5.4 Final figures (`scripts/generate_paper_figures.py`), internal review, proof.

---

## Decision still open (for the user)

**Venue/deadline:** the full multi-dataset set will not all land by 2026-06-12.
Options: (a) submit the SR+Burgers spine to ICCFD13 by Jun 12, extra datasets
in camera-ready / a journal version; (b) retarget a later venue for the full
strong version. Strength was chosen over the date — confirm which path.

## Immediate actions that need no decision (start now, GPU-free)
- Stage 1 (biblio.bib + arch.pdf + compile) — unblocks all figures.
- Stage 2.1 + 3.1 downloads (Burgers ν=1e-4, Riemann HDF5).
- Stage 4.1 JHTDB loader + a quick shape/normalisation sanity check (data is local).
- **Do NOT relaunch the SR queue** — it is running on GPU 0.
