# Paper Completion Plan — ICCFD13 Submission

Concrete, prioritized roadmap to complete the Experiments section of
[paper.tex](../paper.tex) before the **2026-06-12** ICCFD13 deadline.

Created 2026-06-01, after Phase 2.5 (rich-gate Burgers) validated the
core gating claim.  Tracks experimental work, paper revisions, and
the day-by-day budget.  Pairs with [experimental_log.md](experimental_log.md),
which captures what has already been run; this file is the forward-
looking plan.

---

## Current asset inventory

### Experiments done

| Phase | Best rel_l2 | Notes |
|---|---|---|
| Phase 1 — NS 64² time-stepping (6 models) | 1.4e-4 | Full ablation table; gate-collapse story; fixed α=0.5 wins on homogeneous data |
| Phase 2 — PDEBench Burgers ν=1e-3 (5 models, 1×1 gate) | 4.5e-2 | Spatial routing **failed** with 1×1 gate (ρ ≈ 0.03) |
| Phase 2.5 — Burgers with rich gate (k=5, 2L) | **3.3e-2** | Beats FNO by 10 %; spatial routing **validated** (ρ up to +0.32) |
| Phase 2.6 — Sod loader + smoke test | — | Infrastructure ready; not yet trained at length |

### Paper claims vs evidence

| Paper claim (paper.tex) | Evidence status |
|---|---|
| C1 — SR task (LR→HR) | **NO EXPERIMENT YET** ⚠ |
| C2 — High-Reynolds turbulent benchmark | Phase 1 (Re=1000) |
| C3 — Synergistic FNO+WNO fusion | Phase 1 + Phase 2.5 |
| C4 — Enhanced generalisation | NOT TESTED |
| C7 — Localized refinement (eddies / BL) | NOT VISUALIZED |
| C8 — No single operator uniformly optimal | Phase 2.5 ✓ |
| C9 — Gate routes WNO at sharp, FNO at smooth | Phase 2.5 ✓ (with caveat: needs rich gate) |
| C10 — Gibbs reduction | Phase 1 high-freq metric ✓ |
| C11 — Entropy-regularised gating | Implemented; trained Phase 1–2.5 |
| C13 — Reduced computational overhead | NOT QUANTIFIED |

**The critical gap:** the paper's primary task is *super-resolution*, and
we haven't run a single SR experiment.  That must be Stage A.

---

## STAGE A — Essential experiments (days 1–3)

### A1. SR experiment on nsforcing 128² — the paper's actual task

**Why this is non-negotiable:** the paper title is
*"…for Fluid flow Super-resolution"*.  Without an SR experiment, there
is no paper.

**Models to train** (all with `hidden_channels=32`, `n_modes=20`,
`wno_level=3`, `db6`, on `nsforcing_train_128.pt`, 4× downsample with
bicubic LR input → HR 128²):

| Run | Time | Note |
|---|---|---|
| Bicubic (no-model baseline) | 0 min | Just upsampling, evaluate only |
| FNO | ~1.5 hr | hidden_channels=32 |
| FNO-fat (param-matched) | ~3 hr | hidden_channels=128 — answers reviewer fairness question |
| WNO | ~3 hr | Branch-only baseline |
| AW-FNO fixed α=0.5 | ~3 hr | Phase 1's winner; tests "homogeneous → uniform" theory on SR |
| AW-FNO 1×1 adaptive | ~3 hr | Phase 2's loser; sanity check |
| **AW-FNO rich gate (k=5, 2L)** | **~3 hr** | **Primary contribution — does Phase 2.5's win transfer to SR?** |

**Total:** ~17 hr GPU time, single GPU, sequential queue overnight.

**Configs already exist** (`configs/experiment/train_*_nsforcing.yaml`)
— just need to add ablation_no_gate, rich_gate, and fat-FNO variants.

**Deliverables:** main results table (Table 1) + comparison field
figures (Figure 2).

### A2. Gate-spatial-routing analysis on SR predictions

Mirror of `scripts/analyze_gate_burgers.py` but for 2D: extract α(x, y)
maps from rich-gate AW-FNO trained on SR, overlay on |∇ω| (vorticity
gradient magnitude).

**Expected outcome:**
- If ρ(1−α, |∇ω|) > 0.1: paper's "gate routes WNO at vortex filaments"
  claim holds for SR
- If ρ ≈ 0: claim is shocks-specific; need to discuss regime
  characterization

**Deliverable:** gate-map figure (Figure 4).

### A3. Energy spectrum E(k) figure for NS SR

Implement radial PSD computation; plot E(k) vs k for ground truth,
FNO, WNO, AW-FNO-rich, and bicubic.  Show:
- AW-FNO matches Kolmogorov k^{−5/3} inertial range better than FNO
- High-k spectral energy lower than FNO (Gibbs reduction)

**Deliverable:** spectrum figure (Figure 3).

### A4. Write Section 4 (Experiments) — datasets / baselines / metrics / implementation


Subsections:
- 4.1 Datasets (NS 64², nsforcing SR 128², PDEBench Burgers ν=1e-3)
- 4.2 Baselines and ablation variants (FNO, WNO, additive, fixed
  α=0.5, 1×1 gate, rich gate)
- 4.3 Metrics (rel L2, rel H1, high-freq spectral L2, enstrophy, gate
  entropy, ρ((1-α), |∇u|))
- 4.4 Implementation details (Adam, lr=1e-3, step scheduler,
  λ_ent=0.01, random init std=0.2, training time, hardware)

### A5. Write Section 5 (Results) — tables + figures

- 5.1 Main SR comparison (Table 1, Figure 2)
- 5.2 Energy spectrum analysis (Figure 3)
- 5.3 Gate-routing analysis (Figure 4 — gate maps + ρ correlation)
- 5.4 Cross-regime comparison (Table 2): Phase 1 NS vs Phase 2 Burgers
  vs Phase 2.5 Burgers rich-gate vs SR — illustrates the regime-
  characterization story
- 5.5 Ablation study (Table 3 — already have for NS and Burgers)

---

## STAGE B — Strengthening experiments (days 3–6, in parallel with writing)

These add depth, but the paper survives without them.

### B1. Rich gate on NS 64² (Phase 1 data)

Train rich-gate AW-FNO on homogeneous turbulence.  Predicts: gate
converges to α ≈ 0.5 (homogeneous → uniform optimal).  If yes →
confirms regime characterization story.

**Time:** ~1.5 hr.  **Impact:** validates regime story.

### B2. Sod proper training

Train rich-gate AW-FNO on density field of the 7 Sod files (50–200
epochs).  Run per-feature gate analysis (shock vs contact
discontinuity vs rarefaction).

**Time:** ~30 min (small dataset).  **Impact:** multi-wave validation
of gate routing.

### B3. Kernel sweep on Burgers

Train rich-gate AW-FNO with k=3, k=7, k=9 (in addition to existing
k=5).  Tests whether wider receptive field → stronger ρ.

**Time:** ~3 hr (3 runs × 1 hr).  **Impact:** ablation depth in
Results §5.5.

### B4. Zero-shot resolution test

Train AW-FNO @ 64² on NS, evaluate @ 128² and 256².  WNO is **not**
resolution-invariant in our implementation — will fail.  Either
(a) report this honestly as a limitation, or (b) extend WNO to
handle variable input size (~1 day of engineering).

**Time:** 0 hr (just eval); 1 day if extending WNO.  **Impact:**
zero-shot claim in paper holds OR is honestly walked back.

### B5. FLOPs / params comparison table

Use `fvcore.nn.FlopCountAnalysis` (already installed) to measure
forward FLOPs per sample for FNO, WNO, AW-FNO.  Add to
Implementation section.

**Time:** 30 min.  **Impact:** addresses computational-efficiency
claim (paper C13).

### B6. Per-channel α analysis

The current gate analysis averages α over channels.  Look at
per-channel α distributions: do different channels route differently?

**Time:** 30 min (analysis script only).  **Impact:** gate-mechanism
understanding for paper.

---

## STAGE C — Paper revisions (days 4–8, in parallel with experiments)

### C1. Update §3.3 (Adaptive Spatial Gate) to match implementation

The paper currently says (line 193):
> *"$1 \times 1$ convolution followed by a sigmoid activation"*

Phase 2 proved this is **insufficient**.  Replace with:
> *"A two-layer convolutional gate with kernel size $k=5$ and spatial
> padding:
> $\alpha = \sigma(\mathrm{Conv}_{k}(\mathrm{GELU}(\mathrm{Conv}_k([V_F, V_W]))))$.
> The kernel size is chosen so the gate's effective receptive field
> exceeds the characteristic width of sharp features in the data
> (~5–10 grid points for Burgers shocks at $\nu = 10^{-3}$)."*

### C2. Fix the master equation (Eq. 1, line 157-158)

Current:
```
v_{t+1}(x) = σ(W v_t(x) + K_F v_t(x) + K_W v_t(x) + b(x))
```

Must change to include α (or split into two equations):

```
v_{F,t+1}(x) = K_F v_t(x)
v_{W,t+1}(x) = K_W v_t(x)
α(x) = σ(Conv_k(GELU(Conv_k([v_F, v_W]))))
v_{t+1}(x) = σ(W v_t(x) + α(x) ⊙ v_F + (1−α(x)) ⊙ v_W + b(x))
```

### C3. Add §3.4 — Entropy regularisation

Re-enable the commented-out entropy regularisation paragraph
(paper.tex lines 174-178).  Update with the actual loss:

> $L_{\text{total}} = L_{\text{data}} + \lambda_{\text{ent}} \cdot \mathbb{E}_x[H(\alpha(x))]$
> with $\lambda_{\text{ent}} = 0.01$.

### C4. Update Conclusions

Current claim:
> *"The model learn to prioritise FNO for smooth, large-scale coherent
> structures and WNO for sharp gradients..."*

This is only true with the rich gate.  Add the caveat:

> *"This routing behaviour requires the gate to have sufficient
> representational capacity; a single $1 \times 1$ convolution cannot
> express the spatial routing pattern."*

### C5. Bibliography additions

Add: PDEBench (Takamoto 2022), DiffFNO (Liu 2025 — already cited),
Mixture-of-Experts foundations (Jacobs 1991, Shazeer 2017),
Fukami 2019 (turbulence SR), AFNO (Guibas 2022), UNO (Rahman 2022).

### C6. Add Section 5.4 — Regime characterization

The paper's strongest novel insight from our experiments:
*adaptive routing helps when data is spatially heterogeneous (Burgers
shocks); fixed mixing suffices when data is statistically homogeneous
(NS turbulence at modest Re).*  This deserves its own short subsection
— it is both a positive finding and reviewer-friendly framing.

---

## Time budget — June 1 to June 12

| Days | Work |
|---|---|
| **Day 1** (Jun 1) | Set up SR experiment configs (fat-FNO, rich-gate); launch SR queue overnight |
| **Day 2** | SR queue running.  Write §4 (Experiments) + §5.5 (Ablation).  Start B1 (NS rich-gate) on idle GPU time |
| **Day 3** | SR queue done.  Analyse SR gate maps (A2).  Generate energy spectrum (A3).  Write §5.1, §5.2 |
| **Day 4** | Run B2 (Sod), B3 (kernel sweep), B5 (FLOPs).  Write §5.3, §5.4.  C1-C2 paper revisions |
| **Day 5** | C3-C5 paper revisions.  Generate all final figures + tables |
| **Day 6** | Buffer / re-runs of failed experiments |
| **Day 7-8** | C6 (regime characterization).  Polish writing |
| **Day 9-10** | Internal review |
| **Day 11** | Final proofreading |
| **Day 12** | Submit |

---

## What we deliberately skip

Nice to have, but **not for ICCFD13 with this deadline**:

- 3D experiments (out of scope, computationally expensive)
- JHTDB (data access friction, not needed for the SR / Burgers story)
- CFDBench Kármán (would need new infrastructure)
- Multi-physics (RT, combustion) — future work
- Diffusion-based SR baselines (different paradigm)

---

## Immediate next action (today)

Two things should happen first, in parallel:

1. **Create the rich-gate SR experiment config + add to SR queue
   script** (15 min)
2. **Launch the SR queue overnight** (then it runs hands-free)

After the queue is running, the writing of Stage A.4 (Experiments
section) can begin in parallel — no GPU needed for that.

---

## Cross-reference

- [experimental_log.md](experimental_log.md) — what has already been
  run, with numbers and code references
- [gate_entropy_mitigation.md](gate_entropy_mitigation.md) — the
  Mitigation A / B / C playbook that produced the rich gate
- [paper.tex](../paper.tex) — the paper being completed
- [outputs/tables/](../outputs/tables/) — auto-generated CSV +
  LaTeX tables from `scripts/aggregate_*.py`
- [outputs/figures/](../outputs/figures/) — gate-map figures from
  `scripts/analyze_gate_burgers.py`
- [outputs/richgate/figures/](../outputs/richgate/figures/) — rich-
  gate Burgers gate maps (paper hero figure candidates)
