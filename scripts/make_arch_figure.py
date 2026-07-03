#!/usr/bin/env python
"""Generate paper figures that have no data dependency:

  figures/arch.pdf        -- AW-FNO architecture diagram (Fig 1)
  iccfd13_banner.pdf      -- placeholder conference banner (replace with the
                             official ICCFD13 logo before submission)

Both are vector PDFs so they scale cleanly in the LaTeX build. Run:
    python scripts/make_arch_figure.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The paper lives in docs/paper/; emit figures relative to it so the build is
# self-contained (paper.tex uses \includegraphics{figures/arch.pdf} and the
# {iccfd13_banner} logo, both resolved relative to the .tex location).
PAPER_DIR = os.path.join(ROOT, "docs", "paper")
FIGDIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- palette ----
C_LIFT = "#dfe7f3"
C_FNO = "#cfe8d8"
C_WNO = "#f5e0cf"
C_GATE = "#f3d6e4"
C_FUSE = "#e6dcf0"
C_PROJ = "#dfe7f3"
EDGE = "#3a3a3a"


def box(ax, x, y, w, h, text, fc, fs=10, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=lw, edgecolor=EDGE, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, fs=8, rad=0.0, color=EDGE):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                        linewidth=1.1, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom", fontsize=fs)


def cx_right(b):  # mid-right point of a box
    x, y, w, h = b
    return (x + w, y + h / 2)


def cx_left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def cx_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def cx_bot(b):
    x, y, w, h = b
    return (x + w / 2, y)


def make_arch():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---------- top: overall pipeline ----------
    ylift = 6.6
    b_in = box(ax, 0.3, ylift, 2.2, 1.2, "LR input\n$a(x)$", C_LIFT, fs=9)
    b_p = box(ax, 3.0, ylift, 1.6, 1.2, "Lift\n$P$", C_LIFT)
    b_blocks = box(ax, 5.1, ylift, 7.6, 1.2,
                   r"$N \times$  AW-FNO block  (detail below)", "#eef1f6", fs=10)
    b_q = box(ax, 13.2, ylift, 1.6, 1.2, "Project\n$Q$", C_PROJ)
    b_out = box(ax, 15.3, ylift, 2.2, 1.2, "HR output\n$u(x)$", C_LIFT, fs=9)

    arrow(ax, cx_right(b_in), cx_left(b_p))
    arrow(ax, cx_right(b_p), cx_left(b_blocks))
    arrow(ax, cx_right(b_blocks), cx_left(b_q))
    arrow(ax, cx_right(b_q), cx_left(b_out))

    # connector from the block band down to the detail panel
    ax.plot([8.9, 8.9], [ylift, 5.3], color="#9aa3b2", lw=1.0, ls=(0, (4, 3)),
            zorder=0)

    # ---------- bottom: one block detail ----------
    yb = 1.9
    b_vt = box(ax, 0.3, yb + 0.9, 1.9, 1.2, r"$v_t$", "#eef1f6")

    # two branches
    b_fno = box(ax, 3.4, yb + 2.4, 5.0, 1.2,
                r"FNO branch:  $\mathcal{F}^{-1}(R_\phi\,\mathcal{F}\,v_t)$", C_FNO, fs=9)
    b_wno = box(ax, 3.4, yb - 0.1, 5.0, 1.2,
                r"WNO branch:  $\mathcal{W}^{-1}(R_\phi\,\mathcal{W}\,v_t)$", C_WNO, fs=9)

    arrow(ax, cx_right(b_vt), cx_left(b_fno), rad=0.18)
    arrow(ax, cx_right(b_vt), cx_left(b_wno), rad=-0.18)

    # gate
    b_gate = box(ax, 9.4, yb + 1.15, 4.0, 1.2,
                 r"Gate  $\alpha=\sigma(\mathrm{Conv}_k\,\mathrm{GELU}\,\mathrm{Conv}_k[V_F,V_W])$",
                 C_GATE, fs=8)
    arrow(ax, cx_right(b_fno), (9.4, yb + 1.95), rad=-0.15, color="#2e7d4f")
    arrow(ax, cx_right(b_wno), (9.4, yb + 1.55), rad=0.15, color="#b5651d")

    # fusion
    b_fuse = box(ax, 14.4, yb + 1.15, 4.2, 1.2,
                 r"$V_{\mathrm{fused}}=\alpha\,V_F+(1-\alpha)V_W$", C_FUSE, fs=9)
    arrow(ax, cx_right(b_gate), cx_left(b_fuse), text=r"$\alpha$")
    # branch outputs also feed the fusion (curved over/under the gate)
    arrow(ax, cx_top(b_fno), (16.5, yb + 2.35), rad=-0.25, color="#2e7d4f")
    arrow(ax, cx_bot(b_wno), (16.5, yb + 0.95), rad=0.25, color="#b5651d")

    # add + norm  -> v_{t+1}
    b_out2 = box(ax, 19.0, yb + 1.15, 2.7, 1.2,
                 r"$+Wv_t$" + "\nLN, GELU", "#eef1f6", fs=8)
    arrow(ax, cx_right(b_fuse), cx_left(b_out2))
    ax.text(20.35, yb + 2.55, r"$v_{t+1}$", ha="center", fontsize=10)

    # skip connection W v_t
    ax.annotate("", xy=(19.4, yb + 2.35), xytext=(1.25, yb + 0.85),
                arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.0,
                                connectionstyle="arc3,rad=-0.28"), zorder=0)
    ax.text(9.5, yb - 0.95, r"residual  $W v_t$", color="#7a7a7a", fontsize=8,
            ha="center")

    fig.tight_layout()
    out = os.path.join(FIGDIR, "arch.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def make_banner():
    fig, ax = plt.subplots(figsize=(6.0, 1.0))
    ax.axis("off")
    ax.text(0.0, 0.5, "ICCFD13", fontsize=22, fontweight="bold",
            va="center", ha="left", color="#1f3b6e")
    ax.text(0.30, 0.5, "  Milan, Italy  ·  July 06–10, 2026",
            fontsize=11, va="center", ha="left", color="#444",
            transform=ax.transAxes)
    out = os.path.join(PAPER_DIR, "iccfd13_banner.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "(placeholder — replace with official banner)")


if __name__ == "__main__":
    make_arch()
    make_banner()
