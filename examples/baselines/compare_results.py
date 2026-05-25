"""
Comparison script: aggregate results from all SOD super-resolution experiments.

Reads every metadata.json found under PROJECT_ROOT/results/ and produces:
  1. A formatted comparison table (stdout + comparison_table.txt)
  2. Bar chart of per-field relative L2 errors (comparison_bar_chart.png)
  3. Bar chart of parameter counts (comparison_params.png)

Run after training all models:
  python examples/baselines/compare_results.py

Or pass an explicit results root:
  python examples/baselines/compare_results.py /path/to/results
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')

FIELD_NAMES = ['Vx', 'density', 'pressure']

# Colours per model family for consistent plots
_COLOR_MAP = {
    'fno':   '#2196F3',    # blue
    'wno':   '#FF9800',    # orange
    'unet':  '#4CAF50',    # green
    'awfno': '#9C27B0',    # purple
}


def _model_color(name: str) -> str:
    name_l = name.lower()
    for key, col in _COLOR_MAP.items():
        if key in name_l:
            return col
    return '#607D8B'   # grey fallback


def load_all_metadata(results_root: str):
    """Walk results_root and collect every metadata.json."""
    records = []
    for dirpath, _, files in os.walk(results_root):
        if 'metadata.json' in files:
            path = os.path.join(dirpath, 'metadata.json')
            try:
                with open(path) as f:
                    meta = json.load(f)
                meta['_results_dir'] = dirpath
                records.append(meta)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")
    return records


def _display_name(meta: dict) -> str:
    """Human-readable label: prefer model_name field, else directory name."""
    name = meta.get('model_name') or meta.get('fusion_type')
    if name:
        return name
    return os.path.basename(meta['_results_dir'])


def print_table(records):
    """Print a markdown-style comparison table."""
    if not records:
        print("No metadata found.")
        return

    col_w = 28
    hdr = (f"{'Model':<{col_w}} {'Params':>10} "
           f"{'Vx Rel-L2':>12} {'density Rel-L2':>14} "
           f"{'pressure Rel-L2':>16} {'Mean Rel-L2':>12}")
    sep = '-' * len(hdr)

    lines = [sep, hdr, sep]
    for meta in sorted(records, key=lambda m: m.get('rel_l2_mean', 9999)):
        name   = _display_name(meta)
        params = meta.get('n_params', '?')
        rl2    = meta.get('rel_l2_per_field', {})
        mean   = meta.get('rel_l2_mean', float('nan'))

        vx  = rl2.get('Vx',       rl2.get('0', float('nan')))
        rho = rl2.get('density',  rl2.get('1', float('nan')))
        p   = rl2.get('pressure', rl2.get('2', float('nan')))

        row = (f"{name:<{col_w}} {params:>10,} "
               f"{vx:>12.6f} {rho:>14.6f} "
               f"{p:>16.6f} {mean:>12.6f}")
        lines.append(row)
    lines.append(sep)
    table = '\n'.join(lines)
    print(table)
    return table


def plot_bar_chart(records, output_dir: str):
    if not records:
        return

    records_sorted = sorted(records, key=lambda m: m.get('rel_l2_mean', 9999))
    names  = [_display_name(m) for m in records_sorted]
    colors = [_model_color(n) for n in names]

    # ── Per-field relative L2 bar chart ──────────────────────────────────────
    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))
    for i, fname in enumerate(FIELD_NAMES):
        vals = []
        for meta in records_sorted:
            rl2 = meta.get('rel_l2_per_field', {})
            v = rl2.get(fname, rl2.get(str(i), float('nan')))
            vals.append(v)
        ax.bar(x + (i - 1) * width, vals, width,
               label=fname, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Per-Field Relative L2 Error — SOD Super-Resolution (×4)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'comparison_bar_chart.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Bar chart saved → {out}")

    # ── Mean relative L2 bar chart ────────────────────────────────────────────
    means = [m.get('rel_l2_mean', float('nan')) for m in records_sorted]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(names, means, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean Relative L2 Error', fontsize=12)
    ax.set_title('Mean Relative L2 — Lower is Better', fontsize=13, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'comparison_mean_l2.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Mean L2 chart saved → {out}")

    # ── Parameter count chart ─────────────────────────────────────────────────
    params = [m.get('n_params', 0) for m in records_sorted]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(names, [p / 1e6 for p in params], color=colors,
                  alpha=0.85, edgecolor='black', linewidth=0.7)
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{p/1e6:.2f}M', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Parameters (millions)', fontsize=12)
    ax.set_title('Trainable Parameter Count', fontsize=13, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'comparison_params.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Params chart saved → {out}")


if __name__ == '__main__':
    results_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_ROOT
    output_dir   = results_root
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning {results_root} for metadata.json …\n")
    records = load_all_metadata(results_root)
    print(f"Found {len(records)} experiment(s).\n")

    table = print_table(records)

    if table:
        table_path = os.path.join(output_dir, 'comparison_table.txt')
        with open(table_path, 'w') as f:
            f.write(table + '\n')
        print(f"\nTable saved → {table_path}")

    plot_bar_chart(records, output_dir)
