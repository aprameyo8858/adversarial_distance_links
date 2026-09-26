import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# LaTeX / Serif Publication Typography Setup
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except ImportError:
    plt.rcParams.update({'font.family': 'serif', 'text.usetex': True})

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.grid.which": "both",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 300
})

FULL_CONFIGS = [
    {
        'key': 'csl_vs_dadv', 'x_col': 'csl', 'y_col': 'd_adv',
        'color': '#D62728', 'shadow': '#f5b5b5',
        'title': r"\textbf{CSL vs.} $\bm{d_{adv}}$",
        'xlabel': r"\textbf{CSL}", 'ylabel': r"$\bm{d_{adv}}$"
    },
    {
        'key': 'csg_vs_dadv', 'x_col': 'csg', 'y_col': 'd_adv',
        'color': '#1F77B4', 'shadow': '#aec7e8',
        'title': r"\textbf{CSG vs.} $\bm{d_{adv}}$",
        'xlabel': r"\textbf{CSG}", 'ylabel': r"$\bm{d_{adv}}$"
    },
    {
        'key': 'csl_vs_csg', 'x_col': 'csl', 'y_col': 'csg',
        'color': '#E377C2', 'shadow': '#f7b6d2',
        'title': r"\textbf{CSL vs. CSG}",
        'xlabel': r"\textbf{CSL}", 'ylabel': r"\textbf{CSG}"
    },
    {
        'key': 'dadv_proxy_vs_dadv', 'x_col': 'd_adv_proxy', 'y_col': 'd_adv',
        'color': '#9467BD', 'shadow': '#c5b0d5',
        'title': r"$\bm{\bar{d}_{adv, T}}$ \textbf{vs.} $\bm{d_{adv}}$",
        'xlabel': r"$\bm{\bar{d}_{adv, T}}$", 'ylabel': r"$\bm{d_{adv}}$"
    }
]

MIA_CONFIGS = [
    {
        'key': 'csl_vs_yeom', 'x_col': 'csl', 'y_col': 'yeom_adv',
        'color': '#2CA02C', 'shadow': '#98df8a',
        'title': r"\textbf{CSL vs. Loss Threshold Adv.}",
        'xlabel': r"\textbf{CSL}", 'ylabel': r"\textbf{Loss Threshold Adv.}"
    },
    {
        'key': 'csl_vs_lira', 'x_col': 'csl', 'y_col': 'lira_adv',
        'color': '#8C564B', 'shadow': '#c49c94',
        'title': r"\textbf{CSL vs. LiRA Advantage}",
        'xlabel': r"\textbf{CSL}", 'ylabel': r"\textbf{LiRA Advantage}"
    },
    {
        'key': 'yeom_vs_dadv', 'x_col': 'yeom_adv', 'y_col': 'd_adv',
        'color': '#FF7F0E', 'shadow': '#ffbb78',
        'title': r"\textbf{Loss Threshold Adv. vs.} $\bm{d_{adv}}$",
        'xlabel': r"\textbf{Loss Threshold Adv.}", 'ylabel': r"$\bm{d_{adv}}$"
    },
    {
        'key': 'lira_vs_dadv', 'x_col': 'lira_adv', 'y_col': 'd_adv',
        'color': '#17BECF', 'shadow': '#9edae5',
        'title': r"\textbf{LiRA Advantage vs.} $\bm{d_{adv}}$",
        'xlabel': r"\textbf{LiRA Advantage}", 'ylabel': r"$\bm{d_{adv}}$"
    }
]

def bin_data(x, y, num_bins=10):
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
    x_bins = np.array([np.mean(b) for b in np.array_split(x_sorted, num_bins) if len(b) > 0])
    y_bins = np.array([np.mean(b) for b in np.array_split(y_sorted, num_bins) if len(b) > 0])
    return x_bins, y_bins

def compute_decile_stats(df, x_col, y_col, seeds, num_bins=10):
    binned_x, binned_y = [], []
    for s in seeds:
        sub = df[df['seed'] == s]
        if len(sub) == 0:
            continue
        bx, by = bin_data(sub[x_col].values, sub[y_col].values, num_bins)
        binned_x.append(bx)
        binned_y.append(by)
        
    min_len = min(len(bx) for bx in binned_x)
    b_x_arr = np.array([bx[:min_len] for bx in binned_x])
    b_y_arr = np.array([by[:min_len] for by in binned_y])
    
    mean_x = np.mean(b_x_arr, axis=0)
    mean_y = np.mean(b_y_arr, axis=0)
    std_y = np.std(b_y_arr, axis=0)
    return mean_x, mean_y, std_y

def render_axis(ax, x, mean_y, std_y, cfg):
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(mean_y - std_y), np.max(mean_y + std_y)
    
    x_span = max(x_max - x_min, 1e-5)
    y_span = max(y_max - y_min, 1e-5)
    
    ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.12 * y_span)

    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=cfg['shadow'], alpha=0.5, label=r'$\pm$ 1 Std. Dev.')
    ax.plot(x, mean_y, marker='o', markersize=4, color=cfg['color'], linewidth=1.2,
            markeredgecolor='black', markeredgewidth=0.5, label=r'Mean Trend')

    ax.set_title(cfg['title'], fontsize=12)
    ax.set_xlabel(cfg['xlabel'], fontsize=11)
    ax.set_ylabel(cfg['ylabel'], fontsize=11)
    ax.grid(True, which='major', color='#CCCCCC', linestyle='-', alpha=0.5)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=10)
    
    formatter = ticker.FuncFormatter(lambda val, pos: r'$\bm{' + f'{val:g}' + r'}$')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

def main():
    parser = argparse.ArgumentParser(description="Plot 8-Panel Theoretical Verification Grid (10 Bins)")
    parser.add_argument('--full-csv', type=str, required=True, help='Full dataset 50k results CSV')
    parser.add_argument('--mia-csv', type=str, required=True, help='MIA half-dataset 25k results CSV')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='./plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_full = pd.read_csv(args.full_csv)
    df_mia = pd.read_csv(args.mia_csv)

    seeds_full = sorted(df_full['seed'].unique())
    seeds_mia = sorted(df_mia['seed'].unique())

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), dpi=300)
    
    # Top Row: 50k Full Dataset Relationships
    for i, cfg in enumerate(FULL_CONFIGS):
        ax = axes[0, i]
        x, my, sy = compute_decile_stats(df_full, cfg['x_col'], cfg['y_col'], seeds_full, num_bins=10)
        render_axis(ax, x, my, sy, cfg)
        if i == 0:
            ax.legend(frameon=True, fontsize=9, loc='upper right')

    # Bottom Row: 25k Half Dataset MIA Relationships
    for i, cfg in enumerate(MIA_CONFIGS):
        ax = axes[1, i]
        x, my, sy = compute_decile_stats(df_mia, cfg['x_col'], cfg['y_col'], seeds_mia, num_bins=10)
        render_axis(ax, x, my, sy, cfg)

    fig.tight_layout(pad=0.6)
    out_base = os.path.join(args.out_dir, f"verification_grid_{args.dataset}_{args.model}_10bins")
    fig.savefig(out_base + ".pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[+] Successfully generated Figure 1 replication plot at {out_base}.png")

if __name__ == '__main__':
    main()
