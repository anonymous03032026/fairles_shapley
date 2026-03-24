"""
plot_contributions.py  LES-Plot
====================================


Stage 1 (group-level)
  les_group_bar       Horizontal bars: one  per ESL method

Stage 2 (feature-level)
  les_bar             Ranked horizontal bars for one group

Multi-method
  les_summary_dot     One panel per ESL method 
  les_majority_voting Hypothesis-test decision table (majority rule)

All functions return ``(fig, ax_or_axes)`` and accept ``save_path``.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
})

# Colour constants
_C0   = '#D55E00'   # women  (orange-red)
_C1   = '#0072B2'   # men    (blue)
_BG   = '#F7F7F7'
_GRID = '#E0E0E0'

_ESL_COLORS = {
    'Shapley':    '#4C72B0',
    'ES':         '#DD8452',
    'Solidarity': '#55A868',
    'Consensus':  '#C44E52',
    'LSP':        '#8172B2',
}
_ESL_ORDER = ['Shapley', 'ES', 'Solidarity', 'Consensus', 'LSP']


def _methods(d):
    return [m for m in _ESL_ORDER if m in d]


def _finish(fig, save_path):
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def _clean_ax(ax, grid='x'):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if grid == 'x':
        ax.xaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.yaxis.grid(False)
    else:
        ax.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ===========================================================================
# STAGE 1  -  group-level attribution
# ===========================================================================

def les_group_bar(
    phi_dict,
    group_labels=None,
    title='First-stage :  ESL group contributions  ',
    figsize=(7, 3.8),
    save_path=None,
):
    """
    Horizontal grouped bar chart: for each ESL method, two bars
    (one per group) showing the group-level ESL contribution.

    Parameters
    ----------
    phi_dict : dict  {esl_name: [phi_group0, phi_group1]}
    """
    if group_labels is None:
        group_labels = ['Group 0', 'Group 1']
    meths = _methods(phi_dict)
    n, h  = len(meths), 0.30
    y     = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    _clean_ax(ax)

    for j, (lbl, col) in enumerate(
        zip(group_labels, [_C0, _C1])
    ):
        vals = [float(phi_dict[m][j]) for m in meths]
        bars = ax.barh(
            y + (j - 0.5) * h, vals, h,
            color=col, alpha=0.85, label=lbl,
            edgecolor='white', linewidth=0.6, zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.004, bar.get_y() + h / 2,
                f'{v:.3f}',
                va='center', ha='left',
                fontsize=8, color=col, fontweight='bold',
            )

    ax.set_yticks(y)
    ax.set_yticklabels(meths, fontsize=10)
    ax.set_xlabel('ESL group contributions', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axvline(0, color='#AAAAAA', linewidth=0.9, linestyle='--')
    ax.legend(
        frameon=False, fontsize=9,
        bbox_to_anchor=(1.01, 1), loc='upper left',
    )
    return _finish(fig, save_path), ax



# ===========================================================================
# STAGE 2    feature-level attribution
# ===========================================================================


def les_bar(
    contrib_dict,
    feature_names,
    group_label='Group',
    title=None,
    figsize=(7, 4),
    save_path=None,
):
    """
    Horizontal bar chart: feature contributions for one group,
    one coloured bar per ESL method.
    Features sorted top-to-bottom by mean |contribution|.

    Parameters
    ----------
    contrib_dict : dict  {esl_name: array(n_features,)}
    """
    if title is None:
        title = f'LES bar    feature contributions  |  {group_label}'

    meths = _methods(contrib_dict)
    k     = len(feature_names)

    mean_abs = np.mean(np.abs(np.vstack(
        [contrib_dict[m] for m in meths]
    )), axis=0)
    order       = np.argsort(mean_abs)
    feat_sorted = [feature_names[i] for i in order]

    n_m = len(meths)
    h   = 0.75 / n_m
    y   = np.arange(k)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    _clean_ax(ax)

    for j, m in enumerate(meths):
        vals   = np.asarray(contrib_dict[m]).ravel()[order]
        offset = (j - (n_m - 1) / 2) * h
        ax.barh(
            y + offset, vals, h * 0.92,
            color=_ESL_COLORS.get(m, '#777777'),
            alpha=0.85, label=m,
            edgecolor='white', linewidth=0.4, zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(feat_sorted, fontsize=10)
    ax.axvline(0, color='#AAAAAA', linewidth=0.9, linestyle='--')
    ax.set_xlabel('ESL contribution  C^k', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.legend(
        frameon=False, fontsize=8.5,
        bbox_to_anchor=(1.01, 1), loc='upper left',
    )
    return _finish(fig, save_path), ax




# ===========================================================================
# Multi-method summary  
# ===========================================================================

def les_summary_dot(
    contrib_women_dict,
    contrib_men_dict,
    feature_names,
    group_labels=None,
    title='Contributions of features to group-ESL ',
    figsize=(13, 4.5),
    save_path=None,
    show_values=True,
    value_fmt='{:+.3f}',
):
    if group_labels is None:
        group_labels = ['Women', 'Men']

    meths = _methods(contrib_women_dict)
    k = len(feature_names)

    all_abs = np.mean(
        np.abs(
            np.vstack(
                [contrib_women_dict[m] for m in meths]
                + [contrib_men_dict[m] for m in meths]
            )
        ),
        axis=0,
    )
    order = np.argsort(all_abs)
    feat_sorted = [feature_names[i] for i in order]
    y = np.arange(k)

    fig, axes = plt.subplots(
        1,
        len(meths),
        figsize=figsize,
        facecolor='white',
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.10)

    if len(meths) == 1:
        axes = [axes]

    for ax, m in zip(axes, meths):
        _clean_ax(ax)
        cw = np.asarray(contrib_women_dict[m]).ravel()[order]
        cm = np.asarray(contrib_men_dict[m]).ravel()[order]

        for i in range(k):
            ax.plot(
                [cw[i], cm[i]],
                [i, i],
                color='#CCCCCC',
                linewidth=1.8,
            )

        ax.scatter(
            cw,
            y,
            color=_C0,
            s=55,
            zorder=4,
            label=group_labels[0] if m == meths[0] else '',
            edgecolors='white',
            linewidths=0.4,
        )
        ax.scatter(
            cm,
            y,
            color=_C1,
            s=55,
            zorder=4,
            label=group_labels[1] if m == meths[0] else '',
            edgecolors='white',
            linewidths=0.4,
        )

        if show_values:
            for i in range(k):
                ax.text(
                    cw[i] - 0.01,
                    y[i] + 0.10,
                    value_fmt.format(cw[i]),
                    color=_C0,
                    fontsize=7.5,
                    ha='right',
                    va='center',
                )
                ax.text(
                    cm[i] + 0.01,
                    y[i] - 0.10,
                    value_fmt.format(cm[i]),
                    color=_C1,
                    fontsize=7.5,
                    ha='left',
                    va='center',
                )

        ax.axvline(0, color='#AAAAAA', linewidth=0.8, linestyle='--')
        ax.set_title(
            m,
            fontsize=10,
            fontweight='bold',
            color=_ESL_COLORS.get(m, '#333333'),
        )
        ax.set_xlabel('feature contribution', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(feat_sorted, fontsize=10)
    axes[0].set_ylabel('Feature', fontsize=9)

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc='upper right',
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.99, 0.98),
    )
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    return _finish(fig, save_path), axes
# ===========================================================================
# Majority voting
# ===========================================================================

def les_majority_voting(
    test_results,
    feature_names,
    alpha=0.05,
    title='LES majority voting    H0: C^k_men-C^k_women=0',
    figsize=(8, 3.8),
    save_path=None,
):
    """
    / heatmap with majority-vote summary row.

    Parameters
    ----------
    test_results : dict  {esl_name: [{'p_value': float}, &]}
                  one dict per feature
    """
    meths      = _methods(test_results)
    n_m        = len(meths)
    n_f        = len(feature_names)
    reject     = np.zeros((n_m, n_f), dtype=bool)
    for i, m in enumerate(meths):
        for j, r in enumerate(test_results[m]):
            reject[i, j] = r['p_value'] < alpha

    thr      = n_m // 2 + 1
    majority = reject.sum(axis=0) >= thr
    display  = np.vstack([reject, majority.reshape(1, -1)])
    rlabels  = meths + ['Majority vote']

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.6, n_f - 0.4)
    ax.set_ylim(-0.6, len(rlabels) - 0.4)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val    = display[i, j]
            bg     = to_rgba('#2CA02C', 0.18) if val \
                     else to_rgba('#D62728', 0.10)
            symbol = 'x' if val else 'o'
            color  = '#2CA02C' if val else '#D62728'
            ax.add_patch(mpatches.FancyBboxPatch(
                (j - 0.44, i - 0.41), 0.88, 0.82,
                boxstyle='round,pad=0.05',
                facecolor=bg, edgecolor='white', linewidth=1.5,
            ))
            ax.text(
                j, i, symbol, ha='center', va='center',
                fontsize=15, color=color, fontweight='bold',
            )

    ax.axhline(
        n_m - 0.5, color='#AAAAAA',
        linewidth=1.5, linestyle='--',
    )
    ax.set_xticks(range(n_f))
    ax.set_xticklabels(feature_names, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(rlabels)))
    ax.set_yticklabels(rlabels, fontsize=10)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    rej_p = mpatches.Patch(
        facecolor=to_rgba('#2CA02C', 0.25),
        edgecolor='#2CA02C', linewidth=1,
        label=f'Reject H0  (p < {alpha})',
    )
    acc_p = mpatches.Patch(
        facecolor=to_rgba('#D62728', 0.18),
        edgecolor='#D62728', linewidth=1,
        label='Fail to reject H0',
    )
    ax.legend(
        handles=[rej_p, acc_p], frameon=False, fontsize=9,
        bbox_to_anchor=(1.01, 1), loc='upper left',
    )
    return _finish(fig, save_path), ax
