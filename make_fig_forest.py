#!/usr/bin/env python3
"""Fig. 3 -- paired differences of the learned fusion variants from late fusion.

Values from `python stats_fix.py --dir analysis_fix` (controlled protocol,
NVGesture 482 test sequences). Bars: 95% paired-bootstrap CI of the seed-mean
difference over test sequences (10,000 resamples); ticks: individual seeds.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PP = 100 / 482
# name, n_seeds, (diff, lo, hi) vs LF5, per-seed sequence diffs vs LF5, (diff, lo, hi) vs LF-val
ROWS = [
    ('Per-class weighting', 5, (-0.25, -0.66, 0.04), [0, -1, -1, -1, -3], (-0.87, -2.57, 0.75)),
    ('Dense cross-attention', 3, (0.00, -1.73, 1.73), [5, -4, -1], (-0.62, -2.28, 0.97)),
    ('LoRA (color, IR) + bottleneck', 5, (0.17, -1.45, 1.74), [2, -1, 3, 0, 0], (-0.46, -1.99, 1.04)),
    ('Bottleneck + stronger input', 5, (0.91, -0.71, 2.66), [4, 3, 5, 5, 5], (0.29, -1.29, 1.87)),
    ('Bottleneck + hybrid gate', 3, (1.04, -0.62, 2.77), [7, 4, 4], (0.41, -1.31, 2.14)),
]
C_LF5, C_VAL = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e2e2e2'

fig, ax = plt.subplots(figsize=(3.45, 2.55))
y = np.arange(len(ROWS))[::-1].astype(float)
off = 0.17
for yi, (nm, n, a, seeds, b) in zip(y, ROWS):
    # vs five-modality late fusion
    ax.plot([a[1], a[2]], [yi + off] * 2, color=C_LF5, lw=1.6, solid_capstyle='butt', zorder=3)
    ax.plot(a[0], yi + off, 'o', color=C_LF5, ms=5.5, mec='white', mew=0.8, zorder=5)
    jit = np.linspace(-0.05, 0.05, len(seeds))
    ax.plot(np.array(seeds) * PP, yi + off + 0.13 + jit * 0, '|', color=C_LF5, ms=4, mew=0.9,
            alpha=0.75, zorder=4)
    # vs validation-selected late fusion subset
    ax.plot([b[1], b[2]], [yi - off] * 2, color=C_VAL, lw=1.6, solid_capstyle='butt', zorder=3)
    ax.plot(b[0], yi - off, 'D', color=C_VAL, ms=4.6, mec='white', mew=0.8, zorder=5)
ax.axvline(0, color='#555555', lw=0.9, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([f'{r[0]} ({r[1]})' for r in ROWS], fontsize=7, color=INK)
ax.set_xlim(-3.0, 3.0)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xlabel('Difference from late fusion (pp), 95% CI', fontsize=7.5, color=INK)
ax.set_ylim(-0.6, len(ROWS) - 0.35)
sec = ax.secondary_xaxis('top', functions=(lambda v: v / PP, lambda s: s * PP))
sec.set_xticks([-14, -10, -5, 0, 5, 10, 14])
sec.tick_params(labelsize=6.5, colors=MUTED, length=2)
sec.set_xlabel('test sequences (of 482)', fontsize=6.5, color=MUTED, labelpad=2)
for s in sec.spines.values():
    s.set_color('#888888'); s.set_linewidth(0.7)
ax.grid(True, axis='x', lw=0.4, color=GRID, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=7, colors=INK)
ax.tick_params(axis='y', length=0)
for sp in ax.spines.values():
    sp.set_linewidth(0.7); sp.set_color('#888888')
h1 = ax.plot([], [], 'o-', color=C_LF5, ms=5, lw=1.4, label='vs five-modality late fusion (84.44%)')[0]
h2 = ax.plot([], [], 'D-', color=C_VAL, ms=4.2, lw=1.4,
             label='vs validation-selected subset (85.06%)')[0]
h3 = ax.plot([], [], '|', color=C_LF5, ms=5, mew=0.9, label='individual seeds')[0]
ax.legend(handles=[h1, h2, h3], fontsize=6.2, frameon=False, loc='upper center',
          bbox_to_anchor=(0.36, -0.22), ncol=1, handlelength=2.0, labelspacing=0.25)
fig.savefig('fig_forest.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('fig_forest_preview.png', dpi=250, bbox_inches='tight', pad_inches=0.02)
print('saved fig_forest.pdf')
