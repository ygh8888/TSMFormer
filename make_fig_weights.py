#!/usr/bin/env python3
"""Fig. 2 -- per-class weighting after the sampler fix (controlled protocol).

Mean +- s.d. over five seeds from `python extract_trajectory.py perclass_fix`
and `python extract_trajectory.py bri_perclass_fix`. Bands are narrower than the
line width (s.d. <= 0.001). NVGesture validation accuracy is computed by the
training loop on 144 of the 149 validation sequences (last incomplete batch
dropped); Briareo uses all 216.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NVG = dict(
    val=[92.22, 92.08, 92.08, 91.95, 91.81, 91.81, 91.81, 91.81, 91.67, 91.67,
         91.67, 91.67, 91.67, 91.67, 91.67, 91.67, 91.67, 91.67, 91.81, 92.36],
    val_sd=[0.31, 0.38, 0.38, 0.38, 0.31, 0.31, 0.31, 0.31, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0.31, 0],
    w=[[.202, .203, .204, .206, .207, .208, .209, .210, .211, .212, .213, .214, .214, .215, .216, .217, .217, .218, .219, .219],
       [.199, .198, .197, .196, .195, .194, .194, .193, .192, .191, .190, .189, .188, .188, .187, .186, .185, .184, .184, .183],
       [.199, .198, .198, .197, .196, .195, .195, .194, .193, .193, .192, .191, .190, .190, .189, .188, .188, .187, .187, .186],
       [.201, .201, .201, .202, .202, .203, .203, .204, .204, .205, .205, .206, .207, .207, .208, .208, .209, .210, .210, .211],
       [.200, .200, .200, .200, .200, .200, .200, .200, .200, .200, .200, .200, .200, .201, .201, .201, .201, .201, .201, .201]],
    names=['Color', 'Depth', 'IR', 'Normal', 'Opt. flow'],
    title=r'NVGesture (25 classes, $M{\times}C$ = 125 params)',
    ylim_v=(90.9, 93.1),
)
BRI = dict(
    val=[98.61] * 20, val_sd=[0.0] * 20,
    w=[[.203, .205, .207, .209, .211, .213, .215, .216, .218, .219, .220, .221, .222, .223, .223, .224, .224, .225, .225, .225],
       [.201, .201, .201, .202, .202, .202, .203, .203, .203, .203, .203, .203, .203, .203, .203, .203, .203, .203, .203, .203],
       [.196, .193, .189, .186, .183, .180, .177, .174, .172, .169, .166, .164, .161, .159, .157, .154, .152, .150, .148, .146],
       [.199, .197, .195, .194, .192, .191, .190, .188, .187, .186, .185, .184, .183, .182, .180, .179, .178, .177, .176, .175],
       [.203, .205, .207, .209, .211, .213, .215, .218, .220, .223, .225, .228, .230, .233, .236, .239, .242, .245, .248, .251]],
    names=['RGB', 'Depth', 'IR', 'Normal', 'Opt. flow'],
    title=r'Briareo (12 classes, $M{\times}C$ = 60 params)',
    ylim_v=(97.5, 99.7),
)

# reference categorical palette (validated: CVD dE >= 9.2, normal-vision dE >= 27.6);
# dash patterns give a second, colour-independent encoding
COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948']
DASHES = ['-', (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1.5, 2)), (0, (3, 1, 1, 1, 1, 1))]
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e2e2e2'
EP = np.arange(1, 21)

fig, axes = plt.subplots(2, 2, figsize=(6.78, 3.6), sharex='col',
                         gridspec_kw=dict(height_ratios=[1.3, 1.0], hspace=0.12, wspace=0.24))
for col, D in enumerate([NVG, BRI]):
    axw, axv = axes[0, col], axes[1, col]
    w = np.array(D['w'])
    axw.axhline(0.2, ls=':', lw=0.9, color=MUTED, zorder=1)
    axw.text(0.98, 0.04, r'dotted line: uniform weight 1/$M$', transform=axw.transAxes,
             ha='right', va='bottom', fontsize=6.3, color=MUTED)
    for i, (nm, c, d) in enumerate(zip(D['names'], COLS, DASHES)):
        axw.plot(EP, w[i], color=c, lw=1.5, ls=d, label=nm, zorder=3 + i)
    axw.set_ylim(0.10, 0.30)
    axw.set_title(D['title'], fontsize=8.5, pad=4, color=INK)
    axw.legend(fontsize=6.3, frameon=False, loc='upper left', ncol=3,
               columnspacing=0.9, handlelength=2.0, borderaxespad=0.2, labelspacing=0.25)
    if col == 0:
        axw.set_ylabel('Learned weight', fontsize=8, color=INK)

    v, sd = np.array(D['val']), np.array(D['val_sd'])
    axv.fill_between(EP, v - sd, v + sd, color='#2a78d6', alpha=0.18, lw=0, zorder=2)
    axv.plot(EP, v, color='#2a78d6', lw=1.5, marker='o', ms=2.6, zorder=3)
    axv.set_ylim(*D['ylim_v'])
    axv.set_xlabel('Training epoch', fontsize=8, color=INK)
    if col == 0:
        axv.set_ylabel('Val. accuracy (%)', fontsize=8, color=INK)
    note = (f"constant at {v[0]:.2f}% (1 sequence = {100/216:.2f} pp)" if v.min() == v.max() else
            f"range {v.min():.2f}\u2013{v.max():.2f}% (1 sequence = {100/144:.2f} pp)")
    axv.text(0.98, 0.08, note, transform=axv.transAxes, ha='right', va='bottom', fontsize=6.5, color=MUTED)

    for ax in (axw, axv):
        ax.set_xlim(0.4, 20.6)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(True, lw=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors=INK)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7); sp.set_color('#888888')

fig.savefig('fig_weights.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('fig_weights_preview.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
print('saved fig_weights.pdf')
