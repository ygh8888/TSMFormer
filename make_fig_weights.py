#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 2 -- per-class fusion weight trajectories with seed bands.

Values are the mean and s.d. over five random seeds produced by
`python extract_trajectory.py perclass` (controlled protocol: NVGesture
901/149/482 subject-disjoint partition; Briareo official validation split).
Bands are drawn at +-1 s.d.; they are narrower than the line width because
the runs are nearly identical.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NVG = dict(
    val=[92.36, 92.08, 91.95, 91.67, 91.95, 91.95, 91.25, 89.58, 89.58, 89.58,
         89.58, 88.19, 86.81, 85.70, 84.03, 82.36, 79.72, 79.17, 78.06, 77.64],
    val_sd=[0.00, 0.38, 0.38, 0.00, 0.38, 0.38, 0.38, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.38, 0.70, 0.62, 0.58, 0.00, 0.38, 0.31],
    w=[[0.217, 0.235, 0.254, 0.274, 0.294, 0.314, 0.336, 0.357, 0.378, 0.399,
        0.421, 0.442, 0.463, 0.484, 0.504, 0.524, 0.544, 0.562, 0.581, 0.598],
       [0.196, 0.191, 0.187, 0.182, 0.177, 0.172, 0.166, 0.161, 0.156, 0.150,
        0.145, 0.140, 0.134, 0.129, 0.124, 0.119, 0.114, 0.110, 0.105, 0.101],
       [0.196, 0.191, 0.186, 0.181, 0.176, 0.171, 0.166, 0.161, 0.156, 0.150,
        0.145, 0.139, 0.134, 0.129, 0.124, 0.119, 0.114, 0.109, 0.105, 0.100],
       [0.196, 0.191, 0.187, 0.182, 0.177, 0.172, 0.166, 0.161, 0.156, 0.150,
        0.145, 0.140, 0.134, 0.129, 0.124, 0.119, 0.114, 0.110, 0.105, 0.101],
       [0.196, 0.191, 0.186, 0.181, 0.176, 0.171, 0.166, 0.161, 0.155, 0.150,
        0.145, 0.139, 0.134, 0.129, 0.124, 0.119, 0.114, 0.109, 0.105, 0.100]],
    w_sd=0.002,
    title=r'NVGesture (25 classes, $M{\times}C$ = 125 params)',
    first='Color', ylim_w=(0.08, 0.66), ylim_v=(75.5, 94.0),
    drop=-14.72, ann=(20.5, 86.5),
)

BRI = dict(
    val=[98.61]*10 + [98.15]*10,
    val_sd=[0.00]*20,
    w=[[0.223, 0.247, 0.272, 0.299, 0.327, 0.355, 0.383, 0.412, 0.441, 0.469,
        0.497, 0.524, 0.550, 0.575, 0.599, 0.622, 0.643, 0.664, 0.683, 0.701],
       [0.194, 0.188, 0.182, 0.175, 0.169, 0.162, 0.154, 0.147, 0.140, 0.133,
        0.126, 0.119, 0.113, 0.107, 0.100, 0.095, 0.089, 0.084, 0.079, 0.075],
       [0.194, 0.188, 0.182, 0.176, 0.169, 0.162, 0.154, 0.147, 0.140, 0.133,
        0.126, 0.119, 0.112, 0.106, 0.100, 0.095, 0.089, 0.084, 0.079, 0.075],
       [0.194, 0.188, 0.182, 0.175, 0.168, 0.161, 0.154, 0.147, 0.139, 0.132,
        0.125, 0.119, 0.112, 0.106, 0.100, 0.094, 0.089, 0.084, 0.079, 0.074],
       [0.194, 0.188, 0.182, 0.175, 0.168, 0.161, 0.154, 0.147, 0.140, 0.133,
        0.126, 0.119, 0.113, 0.106, 0.101, 0.095, 0.089, 0.084, 0.079, 0.075]],
    w_sd=0.002,
    title=r'Briareo (12 classes, $M{\times}C$ = 60 params)',
    first='RGB', ylim_w=(0.05, 0.76), ylim_v=(97.95, 98.80),
    drop=-0.46, ann=(19.2, 98.38),
)

REST = ['Depth', 'IR', 'Normal', 'Opt. flow']
C_FIRST = '#CC0000'
C_REST = ['#1F4E5F', '#4FA3C7', '#5B7B3A', '#808080']
DASHES = ['-', (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1.5, 2))]
EP = np.arange(1, 21)

fig, axes = plt.subplots(2, 2, figsize=(6.78, 3.78), sharex='col',
                         gridspec_kw=dict(height_ratios=[1.25, 1.0],
                                          hspace=0.12, wspace=0.26))

for col, D in enumerate([NVG, BRI]):
    axw, axv = axes[0, col], axes[1, col]
    w = np.array(D['w'])
    sd = D['w_sd']

    axw.axhline(0.2, ls=':', lw=0.9, color='#999999', zorder=1)
    axw.text(20.0, 0.205, r'uniform (1/$M$)', ha='right', va='bottom',
             fontsize=6.5, color='#888888')
    axw.fill_between(EP, w[0]-sd, w[0]+sd, color=C_FIRST, alpha=0.30, lw=0, zorder=3)
    axw.plot(EP, w[0], color=C_FIRST, lw=2.0, label=D['first'], zorder=4)
    for i, (name, c, dsh) in enumerate(zip(REST, C_REST, DASHES), start=1):
        axw.fill_between(EP, w[i]-sd, w[i]+sd, color=c, alpha=0.25, lw=0, zorder=3)
        axw.plot(EP, w[i], color=c, lw=1.2, ls=dsh, label=name, zorder=4+i)
    axw.axvline(1, ls='--', lw=1.0, color='#D9534F', zorder=2)
    if col == 0:
        axw.annotate('best epoch:\nweights $\\approx$ uniform',
                     xy=(1.15, 0.60), xytext=(3.4, 0.60),
                     fontsize=6.5, color='#CC0000', va='center',
                     arrowprops=dict(arrowstyle='->', color='#CC0000', lw=0.8))
    axw.set_title(D['title'], fontsize=8.5, pad=5)
    axw.set_ylim(*D['ylim_w'])
    axw.legend(fontsize=6.3, frameon=False, loc='center right',
               borderpad=0.2, labelspacing=0.28, handlelength=1.5)
    if col == 0:
        axw.set_ylabel('Learned weight', fontsize=8)

    v = np.array(D['val']); vsd = np.array(D['val_sd'])
    axv.axhline(v[0], ls=':', lw=0.9, color='#999999', zorder=1)
    axv.fill_between(EP, v-vsd, v+vsd, color=C_FIRST, alpha=0.25, lw=0, zorder=2)
    axv.plot(EP, v, color=C_FIRST, lw=1.4, marker='o', ms=2.8, zorder=3)
    axv.axvline(1, ls='--', lw=1.0, color='#D9534F', zorder=2)
    lo, hi = D['ylim_v']
    axv.annotate(f"{D['drop']:.2f} pp".replace('-', '\u2212'),
                 xy=(20, v[-1]), xytext=D['ann'],
                 fontsize=7, color='#CC0000', ha='right',
                 arrowprops=dict(arrowstyle='->', color='#CC0000', lw=0.8))
    axv.set_ylim(lo, hi)
    axv.set_xlabel('Training epoch', fontsize=8)
    if col == 0:
        axv.set_ylabel('Val. accuracy (%)', fontsize=8)

    for ax in (axw, axv):
        ax.set_xlim(0.4, 20.6)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(True, lw=0.4, color='#DDDDDD', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7); sp.set_color('#666666')

fig.savefig('fig_weights.pdf', bbox_inches='tight', pad_inches=0.02)
print('saved fig_weights.pdf')
