# TSMFormer: Late versus Learned Cross-Modal Fusion for Multi-Sensor Gesture Recognition

Code for **"Late Versus Learned Cross-Modal Fusion for Multi-Sensor Dynamic Hand Gesture
Recognition: A Controlled Study on Two Automotive Benchmarks"** (IEEE Sensors Journal, under
review).

The repository reproduces the multimodal dynamic hand gesture recognition experiments on the
**NVGesture** and **Briareo** datasets: the per-modality TSM backbones, the late-fusion
baseline, the learned cross-modal fusion variants with paired statistics, the complementarity
and training-agreement analyses, and the cross-benchmark transfer test reported in the paper.

> **Summary of findings** (controlled protocol: NVGesture 901/149/482 with subjects 7 and 15
> held out for validation, Briareo split distributed with the dataset, identical backbones and
> model selection for every variant, paired tests on the same 482 NVGesture test sequences).
> * **No learned fusion variant differs significantly from late fusion.** Five-modality late
>   fusion reaches 84.44% on NVGesture. Per-class weighting 84.19 ± 0.23% (5 seeds), dense
>   cross-attention 84.44 ± 0.95% (3), LoRA (color, IR) + bottleneck 84.61 ± 0.34% (5),
>   bottleneck + stronger input 85.35 ± 0.19% (5), bottleneck + hybrid gate 85.48 ± 0.36% (3).
>   Every 95% paired-bootstrap interval includes zero (Holm-adjusted p >= 0.87), and late fusion
>   over the modality subset chosen on the validation set reaches 85.06% without any training.
>   On Briareo, per-class weighting returns 98.26% in all five runs, identical to late fusion.
> * **Why.** On the sequences the fusion modules are trained on, the five frozen backbones agree
>   and are correct in 99.3% (NVGesture) and 100% (Briareo) of cases, against 36% and 88% on
>   the test sets, so a learned combiner sees almost no disagreement to learn from.
> * **Complementarity.** On the controlled NVGesture backbones an oracle over the five
>   modalities reaches 95.64%, 11.20 pp above late fusion; the learned variants recover at most
>   1.04 pp of it. On Briareo the oracle equals the best late-fusion subset.
> * **Sensors.** Removing the separate stereo IR device of NVGesture changes late fusion by at
>   most one test sequence under both protocols.
> * **Cross-benchmark transfer.** Linear probes on frozen backbones show representations tied
>   to the device and viewpoint that produced them.

## Changes in `sensors-rev1.4`

All learned-fusion results of earlier tags are **superseded**. In the fusion training scripts
of `sensors-rev1.3` and before, each modality had its own shuffled training loader and the
label was taken from the first (color) loader, so the other four modalities of a training
batch referred to different sequences than the label. Validation and test loaders were not
affected. `sensors-rev1.4`

* draws one shared permutation per epoch for all per-modality training loaders
  (`utils/aligned_sampler.py`, used by `train_cmaf.py`, `train_cmaf_v4.py`,
  `train_cmaf_v4_lora.py` and `train_perclass.py`);
* stores the complete state of the LoRA-adapted backbones, including their batch-normalization
  statistics, in every LoRA checkpoint, and checks after training that the restored model
  reproduces the logged validation accuracy (`VAL CHECK`);
* re-runs every fusion variant (`run_stage2_fix.sh`, `run_lora_fix2.sh`,
  `run_stage2_briareo_fix.sh`) and adds the per-sequence re-evaluation and paired statistics
  (`eval_fix_preds.py`, `stats_fix.py`) and the training-agreement analysis
  (`train_agreement.py`).

Late fusion, the unimodal backbones, the subset, complementarity and transfer results are
unchanged. `run_stage2.sh`, `run_stage2_briareo.sh` and `collect_stage2.py` are kept only to
document the superseded runs.

## Architecture

Each modality is processed by an **independent** pipeline:

```
Input (T frames) -> TSM-ResNet-18 backbone -> GestFormer-style temporal encoder
                 -> FC classifier -> per-modality class probabilities
```

The temporal encoder (6 layers) follows the GestFormer design and, per layer, applies:
1. a wavelet coefficient processing stage (2D DWT with `db3`, one level -> per-subband
   depthwise convolution -> inverse DWT),
2. a multi-scale pooling token mixer (window sizes 3/5/7), and
3. a gated depthwise-convolution feed-forward block.

The modalities are combined only at a final **late-fusion** step: an average of the softmax
probabilities, with no learned parameters. The learned fusion variants replace this step.

## Repository structure

```
models/
|-- temporal.py              # main model (TSM backbone + temporal encoder)
|-- attention.py             # GestFormer-style encoder
|-- backbones/resnet_tsm.py  # TSM-ResNet-18 backbone
|-- fusion.py                # dense cross-attention fusion
|-- fusion_v4.py             # bottleneck (MBT) fusion with hybrid gate
|-- lora_inject.py           # LoRA adapters for backbone adaptation
`-- perclass_fusion.py       # per-class modality weighting
utils/aligned_sampler.py     # shared per-epoch permutation for the per-modality loaders
datasets/                    # NVGesture / Briareo loaders, surface normals, optical flow
hyperparameters/
|-- NVGestures/              # benchmark protocol and fusion configurations
|-- NVGestures_rev/          # controlled protocol backbones (901/149/482)
`-- Briareo/
splits/NVGestures/           # subject-disjoint train/validation lists (controlled protocol)
main.py                      # unimodal training / testing (--phase train|test)
cs.py                        # late fusion and modality-subset evaluation
train_cmaf.py                # dense cross-attention fusion
train_cmaf_v4.py             # bottleneck fusion (hybrid gate / stronger input)
train_cmaf_v4_lora.py        # LoRA (color, IR) + bottleneck
train_perclass.py            # per-class modality weighting
make_seed_cfg.py, make_seed_cfg_bri.py   # per-seed runtime configurations
run_stage1.sh                # controlled-protocol backbones
run_stage2_fix.sh            # fusion variants, multi-seed (NVGesture)
run_lora_fix2.sh             # LoRA variant, multi-seed, full backbone state saved
run_stage2_briareo_fix.sh    # per-class weighting, multi-seed (Briareo)
eval_fix_preds.py            # re-evaluate selected checkpoints, per-sequence posteriors
stats_fix.py                 # McNemar, paired bootstrap, Holm; validation-selected subset
train_agreement.py           # training/validation/test agreement, leave-one-out
analyze_complementarity.py   # pairwise Q statistic, oracle accuracy (benchmark protocol)
extract_trajectory.py        # per-class weight trajectories
make_fig_weights.py          # Fig. 2
make_fig_forest.py           # Fig. 3
transfer_probe.py, run_transfer.sh       # cross-benchmark linear probe
analysis_fix/                # per-sequence posteriors and statistics behind Tables V-VI, Fig. 3
experiments/BL5_revision/analysis/       # weight trajectories behind Fig. 2
experiments/BL6_transfer/    # transfer results
logs_revision/               # training and evaluation logs of the reported runs
requirements-lock.txt, environment.yml   # pinned environment
```

## Datasets

* **NVGesture** [Molchanov et al., CVPR 2016]: 25 classes; official split of 1,050 training
  and 482 test sequences (12 and 6 subjects). The controlled protocol further holds out
  subjects 7 and 15 (149 sequences) for validation, leaving 901 for training.
* **Briareo** [Manganaro et al., ICIAP 2019]: 12 classes; the training, validation and test
  splits distributed with the dataset (936 / 216 / 288 sequences).

Obtain the datasets from their original sources and set the data paths in the JSON
configurations under `hyperparameters/`. Surface normals are computed from depth. Optical
flow (Farnebäck) is computed from depth on NVGesture and from color on Briareo
(`datasets/utils/`).

## Reproducing the results

Run all commands from the repository root after setting `data_path`, `csv_dir` and the
checkpoint paths in the relevant JSON configuration. `<ds>` is `NVGestures` or `Briareo` for
configuration folders and `Nvgestures` or `Briareo` for `--dataset`; `<mod>` ranges over the
five modalities. The same table appears as Table XI in Appendix I of the paper.

| Result | Command |
|---|---|
| Tables II, III (unimodal) | `python main.py --hypes hyperparameters/<ds>/test_<mod>_tsm.json --phase test` |
| Tables IV, VII (late fusion, subsets) | `python cs.py --dataset <ds> --csv_dir experiments/BL2_tsm/<ds>/csv --all` |
| Table VIII (complementarity) | `python analyze_complementarity.py --dataset <ds> --csv_dir <csv>` |
| Controlled-protocol backbones | `./run_stage1.sh` |
| Table V (training, NVGesture) | `./run_stage2_fix.sh ; ./run_lora_fix2.sh` |
| Table V (training, Briareo) | `./run_stage2_briareo_fix.sh` |
| Table V, Fig. 3 (paired statistics) | `python eval_fix_preds.py ; python stats_fix.py ; python make_fig_forest.py` |
| Table VI, leave-one-out | `python train_agreement.py` |
| Fig. 2 (weight trajectories) | `python extract_trajectory.py <tag> ; python make_fig_weights.py` (`<tag>` = `perclass_fix`, `bri_perclass_fix`) |
| Table IX (cross-benchmark transfer) | `./run_transfer.sh ; python transfer_probe.py summary` |

In `run_stage2_fix.sh` the LoRA entry (`lora_p0_fix`) predates the full-state checkpoint and is
superseded by `run_lora_fix2.sh` (`lora_p0_fix2`), which is the LoRA run reported in the paper.

## Trained checkpoints

The checkpoints are linked from the [release page](https://github.com/ygh8888/TSMFormer/releases)
of this tag, together with a manifest (experiment, protocol, seed and supported table or figure
of every file) and SHA-256 checksums. The 26 fusion checkpoints of `sensors-rev1.4` are
distributed as a separate archive; the backbones are those of the earlier archive, whose
fusion checkpoints are superseded. The archives will be moved to Zenodo upon acceptance.

## Citation

```bibtex
@misc{yeo2026tsmformer,
  title  = {Late Versus Learned Cross-Modal Fusion for Multi-Sensor Dynamic Hand Gesture
            Recognition: A Controlled Study on Two Automotive Benchmarks},
  author = {Yeo, Gwangho and Lee, Hyunjik and Lee, Haneum and Lee, Seunghyun
            and Kwon, Soonchul and Hwang, Leehwan},
  note   = {Manuscript under review at IEEE Sensors Journal},
  year   = {2026}
}
```

## Acknowledgment

The backbone design builds on GestFormer and the Temporal Shift Module (TSM). We thank the
authors of NVGesture and Briareo for the public datasets.
