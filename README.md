# TSMFormer: Late versus Learned Cross-Modal Fusion for Multi-Sensor Gesture Recognition

Code for **"Late Versus Learned Cross-Modal Fusion for Multi-Sensor Dynamic Hand Gesture
Recognition: A Controlled Study on Two Automotive Benchmarks"** (IEEE Sensors Journal, under
review).

The repository reproduces the multimodal dynamic hand gesture recognition experiments on the
**NVGesture** and **Briareo** datasets: the per-modality TSM backbones, the late-fusion
baseline, the learned cross-modal fusion variants, the complementarity analysis, and the
cross-benchmark transfer test reported in the paper.

> **Summary of findings.**
> * **Benchmark protocol** (official splits). Late fusion of per-modality TSM backbones, an
>   average of softmax posteriors, reaches 89.63% on NVGesture and 99.31% on Briareo (best
>   modality subset).
> * **Controlled protocol** (NVGesture 901/149/482, subjects 7 and 15 held out for validation,
>   identical backbones and model selection for every variant). No learned fusion variant
>   exceeds late fusion (85.68%): per-class modality weighting 84.77 ± 0.11% (5 seeds),
>   bottleneck fusion with a hybrid gate 83.89 ± 0.63% (3), LoRA backbone adaptation
>   80.00 ± 1.16% (5), dense cross-attention 76.90 ± 1.20% (3), bottleneck fusion with a
>   stronger input 73.90 ± 0.82% (5). On Briareo, per-class weighting gives 98.26 ± 0.00% (5).
>   The smallest gap (0.91 pp) is more than eight times the seed standard deviation.
> * **Complementarity.** On NVGesture an oracle over the five modalities reaches 96.89%, so
>   7.26 pp of complementarity remain that none of the trained mechanisms recovers; on
>   Briareo the oracle equals the best late-fusion subset.
> * **Cross-benchmark transfer.** Linear probes on frozen backbones show that the learned
>   representations are tied to the device and viewpoint that produced them; actively
>   illuminated modalities (depth, IR) transfer better than color.

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
 temporal.py              # main model (TSM backbone + temporal encoder)
 attention.py             # GestFormer-style encoder
 backbones/resnet_tsm.py  # TSM-ResNet-18 backbone
 fusion.py                # dense cross-attention fusion
 fusion_v4.py             # bottleneck (MBT) fusion with hybrid gate
 lora_inject.py           # LoRA adapters for backbone adaptation
 perclass_fusion.py       # per-class modality weighting
datasets/                    # NVGesture / Briareo loaders, surface normals, optical flow
hyperparameters/
 NVGestures/              # benchmark protocol and fusion configurations
 NVGestures_rev/          # controlled protocol backbones (901/149/482)
 Briareo/
splits/NVGestures/           # subject-disjoint train/validation lists (controlled protocol)
main.py                      # unimodal training / testing (--phase train|test)
cs.py                        # late fusion and modality-subset evaluation
train_cmaf.py                # dense cross-attention fusion
train_cmaf_v4.py             # bottleneck fusion
train_cmaf_v4_lora.py        # LoRA ensemble (P0)
train_perclass.py            # per-class modality weighting
run_stage1.sh                # controlled-protocol backbones
run_stage2.sh                # all fusion variants, multi-seed (NVGesture)
run_stage2_briareo.sh        # per-class weighting, multi-seed (Briareo)
collect_stage2.py            # mean / std per variant
analyze_complementarity.py   # pairwise Q statistic, oracle accuracy
extract_trajectory.py        # per-class weight trajectories
make_fig_weights.py          # Fig. 2
transfer_probe.py            # cross-benchmark linear probe
run_transfer.sh              # all transfer pairs
experiments/BL6_transfer/    # transfer results reported in the paper
requirements-lock.txt, environment.yml   # pinned environment
```

## Datasets

* **NVGesture** [Molchanov et al., CVPR 2016]: 25 classes; official split of 1,050 training
  and 482 test sequences (12 and 6 subjects). The controlled protocol further holds out
  subjects 7 and 15 (149 sequences) for validation, leaving 901 for training.
* **Briareo** [Manganaro et al., ICIAP 2019]: 12 classes; the training, validation and test
  splits distributed with the dataset (288 test sequences).

Obtain the datasets from their original sources and set the data paths in the JSON
configurations under `hyperparameters/`. Surface normals are computed from depth. Optical
flow (Farnebäck) is computed from depth on NVGesture and from color on Briareo
(`datasets/utils/`).

## Reproducing the results

Run all commands from the repository root after setting `data_path`, `csv_dir` and the
checkpoint paths in the relevant JSON configuration. `<ds>` is `NVGestures` or `Briareo` for
configuration folders and `Nvgestures` or `Briareo` for `--dataset`; `<mod>` ranges over the
five modalities. The same table appears as Table X in Appendix I of the paper.

| Result | Command |
|---|---|
| Tables II, III (unimodal) | `python main.py --hypes hyperparameters/<ds>/test_<mod>_tsm.json --phase test` |
| Tables IV, VI (late fusion, subsets) | `python cs.py --dataset <ds> --csv_dir experiments/BL2_tsm/<ds>/csv --all` |
| Table VII (complementarity) | `python analyze_complementarity.py --dataset <ds> --csv_dir <csv>` |
| Controlled-protocol backbones | `./run_stage1.sh` (or `python main.py --hypes hyperparameters/NVGestures_rev/train_<mod>_rev.json --phase train`) |
| Table V (fusion variants) | `./run_stage2.sh ; python collect_stage2.py` |
| Fig. 2 (weight trajectories) | `python extract_trajectory.py perclass ; python make_fig_weights.py` |
| Table VIII (cross-benchmark transfer) | `./run_transfer.sh ; python transfer_probe.py summary` |

## Trained checkpoints

The 65 trained backbone and fusion checkpoints behind the reported results (about 8.3 GB) are
linked from the [release page](https://github.com/ygh8888/TSMFormer/releases) of this tag,
together with a manifest (experiment, protocol, seed and supported table or figure of every
file) and SHA-256 checksums. The archive will be moved to Zenodo upon acceptance.

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
