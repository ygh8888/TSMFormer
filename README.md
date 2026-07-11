# TSMFormer: Temporal-Shift Transformer Backbones for Dynamic Hand Gesture Recognition

Official implementation of **"Temporal-Shift Transformer Backbones for Dynamic Hand
Gesture Recognition: A Quantitative Analysis of Late versus Cross-Modal Fusion under
Small-Scale Multimodal Data."**

This repository reproduces the multimodal dynamic hand gesture recognition experiments
on the **NVGesture** and **Briareo** datasets, including the TSM-based backbone, the
late-fusion baseline, and the learned cross-modal fusion variants analyzed in the paper.

> **Summary of findings.** A simple late fusion of per-modality TSM backbones forms a
> strong baseline (89.63% on NVGesture, 99.31% on Briareo). Across a range of learned
> cross-modal fusion variants (dense cross-attention, bottleneck transformer, LoRA
> backbone adaptation, per-class logit weighting), none surpasses this simple average
> on either dataset, which we analyze quantitatively.

## Architecture

Each modality is processed by an **independent** pipeline:

```
Input (T frames) -> TSM-ResNet-18 backbone -> GestFormer-style temporal encoder
                 -> FC classifier -> per-modality logits z_m
```

The temporal encoder (6 layers) follows the GestFormer design and, per layer, applies:
1. a wavelet-based SSL module (2D DWT -> per-subband depthwise conv -> inverse DWT),
2. a multi-scale pooling token mixer (average pooling at window sizes 3/5/7), and
3. a gated depthwise-convolution feed-forward block,

with sinusoidal positional encoding. The modalities are combined only at a final
**late-fusion** step (a simple average of softmax probabilities, no learned parameters).

## Repository structure

```
src_tsmformer/
├── models/
│   ├── temporal.py            # main model (TSM backbone + temporal encoder)
│   ├── attention.py           # GestFormer-style encoder (SSL + pooling mixer + gated FFN)
│   ├── fusion.py / fusion_v4.py
│   ├── lora_inject.py         # LoRA adapters for backbone adaptation
│   ├── perclass_fusion.py     # per-class logit weighting
│   ├── model_utilizer.py
│   └── backbones/
│       ├── resnet_tsm.py      # TSM-ResNet-18 (core backbone)
│       └── resnet.py
├── datasets/                  # NVGesture / Briareo loaders + preprocessing
├── utils/
├── hyperparameters/           # JSON configs (NVGestures/, Briareo/)
├── train.py                   # unimodal training
├── test.py / cs.py            # evaluation (modality-subset)
├── train_cmaf_v4_lora.py      # learned fusion (LoRA), seed-parameterized
├── train_perclass.py          # per-class logit weighting fusion
├── measure_multiseed.py       # reliable multi-seed test-accuracy measurement
└── main.py
```

> Checkpoints, shell scripts, and dataset files are **not** included.

## Datasets

- **NVGesture** [Molchanov et al., CVPR 2016] - 25 classes, ~1,050 train / 482 test.
- **Briareo** [Manganaro et al., ICIAP 2019] - 12 classes, 288 test.

Obtain the datasets from their original sources and set the data paths in the JSON
configs under `hyperparameters/`. Surface normals and optical flow are derived from
depth and color streams (see `datasets/utils/`).

## Usage

> Set `data_path`, `csv_dir`, and checkpoint paths in the relevant JSON config first.

1. Unimodal training (per modality):
```bash
python train.py --hypes hyperparameters/NVGestures/train_color_tsm.json
```

2. Late-fusion / modality-subset evaluation:
```bash
python cs.py --hypes hyperparameters/NVGestures/test_tsm.json --modalities color depth normal optflow
```

3. Learned cross-modal fusion (LoRA ensemble):
```bash
python train_cmaf_v4_lora.py --hypes hyperparameters/NVGestures/p0_seed1994.json --seed 1994
```

4. Per-class weighting fusion:
```bash
python train_perclass.py --hypes hyperparameters/NVGestures/train_perclass.json
```

5. Reliable multi-seed measurement (after training the seeds above):
```bash
python measure_multiseed.py
```

## Citation

```bibtex
@article{yeo2026tsmformer,
  title   = {Temporal-Shift Transformer Backbones for Dynamic Hand Gesture
             Recognition: A Quantitative Analysis of Late versus Cross-Modal
             Fusion under Small-Scale Multimodal Data},
  author  = {Yeo, Gwangho and Lee, Hyunjik and Lee, Haneum and Lee, Seunghyun
             and Kwon, Soonchul and Hwang, Leehwan},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Acknowledgment

The backbone design builds on the GestFormer family of pooling-based transformers and
the Temporal Shift Module (TSM). We thank the authors of NVGesture and Briareo for the
public datasets.
