"""
Reliable multi-seed test-accuracy measurement for the LoRA-ensemble (P0) fusion.

The training script's in-loop FINAL TEST can report incorrect accuracy due to
DataParallel replica caching when LoRA parameters are restored in place. This
standalone script rebuilds a fresh trainer per seed and re-runs the test loop,
which gives the reliable numbers reported in the paper (Table: multi-seed).

Usage:
    python measure_multiseed.py
"""
import torch
import statistics as st
from train_cmaf_v4_lora import CMAFTrainer

SEEDS = [1994, 2024, 777]
CFG = "hyperparameters/NVGestures/p0_seed{seed}.json"
CKPT = ("experiments/BL4_p0_multiseed/seed{seed}/checkpoints/"
        "NVGestures/best_p0_seed{seed}_nvgestures.pth")


def measure_seed(seed, device):
    cfg = CFG.format(seed=seed)
    ckpt_path = CKPT.format(seed=seed)
    trainer = CMAFTrainer(cfg, device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.cmaf.load_state_dict(ck["state_dict"])
    # restore per-backbone LoRA parameters in place
    for model, lsd in zip(trainer.modal_models, ck["lora_state"]):
        model.eval()
        pdict = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in lsd.items():
                if n in pdict:
                    pdict[n].copy_(v.to(device))
    trainer.cmaf.eval()
    for m in trainer.modal_models:
        m.eval()
    acc, _ = trainer._run_epoch(trainer.test_loaders, "test")
    best_epoch = ck.get("epoch", "?")
    del trainer
    torch.cuda.empty_cache()
    return acc * 100, best_epoch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    for seed in SEEDS:
        acc, best_epoch = measure_seed(seed, device)
        results[seed] = acc
        print(f"  seed{seed}: best_epoch={best_epoch}  test={acc:.2f}%")
    vals = list(results.values())
    mean = st.mean(vals)
    std = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"\n=== P0 multi-seed === mean={mean:.2f}%  std={std:.2f}%")
    print("vs BL2 late fusion: 89.63%")


if __name__ == "__main__":
    main()
