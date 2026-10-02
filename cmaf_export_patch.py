import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def export_csv(self, csv_dir: str, csv_name: str = 'cmaf'):
    self.cmaf.eval()
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'{csv_name}.csv')
    gt_check_path = os.path.join(csv_dir, 'original.csv')

    sm = torch.nn.Softmax(dim=1)
    prob_list = []
    gt_list = []
    correct = 0
    total = 0

    n_batches = len(self.test_loaders[0])
    with torch.no_grad():
        for batch_list in tqdm(zip(*self.test_loaders), total=n_batches,
                               desc=f"Export {csv_name}"):
            labels = batch_list[0][1].to(self.device)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)

            features = self._extract_features(batch_list)
            logits = self.cmaf(features)
            prob = sm(logits)

            prob_list.append(prob.cpu().numpy()[0])
            gt_list.append(int(labels.cpu().numpy()[0]))

            if logits.argmax(dim=1).item() == int(labels.item()):
                correct += 1
            total += 1

    pd.DataFrame(prob_list).to_csv(csv_path, header=False, index=False)
    print(f"CMAF CSV 저장 완료: {csv_path}  ({total} rows)")
    print(f"CMAF test accuracy (검증): {correct/total*100:.2f}%")

    if os.path.exists(gt_check_path):
        gt_saved = pd.read_csv(gt_check_path, header=None).iloc[:, 0].tolist()
        if len(gt_saved) == len(gt_list):
            mismatch = sum(1 for a, b in zip(gt_saved, gt_list) if int(a) != int(b))
            if mismatch == 0:
                print(f"[OK] GT 순서가 original.csv와 완전히 일치 ({len(gt_list)} rows)")
            else:
                print(f"[경고] GT 불일치 {mismatch}건 — 모달리티 데이터 순서 확인 필요!")
        else:
            print(f"[경고] 행 개수 불일치: original.csv={len(gt_saved)}, cmaf={len(gt_list)}")
    return correct / total
