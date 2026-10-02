import torch
from train_cmaf import CMAFTrainer
from cmaf_export_patch import export_csv

CMAFTrainer.export_csv = export_csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainer = CMAFTrainer('hyperparameters/NVGestures/train_cmaf.json', device)
ckpt = torch.load(
    'experiments/BL4_cmaf/NVGestures/checkpoints/NVGestures/best_cmaf_nvgestures.pth',
    map_location=device, weights_only=False)
trainer.cmaf.load_state_dict(ckpt['state_dict'])
trainer.export_csv(csv_dir='csv/Nvgestures', csv_name='cmaf')
