import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

# Import Datasets
from datasets.Briareo import Briareo
from datasets.NVGestures import NVGesture
from models.model_utilizer import ModuleUtilizer

# Import Model
from models.temporal import GestureTransoformer

# Import Utils
from tqdm import tqdm
from utils.average_meter import AverageMeter
import time
try:
    from torchstat import stat
    from torchinfo import summary   ## 원본 : import torchsummary -> DWT 모듈과 호환성 문제 발생
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    stat = torchsummary = FlopCountAnalysis = None


# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class GestureTest(object):
    """Gesture Recognition Test class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        train_loader (torch.utils.data.DataLoader): Train data loader variable
        val_loader (torch.utils.data.DataLoader): Val data loader variable
        test_loader (torch.utils.data.DataLoader): Test data loader variable
        net (torch.nn.Module): Network used for the current procedure
        lr (int): Learning rate value
        optimizer (torch.nn.optim.optimizer): Optimizer for training procedure
        iters (int): Starting iteration number, not zero if resuming training
        epoch (int): Starting epoch number, not zero if resuming training
        scheduler (torch.optim.lr_scheduler): Scheduler to utilize during training

    """

    def __init__(self, configer):
        self.configer = configer

        self.data_path = configer.get("data", "data_path")      #: str: Path to data directory

        # Train val and test accuracy
        self.accuracy = AverageMeter()

        # DataLoaders
        self.data_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None

        # Training procedure
        self.transforms = None

        # Other useful data
        self.backbone = self.configer.get("network", "backbone")     #: str: Backbone type
        self.in_planes = None                                       #: int: Input channels
        self.clip_length = self.configer.get("data", "n_frames")    #: int: Number of frames per sequence
        self.n_classes = self.configer.get("data", "n_classes")     #: int: Total number of classes for dataset
        self.data_type = self.configer.get("data", "type")          #: str: Type of data (rgb, depth, ir, leapmotion)
        self.dataset = self.configer.get("dataset").lower()         #: str: Type of dataset
        self.optical_flow = self.configer.get("data", "optical_flow")
        if self.optical_flow is None:
            self.optical_flow = True

    def init_model(self):
        """Initialize model and other data for procedure"""

        if self.optical_flow is True:
            self.in_planes = 2
        elif self.data_type in ["depth", "ir"]:
            self.in_planes = 1
        else:
            self.in_planes = 3

        # Selecting correct model and normalization variable based on type variable
        self.net = GestureTransoformer(self.backbone, self.in_planes, self.n_classes,
                                       pretrained=self.configer.get("network", "pretrained"),
                                       n_head=self.configer.get("network", "n_head"),
                                       dropout_backbone=self.configer.get("network", "dropout2d"),
                                       dropout_transformer=self.configer.get("network", "dropout1d"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module"),
                                       use_tsm=self.configer.get("network", "use_tsm"),
                                       n_frames=self.configer.get("data", "n_frames"),
                                       use_mspe=self.configer.get("network", "use_mspe"),
                                       modality_id=self.configer.get("network", "modality_id")
                                       )

        self.net, _, _, _ = self.model_utility.load_net(self.net)

        # Selecting Dataset and DataLoader
        if self.dataset == "briareo":
            Dataset = Briareo
            self.transforms = iaa.CenterCropToFixedSize(200, 200)
        elif self.dataset == "nvgestures":
            Dataset = NVGesture
            self.transforms = iaa.CenterCropToFixedSize(256, 192)
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        # Setting Dataloaders
        self.data_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="test", data_type=self.data_type,
                    transforms=self.transforms, n_frames=self.clip_length,
                    optical_flow=self.optical_flow),
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """Testing function."""
        self.net.eval()
        c = 0
        tot = 0
        sm = torch.nn.Softmax(dim=1)
        prob_list = []
        gt_list = []

        # CSV 저장 경로 설정
        csv_dir = self.configer.get('data', 'csv_dir') if self.configer.get('data', 'csv_dir') else os.path.join('csv', self.dataset.capitalize())
        os.makedirs(csv_dir, exist_ok=True)
        # optical flow 여부를 파일명에 반영
        modality_name = self.data_type + ('_optflow' if self.optical_flow else '')
        csv_path = os.path.join(csv_dir, f'{modality_name}.csv')
        gt_path  = os.path.join(csv_dir, 'original.csv')

        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.data_loader, desc="Test")):
                """
                input, gt
                """
                inputs = data_tuple[0].to(self.device)
                gt = data_tuple[1].to(self.device)
                if i == 0:
                    flops = FlopCountAnalysis(self.net, inputs)
                    print(f"GFLOPs: {flops.total()/1e9:.2f}")
                    summary(self.net, input_size=inputs[0].unsqueeze(0).shape)

                start_time = time.time()
                output = self.net(inputs)
                end_time = time.time()

                prob = sm(output)
                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach().squeeze(dim=1)

                prob_list.append(prob.cpu().numpy()[0])
                gt_list.append(correct.cpu().numpy()[0])

                if predicted == correct:
                    c += 1
                tot += 1

        # CSV 저장
        pd.DataFrame(prob_list).to_csv(csv_path, header=False, index=False)
        pd.DataFrame(gt_list).to_csv(gt_path, header=False, index=False)
        print(f"CSV 저장 완료: {csv_path}")

        accuracy = c / tot
        inference_time = end_time - start_time
        print("Inference time:", inference_time, "seconds")
        print("Accuracy: {}".format(accuracy))

    def test(self):
        print("=" * 50)
        print(f"  Dataset  : {self.dataset.upper()}")
        print(f"  Modality : {self.data_type}" + (" (optical flow)" if self.optical_flow else ""))
        print("=" * 50)
        self.__test()


    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()