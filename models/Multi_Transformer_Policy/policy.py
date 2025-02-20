import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Multi_Transformer_Policy.detr_deyun.main import build_ACT_model_and_optimizer
import IPython
import test

import torch

e = IPython.embed

def quaternion_geodesic_loss(q_true, q_pred):
    q_true = torch.nn.functional.normalize(q_true, p=2, dim=-1)
    q_pred = torch.nn.functional.normalize(q_pred, p=2, dim=-1)

    dot_product = torch.abs(torch.sum(q_true * q_pred, dim=-1))  # 取绝对值解决q和-q等价
    theta = 2 * torch.arccos(torch.clamp(dot_product,  min=0.0, max=1.0 - 1e-7))
    return theta

class Multi_Transformer_Policy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer

    def __call__(self, point_set, image, realsense_initial_pos, label):
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])  # 进行归一化，有助于提升稳定性和收敛速度。其中的值是根据imagenet数据集得到的。
        # image = normalize(image)
        print(f"point_set shape: {point_set.shape}")
        print(f"image shape: {image.shape}")
        print(f"realsense_initial_pos shape: {realsense_initial_pos.shape}")
        if label is not None:  # training time
            print(f"label shape: {label.shape}")
            translation_hat, rotation_hat = self.model(point_set, image, realsense_initial_pos)

            translation = label[:, 0:3]
            rotation = label[:, 3:7]

            translation_loss = F.mse_loss(translation_hat, translation)
            rotation_loss = quaternion_geodesic_loss(rotation, rotation_hat).mean()
            loss = translation_loss + rotation_loss

            loss_dict = dict()
            loss_dict['translation_loss'] = translation_loss
            loss_dict['rotation_loss'] = rotation_loss
            loss_dict['all_loss'] = loss
            return loss_dict
        else:  # inference time
            translation_hat, rotation_hat = self.model(point_set, image, realsense_initial_pos)
            return translation_hat, rotation_hat

    def configure_optimizers(self):
        return self.optimizer


