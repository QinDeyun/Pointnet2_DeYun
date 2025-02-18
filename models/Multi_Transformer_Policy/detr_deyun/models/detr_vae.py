# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from Multi_Transformer_Policy.detr_deyun.models.backbone import build_backbone
from Multi_Transformer_Policy.detr_deyun.models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
import unittest

import importlib


e = IPython.embed

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):

    def __init__(self, backbones, encoder, MLP, Pointnet2):
        super().__init__()
        self.encoder = encoder
        self.MLP = MLP
        self.Pointnet2 = Pointnet2
        hidden_dim = 512
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.target_pos_linear = nn.Linear(2, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.latent_proj = nn.Linear(hidden_dim, 6)  # 输出平移和旋转
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+920+1, hidden_dim))  # [CLS], qpos, a_seq ， 不会更新
        self.encoder_pointcloud_proj = nn.Linear(256, hidden_dim) # project pointcloud to embedding

    def forward(self, pointcloud, photo, camera_pos):
        """
        pointcloud = torch.randn(2, 1024, 3).to(device)
        photo = torch.randn(2, 720, 1280, 3).to(device)
        camera_pos = torch.randn(2, 6).to(device)
        """
        bs, _, _ = pointcloud.shape

        # cls
        cls_embed = self.cls_embed.weight  # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
        print('cls_embed:', cls_embed.shape)

        # pointcloud features
        pointcloud = pointcloud.permute(0, 2, 1)
        pointcloud_embed, _ = self.Pointnet2(pointcloud) # (bs, 256)
        pointcloud_embed = self.encoder_pointcloud_proj(pointcloud_embed)  # (bs, hidden_dim)
        pointcloud_embed = torch.unsqueeze(pointcloud_embed, axis=1)  # (bs, 1, hidden_dim)
        print('pointcloud_embed:', pointcloud_embed.shape)

        # photo features Image observation features and position embeddings
        photo = photo.permute(0, 3, 1, 2) # (bs, 720, 1280, 3) -> (bs, 3, 720, 1280)
        features, pos = self.backbones[0](photo)  # HARDCODED
        src = self.input_proj(features[0])  # take the last layer feature and project
        pos = pos[0]
        print('src:', src.shape)
        print('pos:', pos.shape)

        # camera_pos features
        '''修改输入维度'''
        camera_pos_embed = self.MLP(camera_pos) # (bs, 512)
        camera_pos_embed = torch.unsqueeze(camera_pos_embed, axis=1)  # (bs, 1, hidden_dim)
        print('camera_pos_embed:', camera_pos_embed.shape)

        # concat all features
        src = src.flatten(2).permute(0, 2, 1) # bsxCxHxW -> bsxCxHW -> bsx(HxW)xC
        encoder_input = torch.cat([cls_embed, pointcloud_embed, src, camera_pos_embed],
                                    axis=1)  # (bs, ?, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)

        # do not mask cls token
        is_pad = torch.zeros((bs, 1 + 1 + 920 + 1), dtype=torch.bool, device=pointcloud.device)  # False: not a padding
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim) ，换位
        # query model
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
        print('encoder_output:', encoder_output.shape) # torch.Size([67, 2, 512])

        encoder_output = encoder_output[0]  # take cls output only
        print('encoder_output:', encoder_output.shape) # torch.Size([2, 512])

        latent_info = self.latent_proj(encoder_output)
        translation = latent_info[:, :3]
        rotation = latent_info[:, 3:]

        return translation, rotation


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)

        # 添加形状调整层以确保残差连接
        if in_features != out_features:
            self.adjust_layer = nn.Linear(in_features, out_features)
        else:
            self.adjust_layer = None

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # 如果输入和输出形状不匹配，则进行调整
        if self.adjust_layer:
            residual = self.adjust_layer(residual)

        out += residual  # 残差连接
        out = self.relu(out)
        return out
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock(6, 8),
            ResidualBlock(8, 32),
            ResidualBlock(32, 128),
            ResidualBlock(128, 512)
        )

    def forward(self, x):
        return self.layers(x)

def build_encoder(args):
    d_model = args.hidden_dim  # 512
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    encoder = build_encoder(args)

    mlp = MLP()

    MODEL = importlib.import_module('pointnet2_cls_msg_DeYun_1')
    Pointnet2 = MODEL.get_model(normal_channel=False).cuda()

    # vae全部
    model = DETRVAE(
        backbones,
        encoder,
        mlp,
        Pointnet2,
    ).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model

# def build_cnnmlp(args):
#     state_dim = 14 # TODO hardcode
#
#     # From state
#     # backbone = None # from state for now, no need for conv nets
#     # From image
#     backbones = []
#     for _ in args.camera_names:
#         backbone = build_backbone(args)
#         backbones.append(backbone)
#
#     model = CNNMLP(
#         backbones,
#         state_dim=state_dim,
#         camera_names=args.camera_names,
#     )
#
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("number of parameters: %.2fM" % (n_parameters/1e6,))
#
#     return model
class TestDETRVAE(unittest.TestCase):
    def setUp(self):
        # 设置测试所需的参数
        class Args:
            hidden_dim = 512
            dropout = 0.1
            nheads = 8
            dim_feedforward = 2048
            enc_layers = 4
            pre_norm = False
            num_queries = 100
            camera_names = ['camera1', 'camera2']
            lr = 1e-4
            lr_backbone = 1e-5
            batch_size = 2
            weight_decay = 1e-4
            epochs = 300
            lr_drop = 200
            clip_max_norm = 0.1
            backbone = 'resnet18'
            dilation = False
            position_embedding = 'sine'
            camera_names = ['camera1', 'camera2']
            enc_layers = 4
            dec_layers = 6
            dim_feedforward = 2048
            hidden_dim = 512
            dropout = 0.1
            nheads = 8
            num_queries = 100
            pre_norm = False
            masks = False
            eval = False
            onscreen_render = False
            ckpt_dir = '/path/to/checkpoint'
            policy_class = 'PolicyClass'
            task_name = 'TaskName'
            seed = 42
            num_epochs = 100
            kl_weight = 1
            chunk_size = 64
            temporal_agg = False

        self.args = Args()
        self.model = build(self.args)

    def test_forward(self):
        # 创建虚拟输入数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pointcloud = torch.randn(2, 1024, 3).to(device)
        photo = torch.randn(2, 720, 1280, 3).to(device)
        camera_pos = torch.randn(2, 6).to(device)

        # 执行前向传播
        translation, rotation = self.model(pointcloud, photo, camera_pos)

        # 检查输出形状
        self.assertEqual(translation.shape, torch.Size([2, 3]))
        self.assertEqual(rotation.shape, torch.Size([2, 3]))

if __name__ == '__main__':
    unittest.main()
