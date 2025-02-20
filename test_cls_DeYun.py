from data_utils.ModelNetDataLoader_DeYun import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import torch.nn.functional as F

from data_utils import ModelNetDataLoader_DeYun

from models.Multi_Transformer_Policy.policy import Multi_Transformer_Policy

sys.path.append(os.path.join(ROOT_DIR, 'models/Multi_Transformer_Policy/detr_deyun/models'))



"""
配置参数：
修改:log_dir, such as:2025-02-15_01-00
"""

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='2025-02-20_15-02', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def quaternion_geodesic_loss(q_true, q_pred):
    q_true = torch.nn.functional.normalize(q_true, p=2, dim=-1)
    q_pred = torch.nn.functional.normalize(q_pred, p=2, dim=-1)

    dot_product = torch.abs(torch.sum(q_true * q_pred, dim=-1))  # 取绝对值解决q和-q等价
    theta = 2 * torch.arccos(torch.clamp(dot_product, min=0.0, max=1.0 - 1e-7))
    return theta

def test(policy, loader, stats):
    mean_translation_loss = []
    mean_quaternion_loss =[]
    mean_loss = []
    print("Stats:", stats)
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        point_set, image, realsense_initial_pos, label = data

        # points = points.transpose(2, 1)
        point_set, image, realsense_initial_pos, label = point_set.cuda(), image.cuda(), realsense_initial_pos.cuda(), label.cuda()
        policy.eval()
        translation_hat, rotation_hat = policy(point_set, image, realsense_initial_pos, None)

        translation = label[:, 0:3]
        rotation = label[:, 3:7]

        translation_loss = F.mse_loss(translation_hat, translation)
        quaternion_loss = quaternion_geodesic_loss(rotation, rotation_hat).mean()
        loss = translation_loss + quaternion_loss

        mean_translation_loss.append(translation_loss.item())
        mean_quaternion_loss.append(quaternion_loss.item())
        mean_loss.append(loss.item())  # Append the scalar value of the distance

        translation = translation * stats['label_distance_std'] + stats['label_distance_mean']
        # rotation = rotation * stats['label_quaternion_std'] + stats['label_quaternion_mean']
        translation_hat = translation_hat * stats['label_distance_std'] + stats['label_distance_mean']
        # rotation_hat = rotation_hat * stats['label_quaternion_std'] + stats['label_quaternion_mean']
        
        for i in range(label.shape[0]):
            print(f"Sample {i}:")
            print(f"  True Translation: {translation[i].cpu().numpy()}")
            print(f"  Predicted Translation: {translation_hat[i].cpu().numpy()}")
            print(f"  True Rotation: {rotation[i].cpu().numpy()}")
            print(f"  Predicted Rotation: {rotation_hat[i].cpu().numpy()}")
            translation_distance = torch.sum(torch.abs(translation[i] - translation_hat[i]))
            print(f"  Translation Distance: {translation_distance}")
            rotation_distance = torch.sum(torch.abs(rotation[i] - rotation_hat[i]))
            print(f"  Rotation Distance: {rotation_distance}")

        translation_loss = F.mse_loss(translation_hat, translation)
        quaternion_loss = quaternion_geodesic_loss(rotation, rotation_hat).mean()
        loss = translation_loss + quaternion_loss

    mean_translation_loss = np.mean(mean_translation_loss)
    mean_quaternion_loss = np.mean(mean_quaternion_loss)
    mean_loss = np.mean(mean_loss)
    print(f"Mean Translation Loss: {mean_translation_loss}")
    print(f"Mean quaternion Loss: {mean_quaternion_loss}")
    print(f"mean_loss: {mean_loss}")
    return mean_loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = './data/Dataset_DeYun/v2_object_z_rotation'
    stats = ModelNetDataLoader_DeYun.get_norm_stats(DATA_PATH)

    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, norm_stats=stats, npoint=args.num_point, split='test', uniform=True, normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)

    # classifier = MODEL.get_model(normal_channel=args.normal).cuda()

    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])

    policy_config = {'lr': 1e-4,
                    'hidden_dim': 512,
                    'dim_feedforward': 2048,
                    'lr_backbone': 1e-5,
                    'backbone': 'resnet18',
                    'enc_layers': 4,
                    'dec_layers': 7,
                    'nheads': 8,
                    }
    ckpt_path = str(experiment_dir) + '/checkpoints/best_model.pth'
    policy = Multi_Transformer_Policy(policy_config)
    
    state = torch.load(ckpt_path)
    loading_status = policy.load_state_dict(state['model_state_dict'])
    print('best_epoch', state['epoch'])
    
    print(f'Loaded: {ckpt_path}')
    print(loading_status)
    policy.cuda()
    policy.eval()


    with torch.no_grad():
        mean_distance = test(policy.eval(), testDataLoader, stats)
        log_string('Mean Distance: %f' % (mean_distance))



if __name__ == '__main__':
    args = parse_args()
    main(args)
