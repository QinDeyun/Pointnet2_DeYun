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



"""
配置参数：
--normal 
--log_dir pointnet2_cls_msg
"""

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_normal', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def test(model, loader, stats):
    mean_distance = []
    print("Stats:", stats)
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data

        # Standardize the target data
        target[0:3] = (target[0:3] - stats['distance_mean']) / stats['distance_std']
        target[3:6] = (target[3:6] - stats['angle_mean']) / stats['angle_std']

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)

        distance = F.mse_loss(pred, target)  # 均方误差

        print("Prediction111111111111:", pred)
        print("Target111111111111:", target)

        target[:, 0:3] = target[:, 0:3] * stats['distance_std'] + stats['distance_mean']
        target[:, 3:6] = target[:, 3:6] * stats['angle_std'] + stats['angle_mean']
        pred[:, 0:3] = pred[:, 0:3] * stats['distance_std'] + stats['distance_mean']
        pred[:, 3:6] = pred[:, 3:6] * stats['angle_std'] + stats['angle_mean']
        
        for i in range(target.shape[0]):
            print(f"Sample {i}: Prediction: {pred[i]}, Target: {target[i]}")

        mean_distance.append(distance.item())  # Append the scalar value of the distance

    mean_distance = np.mean(mean_distance)
    return mean_distance


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
    DATA_PATH = './data/Dataset_DeYun/'
    stats = ModelNetDataLoader_DeYun.get_norm_stats(DATA_PATH)

    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, norm_stats=stats, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(normal_channel=args.normal).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        mean_distance = test(classifier.eval(), testDataLoader, stats)
        log_string('Mean Distance: %f' % (mean_distance))



if __name__ == '__main__':
    args = parse_args()
    main(args)
