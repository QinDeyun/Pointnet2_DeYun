"""
修改DATA_PATH
"""

from data_utils.ModelNetDataLoader_DeYun import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models/Multi_Transformer_Policy/detr_deyun/models'))

import torch.nn.functional as F

from data_utils import ModelNetDataLoader_DeYun

from models.Multi_Transformer_Policy.policy import Multi_Transformer_Policy

"""
需要配置的参数：
--model pointnet2_cls_msg 
--normal 
--log_dir pointnet2_cls_msg
"""

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet2_cls_msg_DeYun', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]') # 是否使用后三列的数据
    return parser.parse_args()

def quaternion_geodesic_loss(q_true, q_pred):
    # 输入形状: [B, 4], L2归一化四元数 (确保单位四元数)
    q_true = torch.nn.functional.normalize(q_true, p=2, dim=-1)
    q_pred = torch.nn.functional.normalize(q_pred, p=2, dim=-1)
    
    dot_product = torch.abs(torch.sum(q_true * q_pred, dim=-1))  # 取绝对值解决q和-q等价
    theta = 2 * torch.arccos(torch.clamp(dot_product, min=-1.0, max=1.0))
    return theta

def test(policy, loader, stats):
    mean_translation_loss = []
    mean_quaternion_loss =[]
    mean_loss = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        point_set, image, realsense_initial_pos, label = data

        # point_set = point_set.transpose(2, 1)
        point_set, image, realsense_initial_pos, label = point_set.cuda(), image.cuda(), realsense_initial_pos.cuda(), label.cuda()
        policy.eval()
        translation_hat, quaternion_hat = policy(point_set, image, realsense_initial_pos, None)

        translation = label[:, 0:3]
        quaternion = label[:, 3:7]
        translation_loss = F.mse_loss(translation_hat, translation)
        quaternion_loss = quaternion_geodesic_loss(quaternion_hat, quaternion).mean()
        loss = translation_loss + quaternion_loss
        
        mean_translation_loss.append(translation_loss.item())
        mean_quaternion_loss.append(quaternion_loss.item())
        mean_loss.append(loss.item())  # Append the scalar value of the distance

    mean_translation_loss = np.mean(mean_translation_loss)
    mean_quaternion_loss = np.mean(mean_quaternion_loss)
    mean_loss = np.mean(mean_loss)
    return mean_translation_loss, mean_quaternion_loss, mean_loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    #创建文件夹
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''start a new wandb run to track this script'''
    wandb.init(
        # set the wandb project where this run will be logged
        project="The_Grasp_Task",

        # track hyperparameters and run metadata
        config=args,
    )

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = './data/Dataset_DeYun/v2_object_z_rotation/'

    args.normal = False

    stats = ModelNetDataLoader_DeYun.get_norm_stats(DATA_PATH)

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, norm_stats=stats, npoint=args.num_point, split='train', uniform=True, 
                                                     normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, norm_stats=stats, npoint=args.num_point, split='test', uniform=True, 
                                                    normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model)
    # shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    policy_config = {'lr': 1e-4,
                    'hidden_dim': 512,
                    'dim_feedforward': 2048,
                    'lr_backbone': 1e-5,
                    'backbone': 'resnet18',
                    'enc_layers': 4,
                    'dec_layers': 7,
                    'nheads': 8,
                    }

    policy = Multi_Transformer_Policy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()
    start_epoch = 0

    # classifier = MODEL.get_model(normal_channel=args.normal).cuda()
    # criterion = MODEL.get_loss().cuda()
    # try:
    #     checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0


    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         classifier.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.decay_rate
    #     )
    # else:
    #     optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_mean_distance = float('inf')
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        translation_loss =[]
        rotation_loss = []
        mean_distance = []
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,
        # 但也不是绝对的，可以根据具体的需求来做。
        # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, image, realsense_initial_pos, label = data

            points = points.data.numpy()
            points = provider.random_point_dropout(points) #进行数据增强
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3]) #在数值上调大或调小，设置一个范围
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3]) #增加随机抖动，使测试结果更好
            print("points shape:", points.shape)
            points = torch.Tensor(points) 
            # target = target[:, 0] # [8, 6] -> [8]


            # points = points.transpose(2, 1)s
            points, image, realsense_initial_pos, label = points.cuda(), image.cuda(), realsense_initial_pos.cuda(), label.cuda()
            optimizer.zero_grad()

            policy.train()
            loss_dict = policy(points, image, realsense_initial_pos, label) #计算损失

            translation_loss.append(loss_dict['translation_loss'])
            rotation_loss.append(loss_dict['rotation_loss'])
            mean_distance.append(loss_dict['all_loss'])  # Append the scalar value of the distance


        ####################################################################

            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))

            print("translation_loss", loss_dict['translation_loss'])
            print("rotation_loss", loss_dict['rotation_loss'])
            print("all_loss", loss_dict['all_loss'])

            loss_dict['all_loss'].backward() #反向传播
            optimizer.step() #最好的测试结果
            global_step += 1

        train_mean_translation_loss = np.mean([dist.cpu().item() for dist in translation_loss])
        train_mean_rotation_loss = np.mean([dist.cpu().item() for dist in rotation_loss])
        train_mean_all_loss = np.mean([dist.cpu().item() for dist in mean_distance])
        log_string('Train Instance Accuracy: %f' % train_mean_all_loss)

        # log metrics to wandb
        wandb.log({"train_mean_translation_loss": train_mean_translation_loss}, step=epoch + 1)
        wandb.log({"train_mean_rotation_loss": train_mean_rotation_loss}, step=epoch + 1)
        wandb.log({"train_mean_all_loss": train_mean_all_loss}, step=epoch + 1)

        with torch.no_grad():
            test_mean_translation_loss, test_mean_quaternion_loss, test_mean_distance = test(policy.eval(), testDataLoader, stats)

            # log metrics to wandb
            wandb.log({"test_mean_translation_loss": test_mean_translation_loss}, step=epoch + 1)
            wandb.log({"test_mean_quaternion_loss": test_mean_quaternion_loss}, step=epoch + 1)
            wandb.log({"test_mean_distance": test_mean_distance}, step=epoch + 1)

            if (test_mean_distance <= best_mean_distance):
                best_mean_distance = test_mean_distance
                best_epoch = epoch + 1

            log_string('Test Mean Distance: %f'% (test_mean_distance))
            log_string('Best Mean Distance: %f'% (best_mean_distance))

            if (test_mean_distance <= best_mean_distance):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_mean_distance,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    print("best_epoch:", best_epoch)
    logger.info('End of training...')
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)
