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
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import torch.nn.functional as F

from data_utils import ModelNetDataLoader_DeYun


"""
需要配置的参数：
--model pointnet2_cls_msg 
--normal 
--log_dir pointnet2_cls_msg
"""

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training [default: 24]')
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

# def test(model, loader, num_class=40):
#     mean_correct = []
#     class_acc = np.zeros((num_class,3))
#     for j, data in tqdm(enumerate(loader), total=len(loader)):
#         points, target = data
#         target = target[:, 0]
#         points = points.transpose(2, 1)
#         points, target = points.cuda(), target.cuda()
#         classifier = model.eval()
#         pred, _ = classifier(points)
#         pred_choice = pred.data.max(1)[1]
#         for cat in np.unique(target.cpu()):
#             classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
#             class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
#             class_acc[cat,1]+=1
#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item()/float(points.size()[0]))
#     class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
#     class_acc = np.mean(class_acc[:,2])
#     instance_acc = np.mean(mean_correct)
#     return instance_acc, class_acc

def test(model, loader, stats):
    mean_distance = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data

        # Standardize the target data
        target[:, 0:3] = (target[:, 0:3] - stats['distance_mean']) / stats['distance_std']
        target[:, 3:6] = (target[:, 3:6] - stats['angle_mean']) / stats['angle_std']

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)

        distance = F.mse_loss(pred, target)  # 均方误差

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
    DATA_PATH = './data/Dataset_DeYun/'

    args.normal = False

    stats = ModelNetDataLoader_DeYun.get_norm_stats(DATA_PATH)

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, norm_stats=stats, npoint=args.num_point, split='train',
                                                     normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, norm_stats=stats, npoint=args.num_point, split='test',
                                                    normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_mean_distance = float('inf')
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        mean_distance = []
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,
        # 但也不是绝对的，可以根据具体的需求来做。
        # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data

            # Standardize the target data
            target[:, 0:3] = (target[:, 0:3] - stats['distance_mean']) / stats['distance_std']
            target[:, 3:6] = (target[:, 3:6] - stats['angle_mean']) / stats['angle_std']

            points = points.data.numpy()
            points = provider.random_point_dropout(points) #进行数据增强
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3]) #在数值上调大或调小，设置一个范围
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3]) #增加随机抖动，使测试结果更好
            print("points shape:", points.shape)
            points = torch.Tensor(points) 
            # target = target[:, 0] # [8, 6] -> [8]


            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target, trans_feat).float() #计算损失
            mean_distance.append(loss.item())  # Append the scalar value of the distance

        ####################################################################

            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))

            print("loss shape", loss.shape)
            print("loss:", loss) # loss: tensor(345.8652, device='cuda:0', grad_fn=<MseLossBackward>)
            loss.backward() #反向传播
            optimizer.step() #最好的测试结果
            global_step += 1

        train_mean_distance = np.mean(mean_distance)
        log_string('Train Instance Accuracy: %f' % train_mean_distance)

        # log metrics to wandb
        wandb.log({"train_mean_distance": train_mean_distance}, step=epoch + 1)

        with torch.no_grad():
            test_mean_distance = test(classifier.eval(), testDataLoader, stats)

            # log metrics to wandb
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
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)
