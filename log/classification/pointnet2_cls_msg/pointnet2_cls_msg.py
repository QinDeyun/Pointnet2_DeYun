import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # 512 = points sampled in farthest point sampling，找中心点
        # [0.1,0.2,0.4] = search radius in local region，画圈
        # [16,32,128] = how many points in each local region，每个圈里的点的个数
        # [[32,32,64], [64,64,128], [64,96,128]] = output size for MLP on each point  # 提取特征
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]]) #多半径的特征提取。[32, 32, 64], [64, 64, 128], [64, 96, 128]为卷积特征个数。
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]]) #在512个点中选择128个点，320为通道数。
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True) #把所有点当做一个组，组里一共有128个点，每个点的特征数为643个
        # fc1 input:1024
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        # fc2 input:512
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        # fc3 input:256
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):  # xyz是位置信息，原始数据给我们的
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        print(xyz.shape) #3个位置特征
        print(norm.shape) #3个法向量，即额外的特征
        # l1_points作为sa1的特征输出320，xx_points表示特征输出
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_points作为sa2的特征输出640
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_points作为sa3的特征输出
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        #3个全连接层
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1) #计算对数概率

        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


