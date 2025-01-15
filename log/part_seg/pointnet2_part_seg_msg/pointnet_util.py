import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):  #用最远点采样方法得到比较均匀的点
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]  #返回哪个点是中心点
    """
    device = xyz.device
    B, N, C = xyz.shape  #8,1024,3
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  #初始化8*512的矩阵，一共有8个batch，每个batch里有512个点
    distance = torch.ones(B, N).to(device) * 1e10 #定义一个8*1024的距离矩阵，里面存储的是除了中心点的每个点距离当前所有已采样点的最小距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#batch里每个样本随机初始化一个最远点的索引（第一个点是随机选择的）
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest #第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)#得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)#计算当前采样点与其他点的距离
        mask = dist < distance#选择距离最近的来更新距离（更新维护这个表），小于当前距离就是TRUE，否则是FALSE
        distance[mask] = dist[mask]#更新距离矩阵
        farthest = torch.max(distance, -1)[1]#重新计算得到最远点索引（在更新后的距离矩阵中选择距离最大的那个点）
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region #圆圈里的点的个数
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) #让每一个圆圈里的点的个数一致
    sqrdists = square_distance(new_xyz, xyz)#得到B*N*M（中心点）的矩阵 （就是N个点中每一个和M中每一个的欧氏距离）  N=1024
    group_idx[sqrdists > radius ** 2] = N #找到距离大于给定半径的设置成一个N值（1024）索引，值为1024表示这个点不在半径当中
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#做升序排序，只取规定的圆圈里的个数就行了。后面的都是大的值（1024）  可能有很多点到中心点的距离都小于半径，而我们只需要16个，所以排序一下，取前16个离中心点最近的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])#如果半径内的点没那么多，则复制离中心点最近的那个点即第一个点来代替值为1024的点
    mask = group_idx == N #判断是否有值=1024，返回TRUE或FALSE（若点在圆圈里则值为FALSE，否则值为TRUE）
    group_idx[mask] = group_first[mask]  #如果有值为1024，则把第一个值赋值给距离=1024的点
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        print(xyz.shape)  #第三次：[8,128,3]
        if points is not None:
            points = points.permute(0, 2, 1)
        print(points.shape) #第三次：[8,128,640],640个特征
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]，D是3个位置特征
        print(new_points.shape) #[8,1,128,643]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        print(new_points.shape) #[8,643,128,1]
        for i, conv in enumerate(self.mlp_convs): #提取特征
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        print(new_points.shape) #[8,1024,128,1]
        new_points = torch.max(new_points, 2)[0]
        print(new_points.shape) #[8,1024,1]
        new_xyz = new_xyz.permute(0, 2, 1)
        print(new_xyz.shape) #[8,3,1]
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]，N=1024
            points: input points data, [B, D, N]，原始的特征信息，3个法向量，D=3,N=1024
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]，不同的半径提取不同的特征，最后将所有特征连接起来，个数就不是3个了
        """
        xyz = xyz.permute(0, 2, 1)  #就是坐标点位置特征
        print(xyz.shape) #第一次：[8,1024，3]，第二次：[8,512，3]
        if points is not None:
            points = points.permute(0, 2, 1)  #就是额外提取的特征，第一次的时候就是那个法向量特征
        print(points.shape) #第二次：[8,512，320]
        B, N, C = xyz.shape
        S = self.npoint  #S为我们选择的中心点的个数，为了使得选择的点均匀分布，用最远点采样的方法
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  #最远点采样方法得到的是点的索引值，我们想得到点的实际值，通过index_points()就能得到采样后的点的实际信息
        print(new_xyz.shape) #第一次：[8,512,3])，第二次：[8,128，3]
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i] #圆圈里圈出来的点的个数
            group_idx = query_ball_point(radius, K, xyz, new_xyz) #返回的是索引  new_xyz是中心点，xyz是原始点  最后得到512个组
            grouped_xyz = index_points(xyz, group_idx) #通过索引得到各个组中实际点
            grouped_xyz -= new_xyz.view(B, S, 1, C) #去均值操作  每个组中的点减去中心点的值 （new_xyz相当于簇的中心点）
            if points is not None:
                grouped_points = index_points(points, group_idx) #法向量特征
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) #把位置特征和法向量特征拼在一起
                print(grouped_points.shape)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # 维度转换操作，将[B,S,K,D]转换成[B, D, K, S]
            print(grouped_points.shape)
            for j in range(len(self.conv_blocks[i])): #卷积核大小1*1，步长=1，通道数6->32->64,进行3次卷积之后，每个点特征为64个
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            print(grouped_points.shape)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S] 就是pointnet里的maxpool操作，即在每一个特征维度上，从一个组中选一个值最大的出来，作为这个维度上的特征值。
            print(new_points.shape) #[8,64,512]（第一次卷积）
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1) #r=0.1,channel=64;r=0.2,channel=128;r=0.4,channel=128,把所有的channel都连接起来。
        print(new_points_concat.shape) #第一次：[8,320,512]，第二次：[8,640,128]
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        print(xyz1.shape)
        print(xyz2.shape)

        points2 = points2.permute(0, 2, 1)
        print(points2.shape)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
            print(interpolated_points.shape)
        else:
            dists = square_distance(xyz1, xyz2)
            print(dists.shape)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            print(weight.shape)
            print(index_points(points2, idx).shape)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            print(interpolated_points.shape)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        print(new_points.shape)
        new_points = new_points.permute(0, 2, 1)
        print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        print(new_points.shape)
        return new_points

