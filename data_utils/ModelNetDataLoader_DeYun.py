import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


# 归一化处理：将点云数据的中心移动到原点，并将其缩放到单位球体内。
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset): #指定好到哪里读数据
    def __init__(self, root, norm_stats, npoint=1024, split='train', uniform=False, normal_channel=False, cache_size=15000): # 是否采用最远点采样，是否包含法向量通道
        self.root = root
        self.npoints = npoint
        self.split = split
        self.uniform = uniform
        self.normal_channel = normal_channel
        self.norm_stats = norm_stats

        self.pointcloud_files = []
        self.label_files = []
        
        # Load file names
        for file_name in os.listdir(root):
            if file_name.startswith('pointcloud_') and file_name.endswith('.npy'):
                self.pointcloud_files.append(file_name)
                label_file = file_name.replace('pointcloud_', 'label_').replace('.npy', '.txt')
                self.label_files.append(label_file)
        
        # Sort files to ensure matching order
        self.pointcloud_files.sort()
        self.label_files.sort()

        assert len(self.pointcloud_files) == len(self.label_files), "Mismatch between pointcloud and label files"
        assert (self.split == 'train' or self.split == 'test')

        # Split into train and test sets
        total_files = len(self.pointcloud_files)
        self.train_ratio = 0.8
        train_size = int(total_files * self.train_ratio)
        
        if self.split == 'train':
            self.pointcloud_files = self.pointcloud_files[:train_size]
            self.label_files = self.label_files[:train_size]
        else:
            self.pointcloud_files = self.pointcloud_files[train_size:]
            self.label_files = self.label_files[train_size:]

        print('The size of %s data is %d'%(split,len(self.pointcloud_files)))

        # self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # self.cat = [line.rstrip() for line in open(self.catfile)]
        # self.classes = dict(zip(self.cat, range(len(self.cat)))) # 将类别转换为字典。类别名：类别索引
        # #指定训练和测试数据的路径
        # shape_ids = {}
        # shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        # shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # assert (split == 'train' or split == 'test')
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]] # shape_names[i] = 'airplane'
        # # list of (shape_name, shape_txt_file_path) tuple
        # self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
        #                  in range(len(shape_ids[split]))]
        # print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.pointcloud_files)

    def _get_item(self, index): #取数据，index表示要处理的数据，循环batchsize次，这里是8
        if index in self.cache:  #设置缓存
            point_set, label = self.cache[index]
        else:
            pointcloud_path = os.path.join(self.root, self.pointcloud_files[index])
            label_path = os.path.join(self.root, self.label_files[index])
            
            label = np.loadtxt(label_path, delimiter=',').astype(np.float32).reshape(-1)

            # Standardize the label data
            label[0:3] = (label[0:3] - self.norm_stats['distance_mean']) / self.norm_stats['distance_std']
            label[3:6] = (label[3:6] - self.norm_stats['angle_mean']) / self.norm_stats['angle_std']

            point_set = np.load(pointcloud_path).astype(np.float32) #得到点的具体信息
            if self.uniform:

                mask = point_set[:, 2] > 0.001
                point_set = point_set[mask]

                indices = np.random.choice(point_set.shape[0], 10000, replace=False)
                point_set = point_set[indices, :]

                # 计算每个点到原点的距离（假设距离是三列的平方和）
                distances = np.sqrt(np.sum(point_set[:, :3]**2, axis=1))

                # 筛选出距离在0.0001和3之间的点，并且z坐标大于0.001
                mask = (distances > 0.001) & (distances < 3)
                point_set = point_set[mask]

                point_set = farthest_point_sample(point_set, self.npoints) #最远点采样
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])#相当于3列值，做标准化

            if not self.normal_channel:
                point_set = point_set[:, 0:3] # 提取前三列的数据

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, label) #把点的信息放到缓存当中

        return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)

def get_norm_stats(root):
    all_distance_labels = []
    all_angle_labels = []
    for label_file in os.listdir(root):
        if label_file.startswith('label_') and label_file.endswith('.txt'):
            label_path = os.path.join(root, label_file)
            label = np.loadtxt(label_path, delimiter=',').astype(np.float32).reshape(-1)
            all_distance_labels.extend(label[0:3])  # First three numbers are distance labels
            all_angle_labels.extend(label[3:6])     # Next three numbers are angle labels
    
    all_distance_labels = np.array(all_distance_labels)
    all_angle_labels = np.array(all_angle_labels)
    
    distance_mean = np.mean(all_distance_labels)
    distance_std = np.std(all_distance_labels)
    distance_std = np.clip(distance_std, 1e-2, np.inf)  # clipping to avoid division by zero
    
    angle_mean = np.mean(all_angle_labels)
    angle_std = np.std(all_angle_labels)
    angle_std = np.clip(angle_std, 1e-2, np.inf)  # clipping to avoid division by zero
    
    stats = {
        "distance_mean": distance_mean,
        "distance_std": distance_std,
        "angle_mean": angle_mean,
        "angle_std": angle_std
    }
    
    return stats


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('./data/Dataset_DeYun/', split='train', uniform=True, normal_channel=False,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    # for point,label in DataLoader:
    #     print(point.shape)
    #     print(label.shape)

    #     print(point)
    #     print(label)
    print(get_norm_stats('./data/Dataset_DeYun/'))