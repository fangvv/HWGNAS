from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
import numpy as np
import torch
import os
import sys
# sys.path.append('/mnt/DatasetMake/')
# from make.randomNodeSplit import RandomNodeSplit

# 定义自己的数据集类
class mydataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(mydataset, self).__init__(root, transform, pre_transform)

    # 原始文件位置
    @property
    def raw_file_names(self):
        return ['device_feature_file.content', 'SIoT_edge_file.cites']
        # siot.content文件 两列：边的起始点 边的终点
        # siot.cites文件：节点id 节点特征 节点标签

    # 文件保存位置
    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass
        # 因为是自己创建的数据集，因此不用下载

    # 数据处理逻辑
    def process(self):
        idx_features_labels = np.genfromtxt(self.raw_paths[0])
        x = idx_features_labels[:, 1:-1]
        print(f'len(x)={len(x)}')
        print(f'x[1]={x[1]}')
        print(f'len(x[0])={len(x[0])}')
        x = torch.tensor(x, dtype=torch.float32)
        y, label_dict = self.encode_labels(np.genfromtxt(self.raw_paths[0], dtype='str', usecols=(-1,)))
        y = torch.tensor(y)
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        id_node = {j: i for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt(self.raw_paths[1], dtype=np.int32)
        print(f'edges_unordered={edges_unordered}')
        
        edge_str = [id_node[each[0]] for each in edges_unordered]
        # print(f'edge_str={edge_str}')
        edge_end = [id_node[each[1]] for each in edges_unordered]
        # print(f'edge_end={edge_end}')
        edge_index = torch.tensor([edge_str, edge_end], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save(data, os.path.join(self.processed_dir, f'data.pt'))

    def encode_labels(self, labels):
        classes = sorted(list(set(labels)))
        labels_id = [classes.index(i) for i in labels]
        label_dict = {i: c for i, c in enumerate(classes)}
        return labels_id, label_dict

    # 定义总数据长度
    def len(self):
        idx_features_labels = np.genfromtxt(self.raw_paths[0], dtype=np.int32)
        uid = idx_features_labels[:, 0:1]
        return len(uid)

    # 定义获取数据方法
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data.pt'))
        return data

# # 
# dataset = mydataset('./data/')
# # data = dataset[0].to(device)
# data = dataset[0]
# print(f'data={data}')
# print(f'data.x={data.x}')
# print(f'data.y={data.y}')
# print(f'data.edge_index={data.edge_index}')
# data = RandomNodeSplit()(data)

# print(f'data.num_nodes = {data.num_nodes}')
# # print(f'num_class = {data.num_labels}')
