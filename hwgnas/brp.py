import torch
import torch.nn.functional as F
import numpy as np
import json
import numpy as np
import argparse
import copy

from models.gcn import GCN
from brp_utils import parse_config, build_model, build_optimizer, build_lr_scheduler, evaluate

from ArcToGraphDataset import ArcToGraphDataset
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

import torch.optim as optim
class PredictorDatasetManager:
    def __init__(self,args):
        
        self.adj_matrx_list=[]
        self.feature_matrix_list=[]
        self.edge_index_list=[]
        self.label_list=[]
        self.args =args

    def wrap_data(self,data, dtype=None, cuda=True):
        data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
        data = data.cuda() if cuda else data
        return data    
    def action_convert_matrix(self,samples):
        adj_matrx_list=[]
        feature_matrix_list=[]
        edge_index_list=[]
        label_list=[]
        actions = samples["actions"]
        labels = samples["accs"]

        arcToGraph = ArcToGraphDataset(args=self.args, use_skip_connection=self.args.use_skip_connection)
        for action,label in zip(actions,labels):  
            architecture = action 

            adj_matrix, feature_matrix, edge_index= arcToGraph.get_adj_and_feature_and_label(
                actions=architecture)
            adj_matrx_list.append(adj_matrix)
            feature_matrix_list.append(feature_matrix)
            edge_index_list.append(edge_index)
            label_list.append(label)
            # print(label)
        adj_matrx_list=self.wrap_data(adj_matrx_list,dtype=torch.double)
        feature_matrix_list = self.wrap_data(feature_matrix_list, dtype=torch.double)
        edge_index_list = self.wrap_data(edge_index_list, dtype=torch.double)
        label_list = self.wrap_data(label_list, dtype=torch.double)
        return adj_matrx_list, feature_matrix_list, edge_index_list,label_list
        # print("self.adj_matrx_list.size()=",self.adj_matrx_list.size())
   
    def make_train_dataset(self,samples):
        adj_matrx_list, feature_matrix_list, edge_index_list,label_list=self.action_convert_matrix(samples)
        # print(f"label_list={label_list}\nadj_matrx_list={adj_matrx_list}")
        dataset=TensorDataset(adj_matrx_list,feature_matrix_list,label_list)
        # print(f"len(dataset)={len(dataset)}")
        # 定义批次大小
        batch_size = 10
        chunk=int(len(dataset)/batch_size)
        print(f"batch_size={batch_size},chunk={chunk}")

        # 划分训练集和测试集的索引        
        train_indices = list(range(chunk * batch_size))
        train_sampler = SubsetRandomSampler(train_indices)
        # 创建 DataLoader 实例，指定数据集、批次大小和采样器
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        # test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        
        
        return train_loader

def build_model(cfg_model):
    model_type = cfg_model['type']
    if model_type == 'GCN':
        model = GCN(**cfg_model['kwargs'])
    else:
        raise NotImplementedError

    return model
def get_brp_args():
        parser = argparse.ArgumentParser(description='An implementation of latency prediction')
        parser.add_argument('--cfg_file', default='cfg.yaml')
        parser.add_argument('-t', dest='test', action='store_true')
        parser.add_argument('--dump', dest='dump', action='store_true')
        args = parser.parse_args()
        return args
def get_cfg(cfg_raw, mode):
    cfg_runtime = copy.deepcopy(cfg_raw)
    cfg_runtime.update(cfg_runtime[mode])
    return cfg_runtime
def get_brp_cfg():

    brp_args=get_brp_args()
    brp_cfg_raw = parse_config(brp_args.cfg_file)
    brp_cfg_test = get_cfg(brp_cfg_raw, 'test')
    if not brp_args.test:
        brp_cfg = get_cfg(brp_cfg_raw, 'train')
    else:
        brp_cfg = brp_cfg_test
    return brp_cfg
class BRP_Manager(object):
    def __init__(self, args):
        self.args=args 
        self.cfg=get_brp_cfg()
        self.model = build_model(self.cfg['model'])
        # self.model = self.build_model()
        self.model = self.model.cuda()
        self.optimizer = build_optimizer(self.cfg['trainer']['optimizer'], self.model)
        # self.optimizer = self.build_optimizer(self.model)
        self.epochs = self.args.latency_predictor_epochs
        self.predictorDatasetManager=PredictorDatasetManager(args)
    def build_model(self,depth=4,feature_dim=17, hidden_dim=600, augments_dim=0, dropout_rate=2.0e-3):
        initializer = {'gc': {'method': 'thomas'}}
        criterion_cfg = {'type': 'MAPE', 'loss_weight': 1.0}
        model = GCN(depth,feature_dim, hidden_dim, augments_dim, dropout_rate, initializer, criterion_cfg)
        return model
    def build_optimizer(self, model,lr=4.0e-5,weight_decay=5.0e-4):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    def mean_absolute_percentage_error(self, y_true, y_pred):
        # 计算绝对百分比误差
        absolute_percentage_error = torch.abs((y_true - y_pred) / y_true)
        
        # 忽略掉真实值为0的情况
        absolute_percentage_error = torch.where(y_true != 0, absolute_percentage_error, torch.zeros_like(absolute_percentage_error))
        
        # 计算平均绝对百分比误差
        mape = 100 * torch.mean(absolute_percentage_error)
    
        return mape

    def train_brp(self, samples):
        print(samples["actions"])
        self.model.train()
        train_loader=self.predictorDatasetManager.make_train_dataset(samples)
        # print(train_loader[0])
        for epoch in range(self.epochs): 
            avg_loss=0             
            # print("出错")
            for batch_idx, (adg_matrix_batch, feature_matrix_batch, label_batch) in enumerate(train_loader):
                y= self.model(adg_matrix_batch,feature_matrix_batch)
                target=label_batch
                if epoch==self.epochs-1:
                    print(f'y={y}\ntarget={target}')
                loss = self.mean_absolute_percentage_error(target,y)          
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss+=loss.item()
            print(f"epoch:{epoch}|| loss:{avg_loss/(batch_idx+1)}-----------------------")
            

    def brp_pred_acc(self, adj_matrix, feature_matrix):
        self.model.eval()
        y= self.model(adj_matrix,feature_matrix)     
        return y.item()  


            










