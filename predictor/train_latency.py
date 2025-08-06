import argparse
import os
import copy
import logging
import torch
from utils import parse_config, build_model, build_optimizer, build_lr_scheduler, evaluate
from utils import EarlyStopping
import torch
import torch.nn as nn
import numpy as np

import os
from torch.optim import Adam
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from ArcToGraphDataset import ArcToGraphDataset

import time

import yaml
def parse_config(cfg_name):
    cfg = None
    with open(cfg_name, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg


def get_cfg(cfg_raw, mode):
    cfg_runtime = copy.deepcopy(cfg_raw)
    cfg_runtime.update(cfg_runtime[mode])
    return cfg_runtime



from models.gcn import GCN
def build_model(cfg_model):
    model_type = cfg_model['type']
    if model_type == 'GCN':
        model = GCN(**cfg_model['kwargs'])
    # elif model_type == 'mlp':
    #     model = MLP(**cfg_model['kwargs'])
    else:
        raise NotImplementedError

    return model


def wrap_data(data, dtype=None, cuda=True):
    data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
    data = data.cuda() if cuda else data
    return data
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

class PredictorDatasetManager:
    def __init__(self,dataset_path):
        print(f"dataset_path={dataset_path}")

        self.dataset_path=dataset_path
        self.adj_matrx_list=[]
        self.feature_matrix_list=[]
        self.edge_index_list=[]
        self.label_list=[]

        self.args = self.get_args()
        #获取数据集
        self.action_convert_matrix()
        #构建数据集
        self.train_loader,self.test_loader=self.make_dataset()

    def wrap_data(self,data, dtype=None, cuda=True):
        data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
        data = data.cuda() if cuda else data
        return data
    def get_args(self):
        parser = argparse.ArgumentParser(description='An implementation of latency prediction')
        # controller
        parser.add_argument('--layers_of_child_model', type=int, default=2)
        # child model
        parser.add_argument("--GNN_dataset", type=str, default="Cora", required=False,
                            help="The input dataset.")
        # Output
        parser.add_argument('--log_output_dir', type=str, default='./log_exp_res')
        # random_seach.py中使用到的
        # 存储随机采样到的GNN架构的路径
        parser.add_argument('--dataset_random_dir', type=str, default='./dataset_random/')
        # random_search_samples_num
        parser.add_argument('--random_search_samples_num', type=int, default=300)
        # 运行平台相关
        parser.add_argument('--hardware_platform', type=str, default='RTX_3080_Ti')
        # 延迟预测器相关
        parser.add_argument('--latency_epochs', type=int, default=30)
        parser.add_argument('--predictor_dataset_type', type=str, default='latency')
        # parser.add_argument('--predictor_type', type=str, default='gin')
        parser.add_argument('--input_dim', type=int, default=17)  # 构造的节点特征维度
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--predictor_layer_nums', type=int, default=3)  # 预测器层数
        # 搜索空间有关
        parser.add_argument('--num_state', type=int, default=5)
        # GNN模型层数
        # parser.add_argument('--layer_nums', type=int, default=2)
        # 是否生成跳跃连接
        parser.add_argument('--use_skip_connection', type=bool, default=False)
        args = parser.parse_args()
        return args
    def action_convert_matrix(self):
        if self.args.predictor_dataset_type == "accuracy":
            print("使用action-accuracy数据集")
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            df = pd.read_csv(self.dataset_path)
            actions = df['action']
            labels = df['val_acc']
        else:
            print("使用latency数据集")
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            df = pd.read_csv(self.dataset_path)
            actions = df['action']
            labels = df['infer_time']

        arcToGraph = ArcToGraphDataset(args=self.args, use_skip_connection=self.args.use_skip_connection)
        for action,label in zip(actions,labels):
            architecture = eval(action)
            adj_matrix, feature_matrix, edge_index= arcToGraph.get_adj_and_feature_and_label(
                actions=architecture)
            self.adj_matrx_list.append(adj_matrix)
            self.feature_matrix_list.append(feature_matrix)
            self.edge_index_list.append(edge_index)
            self.label_list.append(label)
            # print(label)
        self.adj_matrx_list=self.wrap_data(self.adj_matrx_list,dtype=torch.double)
        self.feature_matrix_list = self.wrap_data(self.feature_matrix_list, dtype=torch.double)
        self.edge_index_list = self.wrap_data(self.edge_index_list, dtype=torch.double)
        self.label_list = self.wrap_data(self.label_list, dtype=torch.double)
        # print(self.adj_matrx_list.size())
    def make_dataset(self):
        dataset=TensorDataset(self.adj_matrx_list,self.feature_matrix_list,self.label_list)
        # 定义批次大小
        batch_size = 10
        # 划分训练集和测试集的索引
        train_indices = list(range(1400 * batch_size))
        test_indices = list(range(1400 * batch_size, 2000 * batch_size))

        # 创建 SubsetRandomSampler 对象
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # 创建 DataLoader 实例，指定数据集、批次大小和采样器
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


        # 遍历测试集的每个批次
        return train_loader,test_loader
# pdatamanager=PredictorDatasetManager("Cora_extend_latency_2_layers.csv")
class TrainBrpPredictor:
    def __init__(self): 
        self.pdatamanager=PredictorDatasetManager("./latency_dataset/3090_Cora_actions_latency_2_layers.csv")
        self.train_loader=self.pdatamanager.train_loader
        self.test_loader = self.pdatamanager.test_loader
        self.args=self.get_args()
        self.cfg=self.get_cfg()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
        self.logger = logging.getLogger('global')
    def get_args(self):
        parser = argparse.ArgumentParser(description='An implementation of latency prediction')
        parser.add_argument('--cfg_file', default='cfg_latency.yaml')
        parser.add_argument('-t', dest='test', action='store_true')
        parser.add_argument('--dump', dest='dump', action='store_true')
        args = parser.parse_args()
        return args
    def get_cfg(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
        logger = logging.getLogger('global')
        logger.info(self.args)
        cfg_raw = parse_config(self.args.cfg_file)
        logger.info(f'{cfg_raw}')

        cfg_test = get_cfg(cfg_raw, 'test')
        if not self.args.test:
            cfg = get_cfg(cfg_raw, 'train')
        else:
            cfg = cfg_test
        return cfg
    
    def mean_absolute_percentage_error(self, y_true, y_pred):
        # 计算绝对百分比误差
        absolute_percentage_error = torch.abs((y_true - y_pred) / y_true)
        
        # 忽略掉真实值为0的情况
        absolute_percentage_error = torch.where(y_true != 0, absolute_percentage_error, torch.zeros_like(absolute_percentage_error))
        
        # 计算平均绝对百分比误差
        mape = 100 * torch.mean(absolute_percentage_error)
    
        return mape
    def train(self):
        time_rand=time.time()
        model = build_model(self.cfg['model'])
        model = model.cuda()

        epochs = self.cfg['trainer']['epochs']
        st_epoch = 0
        best_val_acc = 0
        best_epoch = None

        optimizer = build_optimizer(self.cfg['trainer']['optimizer'], model)
        step_on_val_loss = (self.cfg['trainer']['lr_scheduler']['type'] in ['ReduceLROnPlateau'])
        step_on_val_loss_epoch = self.cfg['trainer']['lr_scheduler'].get('step_on_val_loss_epoch', -1)
        lr_scheduler = build_lr_scheduler(self.cfg['trainer']['lr_scheduler'], optimizer)

        es = EarlyStopping(**self.cfg['trainer']['early_stopping']['kwargs'])
        es_start_epoch = self.cfg['trainer']['early_stopping']['start_epoch']

        resume = self.cfg['trainer'].get('resume', None)
        if resume is not None:
            ckpt = torch.load(resume)
            self.logger.info(f'resuming from {resume}')
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            st_epoch = ckpt['epoch'] + 1

        save_freq = self.cfg['trainer'].get('save_freq', 1)
        best_mape_avg=100
        for epoch in range(st_epoch, epochs):
            print(f"epoch:{epoch}-----------------------")
            model.train()

            for batch_idx, (adg_matrix_batch, feature_matrix_batch, label_batch) in enumerate(self.train_loader):
                # print(f"Training Batch {batch_idx}: x1 - {x1_batch.shape}, x2 - {x2_batch.shape}, y - {y_batch.shape}")
                loss = model(adg_matrix_batch,feature_matrix_batch,label_batch)                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            best_mape_avg, val_loss = self.test(best_mape_avg=best_mape_avg, time_rand=time_rand, model=model, return_loss=True)

            if step_on_val_loss:
                lr_scheduler.step(val_loss.cpu().item())
            if epoch > es_start_epoch:
                if es.step(val_loss.cpu().item()):
                    print('Early stopping criterion is met, stop training now.')
                    break



    def test(self,best_mape_avg,model, time_rand,model_save_path="./predictor_model_save/Cora_extend_latency/",return_loss=False):
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        # inference
        model.eval()
        
        resume = self.cfg['trainer'].get('resume', None)
        if resume is not None:
            ckpt = torch.load(resume)
            self.logger.info(f'loading ckpt from {resume}')
            model.load_state_dict(ckpt['state_dict'])
        
        results = []
        all_loss = 0 if return_loss else None
        train_loss_log = {"mape_value":[]}
        with torch.no_grad():            
            n=0
            mape_all=0
            loss_all=0
            for batch_idx, (adg_matrix_batch, feature_matrix_batch, label_batch) in enumerate(self.test_loader):
                y, target,loss = model.forward(adg_matrix_batch,feature_matrix_batch,label_batch, return_loss)
                mape=self.mean_absolute_percentage_error(target, y)
                n+=1
                mape_all+=mape
                loss_all+=mape
            avg_mape=mape_all/n
            avg_loss=loss_all/n
            print(f"avg_mape={avg_mape}")

            if(best_mape_avg>avg_mape.item()):
                print(f"best_mape_avg={best_mape_avg},avg_mape={avg_mape},{best_mape_avg>avg_mape.item()}")
                torch.save(model.state_dict(), model_save_path+"RTX3090_Cora_layer2_depth_4_hidden_dim_600_lr_e6_brp_gcn_weight_"+str(time_rand)+".pth")
                
                train_loss_log["mape_value"].append(avg_mape.item())

                df_test_loss = pd.DataFrame(train_loss_log)
                df_test_loss.to_csv(model_save_path+"RTX3090_Cora_layer2_depth_4_hidden_dim_600_lr_e6_brp_mape_"+str(time_rand)+".csv", index=False)

                best_mape_avg=avg_mape.item()
        return best_mape_avg, avg_loss

manager=TrainBrpPredictor()
manager.train()
