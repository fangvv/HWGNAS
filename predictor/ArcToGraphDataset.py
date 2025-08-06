
import torch
import pandas as pd
import argparse
import time
import numpy as np
from search_space_predictor import MacroSearchSpace

def column_sum(matrix):
    matrix_np = np.array(matrix)
    return list(np.sum(matrix_np, axis=0))

class ArcToGraphDataset:
    def __init__(self,args,use_skip_connection):
        self.args = args
        macroSearchSpace = MacroSearchSpace()
        self.search_space = macroSearchSpace.get_search_space()
        self.num_state=5
        self.use_skip_connection=use_skip_connection


    def get_action_sample(self,path,samples_num):  # 使用已经随即搜索得到的动作，获得其精度和延迟
        df_action_tuple = pd.read_csv(path)
        action_dataset = df_action_tuple["action"]
        acc_dataset=df_action_tuple["infer_time"]
        self.total_samples_actions = []
        self.total_samples_labels=[]
        # print("self.total_samples_num={}".format(self.args.random_search_samples_num))
        for i in range(0, samples_num):
            self.total_samples_actions.append(eval(action_dataset[i]))
            self.total_samples_labels.append(acc_dataset[i])
        print("从文件 {} 中获取 {} 个样本完成".format(path, len(self.total_samples_actions)))

    def getGraphDataset(self,action_path,samples_num,save_path):
        #获取样本
        self.total_samples_actions=[]
        self.total_samples_labels=[]
        self.get_action_sample(action_path,samples_num)

        #将action_tuple转换为Graph
        graph_dataset={"adj_matrix":[],"feature_matrix":[], "edge_index":[],"label":[]}
        for architecture,train_accs in zip(self.total_samples_actions,self.total_samples_labels):
            if self.use_skip_connection:
                adj_matrix, feature_matrix, edge_index,label_accuracy = self.get_adj_and_feature_and_label_use_connection(
                    actions_tuple=architecture,
                    label=train_accs)
            else:
                adj_matrix, feature_matrix, edge_index, label_accuracy = self.get_adj_and_feature_and_label(
                    actions=architecture,
                    label=train_accs)
            graph_dataset["adj_matrix"].append(adj_matrix)
            graph_dataset["feature_matrix"].append(feature_matrix)
            graph_dataset["edge_index"].append(edge_index)
            graph_dataset["label"].append(label_accuracy)
            print("\narchitecture={},train_accs={}".format(architecture,train_accs))
            print("adj_matrix={}, \nfeature_matrix={}, \nedge_index={}".format(adj_matrix, feature_matrix, edge_index))
            df_graph_dataset = pd.DataFrame(graph_dataset)

            print("save_path={}".format(save_path))
            print()
            df_graph_dataset.to_csv(save_path)


    def get_adj_and_feature_and_label(self, actions):
        '''start===========得到每个操作在其操作类型中的索引==================='''
        actions_without_anchor = actions
        # print(f' actions_without_anchor={ actions_without_anchor}')
        skip_connection_list = [[]]
        action_index = 0
        op_index_list = []
        for i in range(0, self.args.layers_of_child_model):
            for key in self.search_space.keys():
                op_index = self.search_space[key].index(actions_without_anchor[action_index])
                action_index += 1
                op_index_list.append(op_index)
        actions_index = torch.tensor(op_index_list, device='cuda:0')
        # print("actions_index={}".format(actions_index))
        '''start=========生成节点的输入特征矩阵=========================='''
        # 节点个数 = 操作类型数*层数(最后一层的最后一个hidden_num可表示数据集种类——>表示输入特征维度、输出特征维度) + 1（全局节点1个，）
        # op_type_num=len(self.search_space.keys())
        node_num = self.args.layers_of_child_model * self.num_state + 1
        candidate_num_list = []
        for key in self.search_space.keys():
            candidate_num_list.append(len(self.search_space[key]))
        max_candidate_num = max(candidate_num_list)
        # feature_dim = 1 + self.num_state + max_candidate_num  # 得到特征维度：1+5+11 = 17, 1为全局节点，op_type_num为操作类型个数，max_candidate_num为最大的候选操作数
        feature_dim=17
        feature_matrix = np.zeros((node_num, feature_dim))  # 定义一个初始化输入特征矩阵
        feature_matrix[0][0] = 1  # 代表索引为0的节点为全局节点

        '''start=========处理中间的操作节点的特征=========================='''
        for layer_index in range(0, self.args.layers_of_child_model):
            for op_type in range(0, self.num_state):
                '''处理操作类型'''
                feature_matrix[1 + layer_index * self.num_state + op_type][1 + op_type] = 1  # 操作类型的特征索引为：1+op_type
                '''处理选中的候选操作'''
                # 行坐标=1(第1行是全局节点)+层索引*操作类型数量+候选操作
                row = 1 + layer_index * self.num_state + op_type
                # 列索引 =1(第1列是全局节点)+ 候选操作种类(第2~op_type_num列表示操作类型) + 选中的操作在其操作类型列表中的索引
                col = 1 + self.num_state + actions_index[layer_index * self.num_state + op_type]
                feature_matrix[row][col] = 1  # 候选操作的特征索引为：8+op_i
                # print("feature_matrix[{}][{}]=1".format(row,col))

        '''==========================构建邻接矩阵=========================='''
        '''构建邻接矩阵'''
        edge_row_list = []
        edge_col_list = []

        adj_matrix = np.zeros(shape=(node_num, node_num))
        adj_matrix[0][0] = 1
        for node_i in range(1, node_num):
            # print("node_i={}".format(node_i))
            adj_matrix[node_i][node_i] = 1  # 将对角元素置为1 ——>自环

            adj_matrix[0][node_i] = 1  # 全局节点与所有其它节点相连

            # 对于不是每层的最后一个操作，其直接与后一个操作相连
            if node_i < self.args.layers_of_child_model * self.num_state:
                adj_matrix[node_i][node_i + 1] = 1

        for i in range(node_num):
            for j in range(node_num):
                if (adj_matrix[i][j] == 1):
                    edge_row_list.append(i)
                    edge_col_list.append(j)
        edge_index=[edge_row_list, edge_col_list]
        # edge_index = torch.tensor([edge_row_list, edge_col_list])

        return adj_matrix, feature_matrix, edge_index

