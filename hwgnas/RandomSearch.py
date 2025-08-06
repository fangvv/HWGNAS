import torch
import time
import argparse
from search_space import MacroSearchSpace
import random
import os
from gnn_model_manager import GeoCitationManager
import pandas as pd  # 数据存入csv文件


def register_default_args(parser):
    # 随机搜索
    parser.add_argument("--total_samples_num", type=int, default=2000,
                        help="number of samples to trian predictor")  # 随机搜索的用于训练预测器的样本数量
    parser.add_argument('--accuracy_path', type=str, default='./dataset_actions/actions_accuracy')

    parser.add_argument('--format', type=str, default='two')
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--top_k', type=int, default=10)
    # parser.add_argument('--search_samples', type=int, default=2000)

    # child model
    parser.add_argument("--num_layer", type=int, default=2,
                        help="number of submodel's layer")  # Li
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--latency_epochs", type=int, default=3,
                        help="number of run epochs for measuring latency")
    # parser.add_argument("--retrain_epochs", type=int, default=300,
    # help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}")

    # 运行平台相关
    parser.add_argument('--platform', type=str, default='3090')

    # Output
    parser.add_argument('--log_output_dir', type=str, default='./log_exp_res')
    parser.add_argument('--action_dir', type=str, default='./actions_latency/')


def build_args():
    parser = argparse.ArgumentParser(description='HWGNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


class RandomSearch:
    def __init__(self, args):
        self.args = args
        # 判断存储动作-精度-延迟的文件夹是否存在
        if not os.path.exists(self.args.action_dir):
            os.mkdir(self.args.action_dir)

        self.macroSearchSpace = MacroSearchSpace()
        self.layers = args.num_layer
        self.search_space = self.macroSearchSpace.get_search_space()
        self.total_samples = []
        self.total_samples_num = args.total_samples_num
        self.submodel_manager = GeoCitationManager(self.args)

    def get_action_sample(self):
        filename = self.args.dataset + "_" + str(self.total_samples_num) + "_actions_2_layers.csv"
        path = "./dataset_actions/" + filename
        df_action = pd.read_csv(path)
        action_dataset = df_action["action"]
        self.total_samples = []
        for i in range(0, self.args.total_samples_num):
            self.total_samples.append(eval(action_dataset[i]))

        print("get {} samples from {} finished".format(len(self.total_samples),path))

    def random_generate_actions(self, save=None):

        action_dataset = {"action": []}
        for sample_num in range(0, self.total_samples_num):  # 生成第sample_num个样本
            actions = []
            for i in range(0, self.layers):  # 生成第i层的动作
                for key in self.search_space.keys():
                    op_index = random.randint(0, len(self.search_space[key]) - 1)
                    action = self.search_space[key][op_index]
                    actions.append(action)
            # print(actions)
            if str(actions) not in self.total_samples:
                self.total_samples.append(actions)
                action_dataset["action"].append(actions)
            else:
                print('Arise a repeated arch', actions)

        if save == True:
            # 创建文件路径
            path = self.args.action_dir
            filename = self.args.dataset + "_" + str(self.total_samples_num) + "_" + "actions_" + str(
                self.args.num_layer) + "_layers.csv"
            save_path = path + filename
            # 存入文件
            df_action_dataset = pd.DataFrame(action_dataset)
            df_action_dataset.to_csv(save_path)

    def get_model_latency_dataset(self):
        print("\n--------------------开始获得模型的推理延迟-----------------------")
        madel_latency_log = {"action": [], "infer_time": []}

        # 创建文件路径
        path = self.args.action_dir
        filename = self.args.platform + '_' + self.args.dataset + "_latency_" + str(self.args.num_layer) + "_layers.csv"
        save_path = path + filename
        # 确保文件能够存入

        for i in range(0, len(self.total_samples)):
            infer_time = self.submodel_manager.get_model_infer_latency(action=self.total_samples[i])
            madel_latency_log["action"].append(self.total_samples[i])
            madel_latency_log["infer_time"].append(infer_time)

        # 存入数据
        df_model_latency = pd.DataFrame(madel_latency_log)
        df_model_latency.to_csv(save_path)


def test_RandomSearch():
    args = build_args()
    randomSearch = RandomSearch(args)

    # 随机生成用作训练样本的架构
    randomSearch.random_generate_actions(save=True)
    randomSearch.get_action_sample()
    # 获得架构对应的推理延迟
    randomSearch.get_model_latency_dataset()

test_RandomSearch()

