from LaNAS.MCTS import MCTS
import copy
import torch
import json
import hwgnas.arch_process as dp
from hwgnas.brp import BRP_Manager
import numpy as np
import time

from ArcToGraphDataset import ArcToGraphDataset

import argparse
from brp_utils import parse_config, build_model

def get_brp_args():
    parser = argparse.ArgumentParser(description='An implementation of latency prediction')
    parser.add_argument('--cfg_latency_file', default='cfg_latency.yaml')
    parser.add_argument('--cfg_acc_file', default='cfg_acc.yaml')
    parser.add_argument('-t', dest='test', action='store_true')
    parser.add_argument('--dump', dest='dump', action='store_true')
    args = parser.parse_args()
    return args
def get_cfg(cfg_raw, mode):
    cfg_runtime = copy.deepcopy(cfg_raw)
    cfg_runtime.update(cfg_runtime[mode])
    return cfg_runtime
def get_brp_cfg(type):
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
    # logger = logging.getLogger('global')
    # logger.info(self.args)
    brp_args=get_brp_args()
    if type=="acc":
        brp_cfg_raw = parse_config(brp_args.cfg_acc_file)
    # logger.info(f'{brp_cfg_raw}')
    elif type=="latency":
        brp_cfg_raw = parse_config(brp_args.cfg_latency_file)
    brp_cfg_test = get_cfg(brp_cfg_raw, 'test')
    if not brp_args.test:
        brp_cfg = get_cfg(brp_cfg_raw, 'train')
    else:
        brp_cfg = brp_cfg_test
    return brp_cfg
def get_latency_predictor(args):
    print(f'args.latency_predictor_type={args.latency_predictor_type}')
    if args.latency_predictor_type=="brp_Jetson_Cora":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/Jetson_Cora_latency_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight.pth'
    elif args.latency_predictor_type=="brp_Jetson_Citeseer":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/Jetson_Citeseer_latency_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight_1724054959.0741792.pth'
    elif args.latency_predictor_type=="brp_Jetson_SIoT":        
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/Jetson_SIoT_layer2_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight_1726374103.9455657.pth'

   
    elif args.latency_predictor_type=="brp_RTX3090_Cora":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/3090_Cora_latency_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight.pth'
    elif args.latency_predictor_type=="brp_RTX3090_Citeseer":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/RTX3090_Citeseer_latency_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight_1724052770.0047565.pth'
    elif args.latency_predictor_type=="brp_RTX3090_SIoT":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/RTX3090_SIoT_layer2_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight_1726209914.1545463.pth'
    
    elif args.latency_predictor_type=="brp_i5_Citeseer":#brp_i5_Citeseer
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/i5_7300HQ_CPU_Citeseer_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight_1724207169.6422195.pth'
    elif args.latency_predictor_type=="brp_i5_Cora":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/i5_7300HQ_CPU_Cora_depth_5_hidden_dim_300_lr_e5_brp_gcn_weight_1724232821.3476002.pth'
    elif args.latency_predictor_type=="brp_i5_SIoT":
        latency_predictor_weights_path = 'RTX3090/log_1000samples_400init_old/latency_predictor_pth/i5_SIoT_layer2_depth_4_hidden_dim_600_lr_e5_brp_gcn_weight_1726219038.1846478.pth'

    
    cfg_latency=get_brp_cfg(type="latency")
    latency_predictor = build_model(cfg_latency['model'])
    latency_predictor = latency_predictor.cuda()
    print(f"=====================")
    # 将加载的权重加载到模型中
    latency_predictor.load_state_dict(torch.load(latency_predictor_weights_path))
    # 将模型设置为预测模式（如果有必要）
    latency_predictor.eval()
    print("latency_predictor_weights_path={}".format(latency_predictor_weights_path))
    return latency_predictor


def wrap_data(data, dtype=None, cuda=True):
    data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
    data = data.cuda() if cuda else data
    return data


def ppo(args, trnr):
    print(f"\nppo中args.latency_predictor_type={args.latency_predictor_type}\n")
    arcToGraph = ArcToGraphDataset(args=args, use_skip_connection=args.use_skip_connection)
    
    latency_predictor=get_latency_predictor(args)

    if args.is_predictor:
        # 待修改
        brp_mg = BRP_Manager(args)
    predictor_train_samples={"actions":[],"accs":[]}
    while len(trnr.total_samples) < args.search_samples:

        if len(trnr.total_samples) % 100 == 0:
            trnr.init_controller()

        if args.is_predictor and len(trnr.total_samples) >= args.predictor_init_samples and flag == 1:
        # if args.is_predictor:
            flag = 0
            # mlp_mg.train_mlp(mask_samples)
            # 待修改
            brp_mg.train_brp(predictor_train_samples)
            for i in range(args.select_samples):
                while True:
                    structure_list, log_probs, entropies, actions_index = trnr.controller_sample()
                    tmp_action = copy.deepcopy(structure_list[0])
                    tmp_action[-1] = trnr.submodel_manager.n_classes
                    if str(tmp_action) not in trnr.total_samples:
                        break
                    else:
                        print('Arise a repeated arch', tmp_action)

                trnr.total_samples.append(str(tmp_action))
                np_entropies = entropies.data.cpu().numpy()

                '''========================================='''
                start_time = time.time()
                
                architecture = structure_list[0]
                adj_matrix, feature_matrix, edge_index = arcToGraph.get_adj_and_feature_and_label(
                    actions=architecture)                
                
                adj_matrix=wrap_data(adj_matrix,dtype=torch.double).unsqueeze(0)
                feature_matrix = wrap_data(feature_matrix,dtype=torch.double).unsqueeze(0)

                #计算延迟奖励
                latency_rewards = latency_predictor(adj_matrix, feature_matrix)
                print("延迟奖励为：",latency_rewards)                

                if args.use_latency_limit==True and latency_rewards>args.latency_limit:
                    print(f"{architecture} 不计入")
                    skip_flag=1
                else:
                    skip_flag=0

                latency_rewards = args.latency_weight*latency_rewards
                #计算精度奖励
                acc_rewards = brp_mg.brp_pred_acc(adj_matrix, feature_matrix)
                print("精度奖励为：",acc_rewards)                
                
                rewards = args.alpha * acc_rewards - args.beta * latency_rewards
                # 将得到的奖励转换为float类型
                rewards = rewards.cpu() 
                rewards = rewards.detach().numpy()[0]
                end_time = time.time()

                '''========================================='''

                if skip_flag==0:
                    path = args.log_output_dir + '/' + 'rand_seed' + str(args.random_seed) + '.txt'
                    trnr.submodel_manager.record_action_info(path, structure_list[0], acc_rewards,end_time-start_time)

                if args.entropy_mode == 'reward':
                    rewards = rewards + args.entropy_coeff * np_entropies
                elif args.entropy_mode == 'regularizer':
                    rewards = rewards * np.ones_like(np_entropies)
                else:
                    raise NotImplementedError(f'Unkown entropy mode: {args.entropy_mode}')

                torch.cuda.empty_cache()

                trnr.collect_trajectory(rewards, log_probs, actions_index)
                print(20 * '-', 'episodes: ', trnr.buf.get_buf_size(), 20 * '-')

                if trnr.buf.get_buf_size() % args.episodes == 0:
                    trnr.train_controller_by_ppo2()
        else:
            flag = 1
            for i in range(args.select_samples):

                while True:
                    structure_list, log_probs, entropies, actions_index = trnr.controller_sample()
                    tmp_action = copy.deepcopy(structure_list[0])
                    tmp_action[-1] = trnr.submodel_manager.n_classes

                    if str(tmp_action) not in trnr.total_samples:
                        break
                    else:
                        print('Arise a repeated arch', tmp_action)

                trnr.total_samples.append(str(tmp_action))
                # mask = dp.encode_arch_to_mask(structure_list)[0]

                #计算延迟奖励
                architecture = structure_list[0]
                adj_matrix, feature_matrix, edge_index = arcToGraph.get_adj_and_feature_and_label(
                    actions=architecture)               
                adj_matrix=wrap_data(adj_matrix,dtype=torch.double).unsqueeze(0)
                feature_matrix = wrap_data(feature_matrix,dtype=torch.double).unsqueeze(0)

                latency_rewards = latency_predictor(adj_matrix, feature_matrix).cpu().detach().numpy()[0]
                print("延迟奖励为：",latency_rewards)

                if args.use_latency_limit==True and latency_rewards > args.latency_limit:
                    print(f"{architecture} 不计入")
                    skip_flag=1
                else:
                    skip_flag=0
                latency_rewards = args.latency_weight*latency_rewards
                np_entropies = entropies.data.cpu().numpy()
                acc_rewards, reward = trnr.get_reward(structure_list, np_entropies,skip_flag)
                torch.cuda.empty_cache()
                # print("acc_rewards=",acc_rewards)
                # print("type(acc_rewards)=",type(acc_rewards))
                rewards = args.alpha * acc_rewards - args.beta * latency_rewards
                # 待修改
                #保存训练精度预测器的样本
                predictor_train_samples["actions"].append(structure_list[0])
                predictor_train_samples["accs"].append(acc_rewards)

                trnr.collect_trajectory(rewards, log_probs, actions_index)
                print(20 * '-', 'episodes: ', trnr.buf.get_buf_size(), 20 * '-')
                if trnr.buf.get_buf_size() % args.episodes == 0:
                    trnr.train_controller_by_ppo2()

