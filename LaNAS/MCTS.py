# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
import egnas.arch_process as dp

from egnas.mlp import MLP_Manager
import copy
import torch

class MCTS(object):
    #############################################

    def __init__(self, search_space, trainer, tree_height, args):
        # self.ARCH_CODE_LEN           =  int( len( search_space["b"] ) / 2 )
        self.ARCH_CODE_LEN = search_space['mask_len']
        self.SEARCH_COUNTER          =  0
        self.samples                 =  {}
        self.nodes                   =  []
        # search space is a tuple, 
        # 0: left side of the constraint, i.e. A
        # 1: right side of the constraint, i.e. b
        self.search_space            =  search_space['init_point']      # self.search_space is invalid in this exp.
        self.Cp                      =  args.Cp
        self.trainer                 =  trainer
        # pre-defined for generating masks for supernet

        self.args = args

        self.select_num = args.select_samples
        self.times_use_predictor = args.times_use_predictor
        
        #initialize the a full tree
        Node.obj_counter = 0
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1)  > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            elif (i -1) > 0:
                is_good_kid = True
            parent_id = i // 2  - 1
            if parent_id == -1:
                self.nodes.append( Node( None, is_good_kid, self.ARCH_CODE_LEN, True ) )
            else:
                self.nodes.append( Node(self.nodes[parent_id], is_good_kid, self.ARCH_CODE_LEN, False) )
        
        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT

        if self.args.is_predictor:
            self.mlp_mg = MLP_Manager(args, self.ARCH_CODE_LEN)
        
    def dump_all_states(self):
        node_path = 'mcts_agent'
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
        
    def collect_samples(self, results):
        for arch in results:
            # if arch not in self.samples and results[arch] != 0.0:
            if arch not in self.samples:
                self.samples[arch] = results[arch]

    def train_nodes(self):
        for i in self.nodes:
            i.train()

    def predict_nodes(self):
        for i in self.nodes:
            i.predict()

    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag( json.loads(k), v )

    def populate_prediction_data(self):
        self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag( k, 0.0 )
    
    def init_train(self):
        for i in range(0, 300):
            net = np.random.choice(self.search_space, 1)
            self.search_space.remove(net)
            net_str = json.dumps( net )
            acc  = self.net_trainer.train_net( net )
            self.samples[net_str] = acc
        print("="*10 + 'collect '+ str(len(self.samples) ) +' nets for initializing MCTS')
        
    def print_tree(self):
        print('-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)

    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )
                print("=====>loads:", self.SEARCH_COUNTER," counter" )
                
    def select(self):
        self.reset_to_root()
        boundaries = []
        curt_node = self.ROOT
        
        curt_node.print_bag()
        starting_point = curt_node.get_rand_sample_from_bag()
        
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]   # choose the max UCT.
            w, b, x_bar    = curt_node.get_boundary( )
            boundaries.append( (w, b, x_bar, choice) )
            curt_node = curt_node.kids[choice]                  # choose left child or right child.
            if curt_node.get_rand_sample_from_bag() is not None:
                starting_point = curt_node.get_rand_sample_from_bag()

            print('select_current_node.id:', curt_node.id, 'left_kid.uct:', UCT[0], 'right_kid.uct:', UCT[1], 'choice', choice)
        return curt_node, boundaries, starting_point

    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            if acc > 0:
                if curt_node.n > 0:
                    curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
                else:
                    curt_node.x_bar = acc
            curt_node.n    += 1
            curt_node = curt_node.parent

    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len( i.bag )
        assert counter == len( self.search_space )

        return counter
    
    def prepare_boundaries(self, boundaries):
        #2*self.ARCH_CODE_LEN+
        W = []
        B = []
        for boundary in boundaries:
            w, b, x_bar, choice = boundary
            righthand = x_bar - b[0]
            lefthand  = w
            if righthand == float("inf"):
                continue
            if choice == 0:
                #transform to W*x <= b
                lefthand  = -1*lefthand
                righthand = -1*righthand

            lefthand = lefthand.reshape(np.prod(lefthand.shape))
            W.append(lefthand)
            B.append(righthand)

        W = np.array(W)
        B = np.array(B)

        return W, B
        
    # def search_samples_under_constraints(self, W, b, self.trainer):
    #     tmp = 0
    #     sample_counter = 0
    #     while True:
    #         sample_counter += 1
    #
    #         while True:
    #             structure_list, log_probs, entropies, actions_index = self.trainer.controller_sample()
    #             tmp_action = copy.deepcopy(structure_list[0])
    #             tmp_action[-1] = self.trainer.submodel_manager.n_classes
    #             # print('tmp_action:', tmp_action)
    #             if str(tmp_action) not in self.trainer.total_samples:
    #                 break
    #             else:
    #                 print('arise repeated arch', tmp_action)
    #
    #         mask = dp.encode_arch_to_mask(structure_list)[0]
    #
    #         # rand_arch = self.trainer.propose_gnnnet_mask()
    #         # mask = sum(dp.encode_arch_to_mask(rand_arch), [])
    #
    #         for r_counter in range(0, len(b) ):
    #             left = mask * W[r_counter]
    #             if np.sum(left) < b[r_counter]:
    #
    #                 # if str(mask) not in self.samples:
    #                 #     print("total sampled:", sample_counter)
    #                 #     return structure_list[0], mask, log_probs, entropies, actions_index
    #
    #                 if tmp == len(b)-1 or len(self.samples) <= self.args.init_samples or sample_counter >= 50:
    #                     # print(len(b))
    #                     # print('r_counter:', r_counter)
    #                     # exit()
    #
    #                     self.trainer.total_samples.append(str(tmp_action))
    #                     # print('sampels:', self.trainer.total_samples)
    #
    #                     print("total sampled:", sample_counter )
    #                     return structure_list, mask, log_probs, entropies, actions_index
    #                 tmp += 1
    #
    #             # print("left:", np.sum(left), np.sum(left) <  b[r_counter] )
    #             # print("right:", b[r_counter] )

    def search_samples_under_constraints(self, W, b):
        sample_counter = 0
        while True:
            sample_counter += 1

            while True:
                structure_list, log_probs, entropies, actions_index = self.trainer.controller_sample()
                tmp_action = copy.deepcopy(structure_list[0])
                tmp_action[-1] = self.trainer.submodel_manager.n_classes

                if str(tmp_action) not in self.trainer.total_samples:
                    break
                else:
                    print('Arise a repeated arch', tmp_action)

            mask = dp.encode_arch_to_mask(structure_list)[0]

            tmp = 0
            for r_counter in range(0, len(b) ):
                left = mask * W[r_counter]
                if np.sum(left) < b[r_counter]:
                    if tmp == len(b)-1 or sample_counter >= 50:
                        self.trainer.total_samples.append(str(tmp_action))
                        print("total sampled:", sample_counter )
                        return structure_list, mask, log_probs, entropies, actions_index
                    tmp += 1
                else:
                    break

    def dump_results(self):
        sorted_samples = sorted(self.samples.items(), key=operator.itemgetter(1))
        final_results_str = json.dumps(sorted_samples )
        with open("result.txt", "w") as f:
            f.write(final_results_str + '\n')

    # def run_use_period_predictor(self):
    #     print('Cp:', self.Cp)
    #     flag = 1
    #     while len(self.trainer.total_samples) < self.args.search_samples:
    #         #assemble the training data:
    #         self.populate_training_data()       # clear all nodes info, and put samples into root's bag.
    #
    #         #training the tree
    #         self.train_nodes()                  # training starts from the bag.
    #         self.print_tree()
    #
    #         #select
    #         target_bin, boundaries, starting_point = self.select()
    #         print('sample_node:', target_bin.id)
    #
    #         W, b = self.prepare_boundaries( boundaries )
    #
    #         if len(self.trainer.total_samples) % 100 == 0:
    #             print(20*'-', 'init controller', 20*'-')
    #             self.trainer.init_controller()
    #
    #         if self.args.is_predictor and len(self.trainer.total_samples) >= self.mlp_mg.num_mlp_start and flag == 1:
    #             flag = 0
    #             self.mlp_mg.train_mlp(self.samples)
    #             for i in range(self.times_use_predictor):
    #
    #                 if len(self.trainer.total_samples) % 100 == 0:
    #                     print(20 * '-', 'init controller', 20 * '-')
    #                     self.trainer.init_controller()
    #
    #                 if len(self.trainer.total_samples) > self.args.search_samples:
    #                     break
    #
    #                 while True:
    #                     structure_list, log_probs, entropies, actions_index = self.trainer.controller_sample()
    #                     tmp_action = copy.deepcopy(structure_list[0])
    #                     tmp_action[-1] = self.trainer.submodel_manager.n_classes
    #                     if str(tmp_action) not in self.trainer.total_samples:
    #                         break
    #                     else:
    #                         print('Arise a repeated arch', tmp_action)
    #
    #                 self.trainer.total_samples.append(str(tmp_action))
    #                 np_entropies = entropies.data.cpu().numpy()
    #                 mask = dp.encode_arch_to_mask(structure_list)[0]
    #                 rewards = self.mlp_mg.mlp_pred_acc(mask)
    #
    #                 path = self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file + '_mlp' + '.txt'
    #                 self.trainer.submodel_manager.record_action_info(path, structure_list[0], rewards, 0.0001)
    #                 print('pred_acc:{0}'.format(rewards))
    #
    #                 if self.args.entropy_mode == 'reward':
    #                     rewards = rewards + self.args.entropy_coeff * np_entropies
    #                 elif self.args.entropy_mode == 'regularizer':
    #                     rewards = rewards * np.ones_like(np_entropies)
    #                 else:
    #                     raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')
    #
    #                 torch.cuda.empty_cache()
    #
    #                 self.trainer.collect_trajectory(rewards, log_probs, actions_index)
    #                 print(20 * '-', 'episodes: ', self.trainer.buf.get_buf_size(), 20 * '-')
    #
    #                 if self.times_use_predictor < self.args.episodes:
    #                     if self.trainer.buf.get_buf_size() % self.times_use_predictor == 0:
    #                         print("update agent with the data that are obtained by the predictor!")
    #                         self.trainer.train_controller_by_ppo2()
    #                 else:
    #                     if self.trainer.buf.get_buf_size() % self.args.episodes == 0:
    #                         print("update agent with the data that are obtained by the true env.")
    #                         self.trainer.train_controller_by_ppo2()
    #         else:
    #             flag = 1
    #             for i in range(self.select_num):
    #
    #                 if len(self.trainer.total_samples) > self.args.search_samples:
    #                     break
    #
    #                 print(10 * '#', 'search_samples_under_constraints', 10 * '#')
    #                 sampled_result = {}
    #
    #                 structure_list, sampled_arch_mask, \
    #                 log_probs, entropies, actions_index = self.search_samples_under_constraints(W, b)
    #
    #                 # calculate reward
    #                 np_entropies = entropies.data.cpu().numpy()
    #
    #                 rewards, sampled_result[json.dumps(sampled_arch_mask)] = self.trainer.get_reward(structure_list, np_entropies)
    #                 torch.cuda.empty_cache()
    #
    #                 self.trainer.collect_trajectory(rewards, log_probs, actions_index)
    #                 print(20 * '-', 'episodes: ', self.trainer.buf.get_buf_size(), 20 * '-')
    #
    #                 if self.trainer.buf.get_buf_size() % self.args.episodes == 0:
    #                     print("update agent with the data that are obtained by the true env.")
    #                     self.trainer.train_controller_by_ppo2()
    #
    #                 self.collect_samples( sampled_result )

    def run(self, arch_train_info_trace):
        print('Cp:', self.Cp)
        flag = 0
        train_mcts_time_cost = 0
        while len(self.trainer.total_samples) < self.args.search_samples:
            m_start_time = time.time()
            #assemble the training data:
            self.populate_training_data()       # clear all nodes info, and put samples into root's bag.

            #training the tree
            self.train_nodes()                  # training starts from the bag.
            self.print_tree()

            #select
            target_bin, boundaries, starting_point = self.select()
            print('sample_node:', target_bin.id)

            W, b = self.prepare_boundaries( boundaries )

            m_end_time = time.time()

            train_mcts_time_cost += (m_end_time - m_start_time)

            if len(self.trainer.total_samples) % 100 == 0:
                print(20*'-', 'init controller', 20*'-')
                self.trainer.init_controller()

            if self.args.is_predictor and len(self.trainer.total_samples) >= self.mlp_mg.num_mlp_start and flag == 1:
                flag = 0
                self.mlp_mg.train_mlp(self.samples)
                for i in range(self.select_num):

                    while True:
                        structure_list, log_probs, entropies, actions_index = self.trainer.controller_sample()
                        tmp_action = copy.deepcopy(structure_list[0])
                        tmp_action[-1] = self.trainer.submodel_manager.n_classes
                        if str(tmp_action) not in self.trainer.total_samples:
                            break
                        else:
                            print('Arise a repeated arch', tmp_action)

                    self.trainer.total_samples.append(str(tmp_action))
                    np_entropies = entropies.data.cpu().numpy()
                    mask = dp.encode_arch_to_mask(structure_list)[0]
                    rewards = self.mlp_mg.mlp_pred_acc(mask)

                    path = self.args.log_output_dir + '/' + 'rand_seed' + str(self.args.random_seed) + '_predictor' + '.txt'
                    self.trainer.submodel_manager.record_action_info(path, structure_list[0], rewards, 0.0001)
                    print('pred_acc:{0}'.format(rewards))

                    if self.args.entropy_mode == 'reward':
                        rewards = rewards + self.args.entropy_coeff * np_entropies
                    elif self.args.entropy_mode == 'regularizer':
                        rewards = rewards * np.ones_like(np_entropies)
                    else:
                        raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

                    torch.cuda.empty_cache()

                    self.trainer.collect_trajectory(rewards, log_probs, actions_index)
                    print(20 * '-', 'episodes: ', self.trainer.buf.get_buf_size(), 20 * '-')

                    if self.trainer.buf.get_buf_size() % self.args.episodes == 0:
                        self.trainer.train_controller_by_ppo2()
            else:
                flag = 1
                for i in range(self.select_num):
                    print(10 * '#', 'search_samples_under_constraints', 10 * '#')
                    sampled_result = {}

                    structure_list, sampled_arch_mask, \
                    log_probs, entropies, actions_index = self.search_samples_under_constraints(W, b)

                    # calculate reward
                    np_entropies = entropies.data.cpu().numpy()

                    start_time = time.time()
                    rewards, val_acc = self.trainer.get_reward(structure_list, np_entropies)
                    sampled_result[json.dumps(sampled_arch_mask)] = val_acc
                    end_time = time.time()
                    arch_train_info_trace.append((structure_list, val_acc, end_time - start_time))

                    # rewards, sampled_result[json.dumps(sampled_arch_mask)] = self.trainer.get_reward(structure_list, np_entropies)
                    torch.cuda.empty_cache()

                    self.trainer.collect_trajectory(rewards, log_probs, actions_index)
                    print(20 * '-', 'episodes: ', self.trainer.buf.get_buf_size(), 20 * '-')

                    if self.trainer.buf.get_buf_size() % self.args.episodes == 0:
                        self.trainer.train_controller_by_ppo2()

                    self.collect_samples( sampled_result )

        return train_mcts_time_cost


