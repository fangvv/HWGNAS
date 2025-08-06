import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

from egnas.gnn_model_manager import CitationGNNManager, evaluate
from graphnas_variants.macro_graphnas.pyg.pyg_gnn import GraphNet
from egnas.utils.label_split import fix_size_split

from graphnas_variants.macro_graphnas.pyg.dataset_split import AddTrainValTestMask
import os
from graphnas_variants.macro_graphnas.pyg.make_dataset import mydataset

def load_data(dataset="Cora", supervised=False, full_data=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    if dataset in ["CS", "Physics"]:
        dataset = Coauthor(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Computers", "Photo"]:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Amazon(path, dataset, T.NormalizeFeatures())
        create_masks = AddTrainValTestMask(
            split="train_rest",
            num_val=500,
            num_test=500,
        )
        data = create_masks.get_data(dataset[0])
        print(sum(data.train_mask))
        print(sum(data.val_mask))
        print(sum(data.test_mask))

    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
        data = dataset[0]
    elif dataset in ["SIoT"]:
        # 获取当前文件所在的目录
        current_dir = os.path.dirname(__file__)
        print(f'current_dir={current_dir}')
        # 获取上级目录
        parent_dir = os.path.dirname(current_dir)
        print(f'parent_dir={parent_dir}')
        SIoT_file=parent_dir+"/data/SIoT_2000/"
        dataset= mydataset(SIoT_file)
        # data = dataset[0]
        create_masks = AddTrainValTestMask(
            split="train_rest",
            num_val=400,
            num_test=400,
        )
        data = create_masks.get_data(dataset[0])
        print(f'\n\nsum(data.train_mask)={sum(data.train_mask)}')
        print(f'sum(data.val_mask)={sum(data.val_mask)}')
        print(f'sum(data.test_mask)={sum(data.test_mask)}\n\n')
    if supervised:
        if full_data:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:-1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
        else:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
    return data


class GeoCitationManager(CitationGNNManager):
    def __init__(self, args):
        super(GeoCitationManager, self).__init__(args)
        if hasattr(args, "supervised"):
            self.data = load_data(args.dataset, args.supervised)
        else:
            self.data = load_data(args.dataset)
        self.args.in_feats = self.in_feats = self.data.num_features
        self.args.num_class = self.n_classes = self.data.y.max().item() + 1
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.data.to(device)

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False, residual=False)

        # for name, param in model.named_parameters():
        #     print(name, param.size())
        # exit()

        return model

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass

    def shuffle_data(self, full_data=True):
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        if full_data:
            self.data = fix_size_split(self.data, self.data.num_nodes - 1000, 500, 500)
        else:
            self.data = fix_size_split(self.data, 1000, 500, 500)
        self.data.to(device)

    # @staticmethod
    def run_model(self, model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0

        val_acc_last50 = []
        print("Number of train datas:", data.train_mask.sum())
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # evaluate
            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, data.y, data.train_mask)
            dur.append(time.time() - t0)

            val_acc = evaluate(logits, data.y, data.val_mask)
            test_acc = evaluate(logits, data.y, data.test_mask)

            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()


            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc

            # else:                           # as the stage of training, calculate the avg of the last 50 epoch to get an accurate val_acc.
            #     if epoch >= (epochs-50):
            #         val_acc_last50.append(val_acc)

            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))

        print(f"val_score:{model_val_acc},test_score:{best_performance}")

        if return_best:
            # print(f"val_score:{model_val_acc},test_score:{best_performance}")
            return model, model_val_acc, best_performance
        else:
            # model_val_acc = float(round(np.mean(val_acc_last50), 4))
            # print(f"val_score:{model_val_acc}")
            return model, model_val_acc

