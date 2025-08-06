import torch
import torch.nn.functional as F
import numpy as np
import json
import numpy as np
import argparse
import hwgnas.arch_process as data_process

class MLP(torch.nn.Module):

    def __init__(self, arch_mask_size):
        '''
        train a mlp with <Arch_mask, val_acc>, and given an Arch_mask derive the val_acc.

        :param args:
        hidden_unit:
        Arch_mask_size:
        output_size:
        mlp_layer_size:
        '''
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(arch_mask_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128,  1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def weights_init(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, mask):
        output = F.relu(self.fc1(mask))
        output = F.relu(self.fc2(output))

        return self.fc3(output)


class MLP_Manager(object):

    def __init__(self, args, arch_mask_len):
        self.num_mlp_start = args.predictor_init_samples
        self.mlp_epochs = args.predictor_epochs
        self.mlp_bs = args.predictor_batch_size
        self.mlp = MLP(arch_mask_len)
        if torch.cuda.is_available():
            self.mlp = self.mlp.cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=args.predictor_lr)

    def train_mlp(self, samples):
        # self.mlp.weights_init()
        self.mlp.train()

        X_sample = None
        Y_sample = None
        for sample in samples:
            if X_sample is None or Y_sample is None:
                X_sample = np.array(json.loads(sample))
                Y_sample = np.array(samples[sample])
            else:
                X_sample = np.vstack([X_sample, json.loads(sample)])
                Y_sample = np.vstack([Y_sample, samples[sample]])

        chunks = int(X_sample.shape[0] / self.mlp_bs)

        print("mlp_dataset_size:", len(samples))
        print('chunk:', chunks)

        if X_sample.shape[0] % self.mlp_bs > 0:
            chunks += 1
        for epoch in range(self.mlp_epochs):
            X_sample_split = np.array_split(X_sample, chunks)
            Y_sample_split = np.array_split(Y_sample, chunks)

            total_loss = 0.0
            for i in range(0, chunks):
                self.optimizer.zero_grad()
                inputs = torch.from_numpy(
                    np.asarray(X_sample_split[i], dtype=np.float32).reshape(X_sample_split[i].shape[0],
                                                                            X_sample_split[i].shape[1]))
                targets = torch.from_numpy(np.asarray(Y_sample_split[i], dtype=np.float32)).reshape(-1, 1)

                # print('input.size:{0}, target.size:{1}'.format(inputs.size(), targets.size()))
                if torch.cuda.is_available():
                    loss = self.loss_fn(self.mlp(inputs.cuda()), targets.cuda())
                else:
                    loss = self.loss_fn(self.mlp(inputs), targets)

                loss.backward()  # back props
                self.optimizer.step()  # update the parameters

                total_loss += loss.item()

            print('mlp_train, epoch:{0}, loss:{1}'.format(epoch, total_loss))

    def mlp_pred_acc(self, arch_mask): 
        self.mlp.eval()

        arch_mask = torch.from_numpy(np.array(arch_mask, dtype=np.float32))

        if torch.cuda.is_available():
            arch_mask = arch_mask.cuda()

        pred_acc = self.mlp(arch_mask)

        return pred_acc.item()

    # def test_mlp(self, samples):
    #     self.mlp.eval()
    #
    #     X_sample = None
    #     Y_sample = None
    #     for sample in samples:
    #         if X_sample is None or Y_sample is None:
    #             X_sample = np.array(json.loads(sample))
    #             Y_sample = np.array(samples[sample])
    #         else:
    #             X_sample = np.vstack([X_sample, json.loads(sample)])
    #             Y_sample = np.vstack([Y_sample, samples[sample]])
    #
    #     inputs = torch.from_numpy(
    #         np.asarray(X_sample, dtype=np.float32).reshape(X_sample_split[i].shape[0],
    #                                                        X_sample_split[i].shape[1]))
    #     targets = torch.from_numpy(np.asarray(Y_sample, dtype=np.float32)).reshape(-1, 1)
    #
    #     pred_acc = self.mlp(inputs)
    #
    #     loss = abs(targets - pred_acc)
    #     print('test_loss:', loss)











