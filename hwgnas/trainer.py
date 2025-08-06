import glob
import os

import numpy as np
import scipy.signal
import torch

import hwgnas.utils.tensor_utils as utils
from hwgnas.gnn_model_manager import CitationGNNManager
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager

from hwgnas.ppo2 import PPOBuffer
from hwgnas.mlp import MLP

logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0

        self.max_length = self.args.shared_rnn_max_length

        self.with_retrain = False
        self.submodel_manager = None
        self.controller = None

        self.build_model()  # build controller and sub-model
        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        self.baseline = None
        self.adv_history = []
        self.entropy_history = []
        self.reward_history = []
        self.buf = PPOBuffer(args.buf_size)

        self.total_samples = []

        if self.args.mode == "derive":
            self.load_model()

    def build_model(self):
        self.args.share_param = False
        self.with_retrain = True
        self.args.shared_initial_step = 0
        if self.args.search_mode == "macro":
            # generate model description in macro way (generate entire network description)
            from hwgnas.search_space import MacroSearchSpace
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            # build RNN controller
            from hwgnas.graphnas_controller import SimpleNASController
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)

            if self.args.dataset in ["cora", "citeseer", "pubmed"]:
                # implements based on dgl
                self.submodel_manager = CitationGNNManager(self.args)
            if self.args.dataset in ["Cora", "Citeseer", "Pubmed", "Computers", "Photo","SIoT"]:
                # implements based on pyg
                self.submodel_manager = GeoCitationManager(self.args)

        if self.args.search_mode == "micro":
            self.args.format = "micro"
            self.args.predict_hyper = True
            if not hasattr(self.args, "num_of_cell"):
                self.args.num_of_cell = 2
            from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace
            search_space_cls = IncrementSearchSpace()
            search_space = search_space_cls.get_search_space()
            from hwgnas.graphnas_controller import SimpleNASController
            from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
            self.submodel_manager = MicroCitationManager(self.args)
            self.search_space = search_space
            action_list = search_space_cls.generate_action_list(cell=self.args.num_of_cell)
            if hasattr(self.args, "predict_hyper") and self.args.predict_hyper:
                self.action_list = action_list + ["learning_rate", "dropout", "weight_decay", "hidden_unit"]
            else:
                self.action_list = action_list
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)
            if self.cuda:
                self.controller.cuda()

        if self.cuda:
            self.controller.cuda()

    def form_gnn_info(self, gnn):
        if self.args.search_mode == "micro":
            actual_action = {}
            if self.args.predict_hyper:
                actual_action["action"] = gnn[:-4]
                actual_action["hyper_param"] = gnn[-4:]
            else:
                actual_action["action"] = gnn
                actual_action["hyper_param"] = [0.005, 0.8, 5e-5, 128]
            return actual_action
        return gnn

    def _construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            print('single_action: ', single_action)
            for action, action_name in zip(single_action, self.action_list):
                predicted_actions = self.search_space[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def _generate_random_individual(self):
        ind = []
        for action in self.action_list:
            ind.append(np.random.randint(0,
                                         len(self.search_space[action])))
        return ind

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters of the child hwgnas
            self.train_shared(max_step=self.args.shared_initial_step)
            # 2. Training the controller parameters theta
            self.train_controller()
            # 3. Derive architectures
            self.derive(sample_num=self.args.derive_num_sample)

            if self.epoch % self.args.save_epoch == 0:
                self.save_model()

        if self.args.derive_finally:
            best_actions = self.derive()
            print("best structure:" + str(best_actions))
        self.save_model()

    def train_shared(self, max_step=50, gnn_list=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().

        """
        if max_step == 0:  # no train shared
            return

        print("*" * 35, "training model", "*" * 35)
        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)

        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            try:
                _, val_score = self.submodel_manager.train(gnn, format=self.args.format)
                logger.info(f"{gnn}, val_score:{val_score}")
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e

        print("*" * 35, "training over", "*" * 35)

    def get_reward(self, gnn_list, entropies, skip_flag=0):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                           with_retrain=self.with_retrain,skip_flag=skip_flag)

            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                reward = reward[1]

            reward_list.append(reward)

        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, reward

    def init_controller(self):
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        self.baseline = None
        self.adv_history = []
        self.entropy_history = []
        self.reward_history = []

    def controller_sample(self):
        structure_list, log_probs, entropies, actions_index = self.controller.sample(with_details=True)
        return structure_list, log_probs, entropies, actions_index

    def collect_trajectory(self, rewards, log_probs, actions_index):

        # discount
        if 1 > self.args.discount > 0:
            rewards = discount(rewards, self.args.discount)
        # moving average baseline
        if self.baseline is None:
            self.baseline = rewards
        else:
            decay = self.args.ema_baseline_decay
            self.baseline = decay * self.baseline + (1 - decay) * rewards

        adv = rewards - self.baseline
        history.append(adv)
        adv = scale(adv, scale_value=0.5)
        actions_index = actions_index.cpu().detach().numpy().tolist()[0]
        log_old_p = log_probs.cpu().detach().numpy().tolist()

        self.buf.store(actions_index, log_old_p, adv)

    def train_controller_by_ppo2(self):
        print(20 * '-' + "Training with PPO2, the iteration steps:{0}".format(self.args.ppo_epochs) + 20 * '-')
        # buffer_size, data = self.buf.get()
        buffer_size, data = self.buf.get_v2()
        if self.args.cuda:
            actions_index, log_old_p, adv = data['actions_index'].long().cuda(), data['log_old_p'].cuda(), data['adv'].cuda()

        # ppo2 policy loss
        for i in range(self.args.ppo_epochs):
            print(20 * '-' + 'ppo update steps: ', i, 20 * '-')
            total_loss = 0
            for k in range(buffer_size):
                log_p, entropies = self.controller.get_logp_entropies(actions_index[k])

                ratio = torch.exp(log_p - log_old_p[k])
                clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv[k]
                loss = -(torch.min(ratio * adv[k], clip_adv))

                if self.args.entropy_mode == 'regularizer':
                    loss -= self.args.entropy_coeff * entropies

                total_loss += loss.sum()

            total_loss = total_loss / self.args.episodes
            print('ppo_total_loss:', total_loss)
            # update
            self.controller_optim.zero_grad()
            total_loss.backward()

            self.controller_optim.step()

        # step += episodes
        torch.cuda.empty_cache()

    def train_controller(self, log_probs, entropies, reward):

        np_entropies = entropies.data.cpu().numpy()
        if self.args.entropy_mode == 'reward':
            rewards = reward + self.args.entropy_coeff *  np_entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward * np.ones_like(np_entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        # discount
        if 1 > self.args.discount > 0:
            rewards = discount(rewards, self.args.discount)

        self.reward_history.extend(rewards)
        self.entropy_history.extend(np_entropies)

        # moving average baseline
        if self.baseline is None:
            self.baseline = rewards
        else:
            decay = self.args.ema_baseline_decay
            self.baseline = decay * self.baseline + (1 - decay) * rewards

        adv = rewards - self.baseline
        history.append(adv)

        adv = scale(adv, scale_value=0.5)
        self.adv_history.extend(adv)

        adv = utils.get_variable(adv, self.cuda, requires_grad=False)
        # policy loss
        loss = -log_probs * adv
        if self.args.entropy_mode == 'regularizer':
            loss -= self.args.entropy_coeff * entropies

        loss = loss.sum()  # or loss.mean()

        # update
        self.controller_optim.zero_grad()
        loss.backward()

        self.controller_optim.step()

        torch.cuda.empty_cache()


    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        gnn = self.form_gnn_info(gnn)
        results = self.submodel_manager.retrain(gnn, format=self.args.format)
        if results:
            reward, scores = results
        else:
            return

        logger.info(f'eval | {gnn} | reward: {reward:8.2f} | scores: {scores:8.2f}')

    def derive_from_history(self, path, top=10):
        top_arch = []
        actions_list = []
        top_p = 0

        path_top_k=path+'_topK.txt'
        with open(path_top_k, "a") as file:              
            file.write("actions;val_acc\n")

        if self.args.is_predictor:
            top_p = top
            path = path  + '.txt'

            with open(path, "r") as f:
                lines_mlp = f.readlines()

            results = []

            for line in lines_mlp:
                if line[0] != '[':
                    continue

                actions = line[:line.index(";")]
                val_score = line.split(";")[-2]

                if actions in actions_list:
                    print('exist a repeated arch:', actions)
                else:
                    results.append((actions, val_score))

            results.sort(key=lambda x: x[-1], reverse=True)
            for i in results[:top_p]:
                top_arch.append(i)

        else:
            path = path + '.txt'
            with open(path, "r") as f:
                lines = f.readlines()

            results = []

            for line in lines:
                if line[0] != '[':
                    continue

                actions = line[:line.index(";")]
                val_score = line.split(";")[-2]

                if actions in actions_list:
                    print('exist a repeated arch:', actions)
                else:
                    results.append((actions, val_score))

            top_t = top - top_p
            print("top_t:", top_t, " top:", top, " top_p:", top_p)
            print("results:", results)
            results.sort(key=lambda x: x[-1], reverse=True)
            for i in results[:top_t]:
                top_arch.append(i)

        print(top_arch)
        best_score = 0
        for actions in top_arch:
            print(20 * '-')
            print('actions:', actions[0], 'val_acc:', actions[1])
            print(20 * '-')
            actions = eval(actions[0])

            np.random.seed(self.args.random_seed)
            torch.manual_seed(self.args.random_seed)
            torch.cuda.manual_seed_all(self.args.random_seed)

            val_scores_list = []
            for i in range(20):
                val_acc, test_acc = self.submodel_manager.evaluate(actions)
                val_scores_list.append(val_acc)

            tmp_score = np.mean(val_scores_list)
            with open(path_top_k, "a") as file:              
                file.write(str(actions))
                file.write(";")
                file.write(str(tmp_score))                
                file.write("\n")

            if tmp_score > best_score:
                best_score = tmp_score
                best_structure = actions

        print(20 * '-')
        print("best structure:" + str(best_structure), 'best_score:', best_score)
        print(20 * '-')
        # train from scratch to get the final score
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        test_scores_list = []
        if self.args.dataset == 'Photo':
            repeated_time = 20
        else:
            repeated_time = 100

        for i in range(repeated_time):
            print(20 * '-', i, 20 * '-')
            if self.args.dataset == 'Photo':
                self.submodel_manager.shuffle_data()
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print(f"best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")

        # store the best_actions of meta_test tasks
        with open(path, "a") as file:
            file.write('top {0}'.format(top))
            file.write(";")

            file.write('BEST RESULTS:')
            file.write(str(best_structure))

            file.write(";")
            file.write(str(np.mean(test_scores_list)))

            file.write("+/-")
            file.write(str(np.std(test_scores_list)))
            file.write("\n")
        with open(path_top_k, "a") as file:
            print("最佳gnn保存路径：{}".format(path))
            file.write('top {0}'.format(top))
            file.write(";")

            file.write('BEST RESULTS:')
            file.write(str(best_structure))

            file.write(";")
            file.write(str(np.mean(test_scores_list)))

            file.write("+/-")
            file.write(str(np.std(test_scores_list)))
            file.write("\n")
        return np.mean(test_scores_list)

    def derive(self, sample_num=None, path=None, top=None,skip_flag=0):
        """
        sample a serial of structures, and return the best structure.
        """
        if sample_num is None and self.args.derive_from_history:
            return self.derive_from_history(path=path, top=top, slices=slices)
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample

            gnn_list, _, entropies, _ = self.controller.sample(sample_num, with_details=True)
            # samples = {}

            max_R = 0
            best_actions = None
            filename = self.model_info_filename
            for action in gnn_list:

                gnn = self.form_gnn_info(action)
                reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                               with_retrain=self.with_retrain,skip_flag=skip_flag)

                if reward is None:  # cuda error hanppened
                    continue
                else:
                    results = reward[1]

                if results > max_R:
                    max_R = results
                    best_actions = action

            logger.info(f'derive |action:{best_actions} |max_R: {max_R:8.6f}')
            self.evaluate(best_actions)
            return best_actions

    @property
    def model_info_filename(self):
        return f"{self.args.dataset}_{self.args.search_mode}_{self.args.format}_results.txt"

    @property
    def controller_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.args.dataset}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        logger.info(f'[*] LOADED: {self.controller_path}')
