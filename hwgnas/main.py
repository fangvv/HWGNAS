"""Entry point."""
import sys

sys.path.append('/mnt/compare-search-time/training-space-limit-acc-latency/')
print(sys.path)
import argparse
import time
import torch
import hwgnas.trainer as trainer
import random
import numpy as np
import hwgnas.search_strategy as search_strategy_manager
import os


def build_args():
    parser = argparse.ArgumentParser(description='HWGNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training HWNAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--top_k', type=int, default=10)  # 10
    parser.add_argument('--search_samples', type=int, default=2000)  # 2000

    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=100,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=100)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,  # 300
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
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

    # MCTS
    parser.add_argument('--search_strategy', type=str, default="PPO")
    parser.add_argument('--init_samples', type=int, default=200)  # 200
    parser.add_argument('--select_samples', type=int, default=20)  # 20
    parser.add_argument('--Cp', type=float, default=0.1)
    parser.add_argument('--tree_height', type=int, default=5)

    # Predictor
    parser.add_argument('--is_predictor', type=int, default=1)
    parser.add_argument('--predictor_lr', type=float, default=0.0001)

    # 训练BRP
    parser.add_argument('--latency_predictor_epochs', type=int, default=40)

    parser.add_argument('--predictor_batch_size', type=int, default=100)  # 100
    parser.add_argument('--predictor_init_samples', type=int, default=400)  # 400
    parser.add_argument('--times_use_predictor', type=int, default=20)

    # PPO2
    parser.add_argument('--buf_size', type=int, default=20)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--ppo_epochs', type=int, default=10)

    # Output
    parser.add_argument('--log_output_dir', type=str, default='./RTX3090/log_α-β')

    # 是否使用注意力机制
    parser.add_argument('--use_curiosity', type=bool, default=False)
    # 是否扩展搜索空间
    parser.add_argument('--use_space', type=bool, default=False)

    # 精度预测器
    parser.add_argument('--acc_predictor_type', type=str, default='brp')
    parser.add_argument('--acc_input_dim', type=int, default=17)  # 构造的节点特征维度
    parser.add_argument('--acc_embedding_dim', type=int, default=128)
    parser.add_argument('--acc_hidden_dim', type=int, default=256)
    parser.add_argument('--acc_predictor_layer_nums', type=int, default=3)  # 预测器层数

    # 延迟预测器相关
    parser.add_argument('--latency_weight', type=int, default=100)  # 将延迟奖励乘以该数，因为RTX3090的延迟在0.00量级
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--latency_epochs', type=int, default=30)
    parser.add_argument('--latency_predictor_type', type=str, default='brp_RTX3090_Citeseer')

    parser.add_argument('--latency_limit', type=float, default=0.2)
    parser.add_argument('--use_latency_limit', type=bool, default=True)

    # 搜索空间有关
    parser.add_argument('--num_state', type=int, default=5)
    # 是否生成跳跃连接
    parser.add_argument('--use_skip_connection', type=bool, default=False)


def search(args, log_path):
    print(f'search函数中args.latency_predictor_type={args.latency_predictor_type}')
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    trnr = trainer.Trainer(args)

    arch_train_info_trace = []

    start_time = time.time()
    mcts_time_costs = 0

    # start to search
    if args.search_strategy == 'PPO+MCTS':
        mcts_time_costs = search_strategy_manager.mcts_ppo(args, trnr, arch_train_info_trace)
    elif args.search_strategy == 'PPO':
        search_strategy_manager.ppo(args, trnr)
    elif args.search_strategy == 'PG':
        search_strategy_manager.pg(args, trnr)

    end_time = time.time()
    search_time = end_time - start_time + mcts_time_costs
    print('Total elapsed time: ' + str(search_time))
    path = args.log_output_dir + '/' + 'rand_seed' + str(args.random_seed) + '.txt'
    with open(path, "a") as file:

        file.write('Total elapsed time:')
        file.write(str(search_time))
        file.write("\n")

        file.write('train_mcts_time_costs:')
        file.write(str(mcts_time_costs))
        file.write("\n")

    path = args.log_output_dir + '/' + 'rand_seed' + str(args.random_seed)
    # derive
    avg_best_test = trnr.derive_from_history(path=path, top=args.top_k)

    return avg_best_test, search_time


def main(args):
    if args.is_predictor == 1:
        log_path = args.log_output_dir + '/' + args.dataset + "_predictor"

    log_path = log_path + "_alpha" + str(args.alpha) + "_weight" + str(args.latency_weight)
    if args.use_latency_limit:
        log_path = log_path + "_limit" + str(args.latency_limit)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    args.log_output_dir = log_path

    avg_test_acc = []
    avg_search_time = []
    seed = [123]
    print(seed)
    if args.mode == 'train':
        for s in seed:
            print('Random seed:', s)
            args.random_seed = s
            args.submanager_log_file = f"sub_manager_logger_file_{time.time()}"
            path = args.log_output_dir + '/' + 'rand_seed' + str(s) + '.txt'
            with open(path, "w") as file:
                file.write(str(args))
                file.write("\n")

            test_acc, search_time = search(args, log_path)
            avg_test_acc.append(test_acc)
            avg_search_time.append(search_time)

    elif args.mode == 'derive':
        if not os.path.isdir(args.log_output_dir):
            raise ("Invalid log_output_dir: ", args.log_output_dir)

        trnr = trainer.Trainer(args)
        search_time = 0.0
        # for path in args.log_output_dir:
        for s in seed:
            path = args.log_output_dir + '/' + 'rand_seed' + str(s)
            best_test = trnr.derive_from_history(path=path, top=args.top_k)
            avg_test_acc.append(best_test)
            avg_search_time.append(search_time)

    path = args.log_output_dir + '/' + "avg_results" + '.txt'
    with open(path, "w") as file:
        file.write('top {0}'.format(args.top_k))
        file.write("\n")

        file.write('avg_test_mean:')
        file.write(str(np.mean(avg_test_acc)))
        file.write("\n")

        file.write('avg_test_std:')
        file.write(str(np.std(avg_test_acc)))
        file.write("\n")

        file.write('avg_search time:')
        file.write(str(np.mean(avg_search_time)))
        file.write("\n")


if __name__ == '__main__':
    args = build_args()
    print(args)
    print(f'args.latency_predictor_type={args.latency_predictor_type}')
    main(args)
    os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")



