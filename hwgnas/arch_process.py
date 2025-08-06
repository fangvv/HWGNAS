import itertools
import numpy as np
from hwgnas.search_space import MacroSearchSpace
import random


def arch_info(layers):

    global search_space
    global action_list

    search_space_cls = MacroSearchSpace()
    search_space = search_space_cls.get_search_space()
    action_list = search_space_cls.generate_action_list(num_of_layers=layers)

    init_point = []
    tmp = []
    for action in action_list:
        tmp.append(search_space[action])

    for i in range(0, len(sum(tmp, []))):
        if random.random() >= 0.5:
            init_point.append(0.0)
        else:
            init_point.append(1.0)

    return {'init_point': init_point, 'mask_len': len(init_point)}


def encode_arch_to_mask(archs):
    mask_search_space = []
    for single_arch in archs:
        tmp = []
        for i, value, action in zip(np.arange(0, len(action_list)), single_arch, action_list):
            tmp2 = np.zeros(len(search_space[action]))

            # replace the last action with task label.
            if i != len(action_list) - 1:
                tmp2[search_space[action].index(value)] = 1

            tmp.append(tmp2.tolist())

        mask_search_space.append(list(itertools.chain.from_iterable(tmp)))

    return mask_search_space


def decode_mask_to_arch(masks):
    gnn_arch = []
    for single_mask in masks:
        start, end = 0, 0
        tmp = []
        for action in action_list:
            start = 0 + end
            end += len(search_space[action])
            tmp.append(search_space[action][single_mask[start: end].index(1.0)])
        gnn_arch.append(list(itertools.chain.from_iterable(tmp)))

    return gnn_arch



