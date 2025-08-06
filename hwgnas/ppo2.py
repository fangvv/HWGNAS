
import torch
import numpy as np


class PPOBuffer:

    def __init__(self, size):
        # self.state_buf = []
        self.actions_index_buf = []
        self.log_old_p_buf = []
        self.adv_buf = []
        self.ptr, self.max_size = 0, size

    def store(self, action, logp, adv):
        if self.ptr == 0:
            # self.state_buf.clear()
            self.actions_index_buf.clear()
            self.log_old_p_buf.clear()
            self.adv_buf.clear()
        # self.state_buf.append(state)
        self.actions_index_buf.append(action)
        self.log_old_p_buf.append(logp)
        self.adv_buf.append(adv)
        self.ptr += 1

    def get_v2(self):
        buffer_size = self.ptr
        self.ptr = 0
        data = dict(log_old_p=list(self.log_old_p_buf), actions_index=list(self.actions_index_buf),
                    adv=list(self.adv_buf))
        return buffer_size, {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def get(self):
        if self.ptr >= self.max_size:
            self.ptr = 0

        data = dict(log_old_p=list(self.log_old_p_buf), actions_index=list(self.actions_index_buf),
                    adv=list(self.adv_buf))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def get_buf_size(self):
        return len(self.actions_index_buf)

    def flatten(self, lst):
        for each in lst:
            if not isinstance(each, list):
                yield each
            else:
                yield from self.flatten(each)

