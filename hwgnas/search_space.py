import torch
import torch.nn as nn
import torch.nn.functional as F

class MacroSearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            # Define operators in search space
            self.search_space = {
                "attention_type": ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear'],
                'aggregator_type': ["sum", "mean", "max", "mlp", ],
                'activate_function': ["none","swish","mish","sigmoid", "tanh", "relu", "linear",
                                      "softplus", "leaky_relu", "relu6", "elu"],
                'number_of_heads': [1, 2, 4, 6, 8, 16],
                'hidden_units': [4, 8, 16, 32, 64, 128, 256],
            }

    def get_search_space(self):
        return self.search_space

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    def generate_action_list(self, num_of_layers=2):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list



class Swish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Swish_func.apply(input_tensor)


class Mish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)
        grad_hx = i.sigmoid()
        grad_gx = grad_gh * grad_hx
        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx
        return grad_output * grad_f


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif act == "swish":
        return Swish()
    elif act == "mish":
        return Mish()
    else:
        raise Exception("wrong activate function")

