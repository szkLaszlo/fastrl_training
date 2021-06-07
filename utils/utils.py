"""
This file contains the common utils for the models.
"""
import glob
import os
import random
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import torch

from fastrl_training.utils.models import SimpleMLP

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Transitionv1 = namedtuple('Transition',
                          ('state', 'action', 'next_state', 'reward', 'index'))


class ReplayMemory(object):
    """
    Class for Replay Memory for the DQN agents.
    """

    def __init__(self, capacity, trans=Transition):
        """
        Initializer for the class.
        :param capacity: defines how many transitions to keep in the memory.
        """
        self.capacity = capacity
        self.trans = trans
        self.memory = []

    def push(self, *args):
        """
        Saves a transition to the replay memory.
        :param args: should contain state, reward, done, info
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(self.trans(*args))

    def sample(self, batch_size):
        """
        The function returns a batch_size of the saved transitions.
        :param batch_size: defines how many samples we want.
        :return: batch of transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Gets the length of the memory replay.
        :return: length of the memory
        """
        return len(self.memory)


def generate_video_from_images(img_dir_path,
                               video_name="video.avi", img_extention="*.jpg",
                               frame_rate=10, delete_images=True, scale_percent=1):
    img_array = []
    files = glob.glob(f'{os.path.join(img_dir_path, img_extention)}')
    files.sort(key=os.path.getmtime)

    for filename in files:
        img = cv2.imread(filename)
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        img_array.append(img)
        if delete_images:
            os.remove(filename)

    if files.__len__() > frame_rate:
        out = cv2.VideoWriter(os.path.join(img_dir_path, video_name), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              frame_rate, dim)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def select_optimizer(model, optimizer_, learning_rate=0.0001, weight_decay=0.01):
    """
    This function selects the appropriate optimizer based on the argparser input.
    :param: args_: Contains learning rate and weight_decay attributes
    :param: model: is a torch.nn.Module with trainable parameters.
    :return: the selected optimizer
    """
    # Creating optimizer
    if optimizer_.lower() == 'asgd':
        optimizer = torch.optim.ASGD(params=model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        print('ASGD optimizer is used')
    elif optimizer_.lower() == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay,
                                     amsgrad=True)
        print('Adam optimizer is used')
    elif optimizer_.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
        print('AdamW optimizer is used')
    elif optimizer_.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
        print('SGD optimizer is used')
    elif optimizer_.lower() == 'sparseadam':
        optimizer = torch.optim.SparseAdam(params=model.parameters(), lr=learning_rate)
        print('SparseAdam optimizer is used')
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        print('Adam optimizer is used')
    return optimizer


def plot_grad_flow(named_parameters):
    """
    This function was used to plot gradient flow
    Parameters
    ----------
    named_parameters: Names of nn.Module parameters

    Returns
    -------

    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def network_grad_plot(model, writer, epoch, name=''):
    """
    Function to plot gradients of networks
    Parameters
    ----------
    model: nn.Module
    writer: SummaryWriter to write to
    epoch: defines the timestep for which this gradient is plotted

    Returns None
    -------

    """
    for name, f in model.named_parameters():
        if hasattr(f.grad, 'data'):
            hist_name = f'{name}/ + {list(f.grad.data.size())}'
            writer.add_histogram(hist_name, f.grad.data, epoch)


def network_weight_plot(model, writer, epoch):
    """
    Function to plot gradients of networks
    Parameters
    ----------
    model: nn.Module
    writer: SummaryWriter to write to
    epoch: defines the timestep for which this gradient is plotted

    Returns None
    -------

    """
    for name, f in model.named_parameters():
        if hasattr(f, 'data') and f.requires_grad:
            writer.add_histogram(name, f.data, epoch)


def find_latest_weight(path='torchSummary', file_end='.weight', exclude_end='.0'):
    """
    Function to search the latest weights in a directory.
    :param path: path to the directory.
    :param file_end: the string which will be used at the end
    :param exclude_end: this text will be excluded during search.
    :return:
    """
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    for i in range(len(time_sorted_list) - 1, -1, -1):
        if not (time_sorted_list[i].endswith(file_end) or os.path.isdir(time_sorted_list[i])):
            del time_sorted_list[i]
            continue
        if len(time_sorted_list) and os.path.isdir(time_sorted_list[i]):
            latest_weight = find_latest_weight(path=time_sorted_list[i], file_end=file_end, exclude_end=exclude_end)
            if latest_weight is not None:
                return latest_weight
            else:
                del time_sorted_list[-1]
                continue
        return time_sorted_list[i]


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        print(name, "has: ", param)
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params


def args_to_hparam_dict(args):
    """
    Function to convert args to hparams for tensorboard
    :param args: argumentparser instance
    :return: dict with compatible types.
    """
    hp_dict = {}
    for hpkey, hpvalue in args.__dict__.items():
        if not isinstance(hpvalue, (str, bool, int, float)):
            hp_dict[hpkey] = str(hpvalue)
        else:
            hp_dict[hpkey] = hpvalue
    return hp_dict


def load_network_from_weights(loaded_weights, name="policy_net"):
    """
    The fuction loads the weights and create the model. Now it requires the linear one.
    :param loaded_weights:
    :param name:
    :return:
    """
    loaded_models = []
    for weight in loaded_weights:
        pre = torch.load(weight)
        keys = [key for key in pre.keys() if name in key]
        keys_ = {}
        for key, value in pre.items():
            if name in key:
                keys_[key.split('.', 1)[1]] = value
        if "linear" in keys[0]:
            in_weight = keys[0]
            out_weight = keys[-1]
            net = SimpleMLP(input_size=pre[in_weight].shape[1],
                            hidden_size=pre[in_weight].shape[0],
                            output_size=pre[out_weight].shape[0]).eval()
            net.load_state_dict(keys_)
            net.training = False
            for param in net.parameters():
                param.requires_grad = False
            loaded_models.append(net)

    return loaded_models
