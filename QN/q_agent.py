"""
This script shows the QN agent related functions and wrappers.
"""
import itertools
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from fastrl_training.utils.utils import Transition
from fastrl_training.utils.wrapper import ModelWrapperInterface


class QWrapper(ModelWrapperInterface):

    def __init__(self, model, env, memory, timesteps_observed=1, use_gpu=True, sumo_wrapper=False,
                 batch_size=128, target_after=10, gamma=0.99, eps_start=0.9, eps_stop=0.05, eps_decay=200,
                 evaluate=False, **kwargs
                 ):
        """
        This init creates the wrapper of the DQN agent.
        :param model: represents a neural network with policy and target submodules.
        :param env: an OpenAI gym environment
        :param memory: represents a memory replay where we collect experience
        :param timesteps_observed: gives us estimation how many timesteps to observe (depends on the model )
        :param use_gpu: if we want to use the GPU
        :param sumo_wrapper: defines if the wrapper wraps a sumo env (True) or a normal gym env (False)
        :param batch_size: defines the batch size of the parameter update
        :param target_after: defines after how many updates do we want to copy the weights to the target network
        :param gamma: how big the discount factor should be.
        :param eps_start: defines the epsilon start value
        :param eps_stop: defines the epsilon stop value
        :param eps_decay: defines the epsilon decay value
        """
        super(QWrapper, self).__init__()
        self.name = f"Qnetwork_{model.name}_{env.name}_{env.type_as}"
        self.env = env
        self.sumo_wrapper = sumo_wrapper
        self.use_gpu = use_gpu if torch.cuda.is_available() else False
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        self.model = model.to(device=self.device)
        self.memory = memory
        self.batch_size = batch_size
        self.target_after = target_after
        self.timesteps_observed = timesteps_observed  # Defines how many timesteps to feed for the network
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_stop = eps_stop
        self.eps_decay = eps_decay

        self.current_update = 0
        self.steps_done = 0
        self.reward_episode = []

    def one_episode(self, log_path=None):
        """
        This function runs one episode and returns the information about it.
        :return: Dict containing the following data:
        'steps': t + 1,
        'reward': episode_reward,
        'success': 1 if info['cause'] is None else 0,
        'speed': sum(running_speed) / len(running_speed),
        'lane_change': info['lane_change'],
        'distance': info["distance"],
        'type': self.env.simulation_list.index(self.env.sumoCmd[2]),
        """
        # Init logging values
        done, error_running_traci = False, False
        t = 0
        episode_reward = 0
        info = None
        eps = 0
        running_speed = 0 if self.sumo_wrapper else None
        # this while is needed due to unhandled sumo errors
        while not done and not error_running_traci:
            # resetting env
            state_list = [self.env.reset()]
            action_list = []
            episode_reward = 0
            running_speed = [] if self.sumo_wrapper else None
            t = 0
            info = 0
            error_running_traci = False
            # going through the episode step-by-step
            for t in itertools.count():
                # creating the state for the input. Usually 1 timestep but for sumo it may be more.
                current_state = np.stack(state_list[-self.timesteps_observed:])
                # Selecting action based on current state
                action, eps = self.forward(current_state)
                # Step through environment using chosen action
                # try catch is due to the unhandled sumo errors.
                try:
                    next_state, reward, done, info = self.env.step(action.item())
                    # recalculating reward based on the current policy
                    # reward = torch.matmul(torch.tensor(info['cumulants']), self.w[self.j].cpu()).item()
                    # reward_ = info['cumulants']
                except RuntimeError as err:
                    self.env.reset()
                    self.reward_episode = []
                    error_running_traci = True
                    break
                # adding state to the history
                state_list.append(next_state)
                action_list.append(action.item())

                if self.sumo_wrapper:
                    running_speed.append(info['velocity'])
                # Save reward
                self.reward_episode.append(reward)
                # Store the transition in memory
                self.memory.push(state_list[-2], action, next_state if not done else None, reward)
                # Printing to console if episode is terminated
                if done:
                    episode_reward = sum(self.reward_episode)
                    print(f"Steps:{t + 1}, reward: {episode_reward:.3f},"
                          f"cause: {info.get('cause', None)}, "
                          f"distance:{info.get('distance', 0):.2f}")
                    break
            if log_path is not None:
                if not os.path.exists(os.path.dirname(log_path)):
                    os.makedirs(os.path.dirname(log_path))
                with open(log_path, "bw") as file:
                    log_dict = {"state": [s.tolist() for s in state_list], "action": action_list,
                                "reward": self.reward_episode, "success": 1 if info.get('cause', 0) is None else 0,
                                "cause": info.get('cause', "nan")}
                    pickle.dump(log_dict, file)
        self.reward_episode = []

        # Used for tensorboard visualisation. Must contain steps, reward, success
        episode_info = {
            'steps': t + 1,
            'reward': episode_reward,
            'success': 1 if info.get('cause', 0) is None else 0,
            'eps': eps
        }

        if self.sumo_wrapper:
            episode_info.update({'speed': sum(running_speed) / len(running_speed),
                                 'lane_change': info['lane_change'],
                                 'distance': info["distance"],
                                 'type': self.env.simulation_list.index(self.env.sumoCmd[2])})

        return episode_info

    def update(self, optimizer):
        """
        This function solves the network parameter update of the agent.
        :param optimizer: an optimizer that we use to update the parameters.
        :return: loss of the update
        """

        if len(self.memory) < self.batch_size:
            return {"loss": None}
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state
                                              if s is not None], device=self.device, dtype=torch.float32)
        state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long).unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # Gradcontol but currently experimenting without it.
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()

        self.current_update += 1

        return {"loss": loss.detach().cpu() if loss.is_cuda else loss.detach()}

    def forward(self, state_):
        """
        This method makes the forward pass of the Q wrapper. It selects an action.
        :param state_: State to predict from
        :return: Chosen action based on input
        """

        state_ = torch.tensor(state_, dtype=torch.float, requires_grad=True, device=self.device)
        sample = random.random()
        eps_threshold = self.eps_stop + (self.eps_start - self.eps_stop) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        self.model.eval()
        with torch.no_grad():
            pred_action = self.model(state_)

            if sample > eps_threshold:
                # batch x actions
                action_ = pred_action.max(-1)[1]
            else:
                # Creating categorical distribution based on the output of the network
                c = Categorical(F.softmax(pred_action, dim=-1))
                # Selecting action
                action_ = torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.device,
                                       dtype=torch.long)
            self.model.train()

        return action_.detach().cpu() if action_.is_cuda else action_.detach(), eps_threshold

    def eval(self):
        """
        Function to set the model to eval mode
        """
        self.model.eval()

    def train(self):
        """
        Function to set the model to train mode.
        """
        self.model.train()
