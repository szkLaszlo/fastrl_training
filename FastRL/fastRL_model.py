"""
@author "Laszlo Szoke"
"""
import copy
import itertools
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastrl_training.utils.models import SimpleMLP
from fastrl_training.utils.utils import Transitionv1
from fastrl_training.utils.wrapper import ModelWrapperInterface


class FastRLv1(nn.Module):
    """
    Model based on the FastRL article: https://www.pnas.org/content/117/48/30079
    """

    def __init__(self, input_size, actions, d, hidden_size=64, num_policies=6):
        super(FastRLv1, self).__init__()
        self.name = "FastRLv1"

        self.num_policies = num_policies
        # Creating the Successor Features
        sfs = {}
        for i in range(d):
            sfs[f"sf{i}"] = SimpleMLP(input_size=input_size + 1, output_size=num_policies, hidden_size=hidden_size)

        # The policies will be induced by the sf multiplication.
        self.sfs = nn.ModuleDict(sfs)

    def forward(self, input_state, actions):
        """
        Function for model prediction
        :param input_state: [batch x features]
        :param actions: [batch x features]
        :return: model prediction [batch x policy x actions x SF]
        """
        # Collecting SFs
        sf_i = []
        # Extending state with the actions for simplified forward pass
        extended_input = input_state.unsqueeze(1).expand((input_state.size(0), actions.size(1), -1))
        temp_state_action_pairs = torch.cat((extended_input, actions), dim=-1)
        state_action_pairs = torch.cat(temp_state_action_pairs.split(1, dim=1), dim=0).squeeze(1)
        # Calculating all SFs
        for sf in self.sfs.values():
            # batch x actions x 1
            out = sf(state_action_pairs)
            sf_i.append(out)
        # batch x actions x SFs
        sf_pi_j = torch.stack(sf_i, dim=-1)
        sf_pi_j = torch.stack(sf_pi_j.split(input_state.size(0), dim=0), dim=2)

        return sf_pi_j


class FastRLv1TrainerWrapper(ModelWrapperInterface):

    def __init__(self, model, env, memory, w, timesteps_observed=1, use_gpu=True, sumo_wrapper=False,
                 batch_size=128, target_after=10, gamma=0.99, eps_start=0.9, eps_stop=0.05, eps_decay=200,
                 evaluate=False,
                 ):
        """
        This init creates the wrapper of the Fast RL agent.
        :param model: represents a neural network with w.
        :param env: an OpenAI gym environment
        :param memory: represents a memory replay where we collect experience
        :param w: contains the preference vector for the training
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
        super(FastRLv1TrainerWrapper, self).__init__()
        self.name = f"{model.name}_{env.name}_{env.type_as}"
        self.env = env
        self.num_actions = env.action_space.n
        self.sumo_wrapper = sumo_wrapper
        self.use_gpu = use_gpu if torch.cuda.is_available() else False
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        self.model = model.to(device=self.device)
        self.w = torch.tensor(w, dtype=torch.float32, device=self.device, requires_grad=False)
        self.memory = memory
        self.batch_size = batch_size
        self.target_after = target_after
        self.timesteps_observed = timesteps_observed  # Defines how many timesteps to feed for the network
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_stop = eps_stop
        self.eps_decay = eps_decay
        self.evaluate = evaluate
        self.current_update = 0
        self.steps_done = 0
        self.reward_episode = []
        self.j = 0

    def forward(self, state_, j):
        """
        This method makes the forward pass of the FastRL wrapper. It selects an action.
        :param state_: State to predict from
        :param j: selects the actual policy to use
        :return: Chosen action based on input and the selected policy
        """

        state_ = torch.tensor(state_, dtype=torch.float, requires_grad=False, device=self.device)
        sample = random.random()
        eps_threshold = self.eps_stop + (self.eps_start - self.eps_stop) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        self.model.eval()  # Setting model to eval because of batchnorm does not support 1 sample
        with torch.no_grad():
            pred_action, _ = self.process_forward(state_, j)
            # Selecting action or exploring
            if sample > eps_threshold or self.evaluate:
                # batch x actions
                action_ = pred_action.max(-1)[1]
            else:
                action_ = torch.tensor([[random.randrange(pred_action.size(-1))]], device=self.device,
                                       dtype=torch.long)
        self.model.train()

        return action_.detach().cpu() if action_.is_cuda else action_.detach(), eps_threshold

    def process_forward(self, state_, index=None, greedy=False):
        """
        Processes the forward of the FastRL networks.
        :param state_: output of the FastRL. dim: [batch x actions x features]
        :param index: batch of selected policies dim: [batch x policy_index]
        :param greedy: If true the max action will be selected
        :return: batch x actions
        """

        actions = torch.arange(self.env.action_space.n, dtype=state_.dtype, device=self.device).unsqueeze(0).expand(
            (state_.size(0), -1)).unsqueeze(-1)
        actions /= actions.size(-2) - 1
        sfs = self.model(state_, actions)
        # batch x policies x actions<-- Q_w_pi_a
        if getattr(index, "ndim", 0) != 1:
            index = torch.tensor([index], requires_grad=False, device=self.device)

        assert index.ndim == 1
        assert sfs.ndim == 4

        w_prepared = self.w[index].unsqueeze(1).expand((sfs.size(0), sfs.size(1), sfs.size(-1))).unsqueeze(-1)
        q_val = torch.matmul(sfs, w_prepared).squeeze(-1)

        if self.evaluate or greedy:
            # Q_max per actions max over the policies [batch x actions]
            q_max_a = torch.max(q_val, 1).values
        else:
            # batch x policies x actions --> batch x actions
            # if the policies are given, we select those. (useful for next_value selection)
            if index is None:
                return q_val, sfs
            q_max_a = q_val[torch.arange(q_val.size(0)), index]

        # batch x actions and batch x policy x actions x features
        return q_max_a, sfs

    def update(self, optimizer):
        """
        This function solves the network parameter update of the agent.
        :param optimizer: an optimizer that we use to update the parameters.
        :return: loss of the update
        """

        if len(self.memory) < self.batch_size * 2:
            return {"loss": None}

        transitions = self.memory.sample(self.batch_size)
        batch = Transitionv1(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state
                                              if s is not None], device=self.device, dtype=torch.float32)
        state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float32, requires_grad=False)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long, requires_grad=False)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32, requires_grad=False)
        index_batch = torch.tensor(batch.index, device=self.device, dtype=torch.long, requires_grad=False)

        # We calculate the q_max values for the policies, then select the psi values which caused the q_max actions.
        _, psi_t = self.process_forward(state_batch, index_batch)
        psi_t = psi_t[torch.arange(psi_t.size(0)), :, action_batch, ...]

        # The future values are also calculated, then the best actions based on the current policy are calculated.
        # After this, the future psi values are created.
        with torch.no_grad():
            next_psi_t1 = torch.zeros_like(psi_t, device=self.device, dtype=torch.float32, requires_grad=False)
            next_values, psi_t1 = self.process_forward(non_final_next_states, index=index_batch[non_final_mask],
                                                       greedy=False)
            a_ = next_values.max(-1)[1]
            next_psi_t1[non_final_mask] = psi_t1[torch.arange(psi_t1.size(0)), :, a_, ...].detach()

        # Compute the expected psi values
        expected_psi = (next_psi_t1 * self.gamma) + reward_batch.unsqueeze(1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(psi_t, expected_psi)

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

    def one_episode(self, log_path=None):
        """
        This function runs one episode and returns the information about it.
        :return: Dict containing the following data:
        {
                'steps': t + 1,
                'reward': episode_reward,
                'success': 1 if info['cause'] is None else 0,
                'speed': sum(running_speed) / len(running_speed),
                'lane_change': info['lane_change'],
                'distance': info["distance"],
                'type': self.env.simulation_list.index(self.env.sumoCmd[2]),
        }

        """
        # Init log values
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
            # Selecting the policy to train with
            self.j = random.randint(0, len(self.w) - 1)

            # going through the episode step-by-step
            for t in itertools.count():
                # creating the state for the input. Usually 1 timestep but for sumo it may be more.
                current_state = np.stack(state_list[-self.timesteps_observed:])
                # Selecting action based on current state
                action, eps = self.forward(current_state, self.j)
                # Step through environment using chosen action
                # try catch is due to the unhandled sumo errors.
                try:
                    next_state, _, done, info = self.env.step(action.item())
                    # recalculating reward based on the current policy
                    reward_ = info['cumulants']
                except RuntimeError as err:
                    self.env.reset()
                    self.reward_episode = []
                    error_running_traci = True
                    break
                # adding state to the history
                state_list.append(copy.deepcopy(next_state))
                action_list.append(action.item())
                if self.sumo_wrapper:
                    running_speed.append(info['velocity'])
                # Save reward
                self.reward_episode.append(reward_)

                # Printing to console if episode is terminated
                if done:
                    # Store the transition in memory
                    self.push_memory_with_discounted_rewards(action_list, state_list,
                                                             immediate_reward=np.asarray([1, 0, 0, 0, 0, 0]))

                    if self.sumo_wrapper:
                        ep_rew = torch.tensor(self.reward_episode, dtype=torch.float32).sum(0)
                        episode_reward = torch.matmul(ep_rew, self.w[self.j].cpu()).item()
                    else:
                        ep_rew = torch.tensor(self.reward_episode, dtype=torch.float32).sum(0)
                        episode_reward = torch.matmul(ep_rew, self.w[self.j].cpu()).item()

                    print(f"Steps:{t + 1}, reward: {episode_reward:.3f},"
                          f"cause: {info.get('cause', None)}, policy:{list(self.w[self.j].cpu().numpy())}, "
                          f"distance:{info.get('distance', 0):.2f}, global_steps:{self.steps_done}")
                    break
            # logging for the env evaluation
            if log_path is not None:
                if not os.path.exists(os.path.dirname(log_path)):
                    os.makedirs(os.path.dirname(log_path))
                with open(log_path, "bw") as file:
                    log_dict = {"state": [s.tolist() for s in state_list], "action": action_list,
                                "reward": self.reward_episode, "success": 1 if info.get('cause', 0) is None else 0,
                                "cause": info.get('cause', "nan")}
                    pickle.dump(log_dict, file)

        self.reward_episode = []
        success = 1 if info.get('cause', 0) is None else 0

        # Used for tensorboard visualisation. Must contain steps, reward, success
        episode_info = {
            'steps': t + 1,
            'reward': episode_reward,
            'success': success,
            'eps': eps
        }

        if self.sumo_wrapper:
            episode_info.update({'speed': sum(running_speed) / len(running_speed),
                                 'lane_change': info['lane_change'],
                                 'distance': info["distance"],
                                 'type': self.env.simulation_list.index(self.env.sumoCmd[2])})

        return episode_info

    def push_memory_with_discounted_rewards(self, action_list, state_list, immediate_reward=None, last_x=500):
        """
        Function to push discounted returns to the memory.
        :param action_list: batch of actions
        :param state_list: batch of states
        :param immediate_reward: immediate reward multiplier
        :param last_x: how many samples to push maximum. This can be useful for long episodes.
        :return: None
        """

        if immediate_reward is None:
            immediate_reward = np.zeros_like(self.reward_episode[0])

        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = np.array(r) + self.gamma * immediate_reward * R
            rewards.insert(0, R.tolist())
        # Pushing to memory
        for step_i in range(max(len(rewards) - last_x, 0), len(rewards), 1):
            self.memory.push(state_list[step_i].tolist(), action_list[step_i],
                             state_list[step_i + 1].tolist() if step_i < len(rewards) - 1 else None,
                             rewards[step_i], self.j)

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
