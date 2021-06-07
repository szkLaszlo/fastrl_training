"""
@author "Laszlo Szoke" <szoke.laszlo@kjk.bme.hu>
# This script was created based ont he DeepMind published code.
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# ============================================================================
"""
import copy
import enum
import os
import sys

import cv2
import gym as gym
import matplotlib.pyplot as plt
import numpy as np

this_module = sys.modules[__name__]


class Action(enum.IntEnum):
    """Actions available to the player."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def _one_hot(indices, depth):
    """Creating one hot vector of depth"""
    return np.eye(depth)[indices]


def _random_pos(arena_size):
    """Creating random position on the arena"""
    return tuple(np.random.randint(0, arena_size, size=[2]).tolist())


class Scavenger(gym.Env):
    """Simple Scavenger."""

    def __init__(self,
                 arena_size,
                 num_channels,
                 max_num_steps,
                 default_w=None,
                 num_init_objects=15,
                 object_priors=None,
                 egocentric=True,
                 rewarder=None,
                 aux_tasks_w=None):
        self.name = "Scavanger"
        self.save_path = None
        self.arena_size = arena_size + 1
        self.num_channels = num_channels
        self.type_as = "discrete"
        self.max_num_steps = max_num_steps
        self.num_init_objects = num_init_objects
        self.egocentric = egocentric
        self.rewarder = (
            getattr(this_module, rewarder)() if rewarder is not None else None)
        self.aux_tasks_w = aux_tasks_w
        self.render_mode = "none"
        if object_priors is None:
            self.object_priors = np.ones(num_channels) / num_channels
        else:
            assert len(object_priors) == num_channels
            self.object_priors = np.array(object_priors) / np.sum(object_priors)

        if default_w is None:
            self.default_w = np.ones(shape=(num_channels,))
        else:
            self.default_w = default_w
        self.num_channels_all = self.num_channels + 2
        self.step_in_episode = None
        self.reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(self.get_observation_space(self.observation()).flatten().size)

    @property
    def state(self):
        return copy.deepcopy([
            self.step_in_episode,
            self.walls,
            self.objects,
            self.player_pos,
            self.prev_collected,
        ])

    def set_state(self, state):
        state_ = copy.deepcopy(state)
        self.step_in_episode = state_[0]
        self.walls = state_[1]
        self.objects = state_[2]
        self.player_pos = state_[3]
        self.prev_collected = state_[4]

    @property
    def player_pos(self):
        return self._player_pos

    def reset(self):
        self.step_in_episode = 0

        # Walls.
        self.walls = []
        for col in range(self.arena_size):
            new_pos = (0, col)
            if new_pos not in self.walls:
                self.walls.append(new_pos)
        for row in range(self.arena_size):
            new_pos = (row, 0)
            if new_pos not in self.walls:
                self.walls.append(new_pos)

        # Objects.
        self.objects = dict()
        for _ in range(self.num_init_objects):
            while True:
                new_pos = _random_pos(self.arena_size)
                if new_pos not in self.objects and new_pos not in self.walls:
                    self.objects[new_pos] = np.random.multinomial(1, self.object_priors)
                    break

        # Player
        self.player_pos = _random_pos(self.arena_size)
        while self.player_pos in self.objects or self.player_pos in self.walls:
            self.player_pos = _random_pos(self.arena_size)

        self.prev_collected = np.zeros(shape=(self.num_channels,))

        obs = self.observation()

        return self.get_observation_space(obs)

    def step(self, action):
        self.step_in_episode += 1

        if action == Action.UP:
            new_player_pos = (self._player_pos[0], self._player_pos[1] + 1)
        elif action == Action.DOWN:
            new_player_pos = (self._player_pos[0], self._player_pos[1] - 1)
        elif action == Action.LEFT:
            new_player_pos = (self._player_pos[0] - 1, self._player_pos[1])
        elif action == Action.RIGHT:
            new_player_pos = (self._player_pos[0] + 1, self._player_pos[1])
        else:
            raise ValueError("Invalid action `{}`".format(action))

        # Toroidal.
        new_player_pos = (
            (new_player_pos[0] + self.arena_size) % self.arena_size,
            (new_player_pos[1] + self.arena_size) % self.arena_size,
        )

        if new_player_pos not in self.walls:
            self.player_pos = new_player_pos

        # Compute rewards.
        consumed = self.objects.pop(self.player_pos,
                                    np.zeros(shape=(self.num_channels,)))
        if self.rewarder is None:
            reward = np.dot(consumed, np.array(self.default_w))
        else:
            reward = self.rewarder.get_reward(self.state, consumed)
        self.prev_collected = np.copy(consumed)

        assert self.player_pos not in self.objects
        assert self.player_pos not in self.walls

        # Render everything.
        obs = self.observation()
        self.render()

        if self.step_in_episode < self.max_num_steps:
            return self.get_observation_space(obs), reward, False, {"cause": None, "cumulants": obs['cumulants']}
        else:
            # termination with discount=1.0
            return self.get_observation_space(obs), reward, True, {"cause": None, "cumulants": obs['cumulants']}

    def get_observation_space(self, obs):
        # arena x arena x (obj_i x agent x walls)
        return obs["arena"][:, :, :-1].flatten()

    def observation(self, force_non_egocentric=False):
        arena_shape = [self.arena_size] * 2 + [self.num_channels_all]
        arena = np.zeros(shape=arena_shape, dtype=np.float32)

        def offset_position(pos_):
            use_egocentric = self.egocentric and not force_non_egocentric
            offset = self.player_pos if use_egocentric else (0, 0)
            x = (pos_[0] - offset[0] + self.arena_size) % self.arena_size
            y = (pos_[1] - offset[1] + self.arena_size) % self.arena_size
            return (x, y)

        player_pos = offset_position(self.player_pos)
        # arena x arena x (obj_i x agent x walls) if egocentric
        # arena x arena x (obj_i x walls x agent) if not egocentric
        arena[player_pos] = _one_hot(self.num_channels + (int(not force_non_egocentric)), self.num_channels_all)

        for pos, obj in self.objects.items():
            x, y = offset_position(pos)
            arena[x, y, :self.num_channels] = obj

        for pos in self.walls:
            x, y = offset_position(pos)
            # arena x arena x (obj_i x agent x walls) if egocentric
            # arena x arena x (obj_i x walls x agent) if not egocentric
            arena[x, y] = _one_hot(self.num_channels + (int(force_non_egocentric)), self.num_channels_all)

        collected_resources = np.copy(self.prev_collected).astype(np.float32)

        obs = dict(
            arena=arena,
            cumulants=collected_resources,
        )
        if self.aux_tasks_w is not None:
            obs["aux_tasks_reward"] = np.dot(
                np.array(self.aux_tasks_w), self.prev_collected).astype(np.float32)

        # arena x arena x (obj_i x agent x walls) if egocentric
        # arena x arena x (obj_i x walls x agent) if not egocentric
        return obs

    @player_pos.setter
    def player_pos(self, value):
        self._player_pos = value

    def render(self, mode="human"):

        if self.render_mode == 'human':
            img = self.observation(force_non_egocentric=True)["arena"][1:, 1:, :-1]
            if self.save_path is not None:
                dir_name = os.path.split(self.save_path)[0]
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                im = np.zeros((500, 500, 3))
                scale_x = im.shape[0] // img.shape[0]
                scale_y = im.shape[1] // img.shape[1]
                for k in range(0, im.shape[0], scale_x):
                    for j in range(0, im.shape[1], scale_y):
                        value = img[k // scale_x, j // scale_y]

                        im[k:k + scale_x, j:j + scale_y] = value * 255 * np.ones_like(
                            (im[k:k + scale_x, j:j + scale_y]))
                cv2.imwrite(f"{self.save_path}_img{self.step_in_episode}.jpg", img=im)
            else:
                plt.imshow(img)
                plt.show()

    def set_render(self, mode, save_path=None):
        self.render_mode = mode
        self.save_path = save_path

    def calculate_good_objects_based_on_policy(self, policy):
        sum_ = 0
        for key, value in self.objects.items():
            sum_ += value * policy
        return sum_


class SequentialCollectionRewarder(object):
    """SequentialCollectionRewarder."""

    def get_reward(self, state, consumed):
        """Get reward."""

        object_counts = sum(list(state[2].values()) + [np.zeros(len(consumed))])

        reward = 0.0
        if np.sum(consumed) > 0:
            for i in range(len(consumed)):
                if np.all(object_counts[:i] <= object_counts[i]):
                    reward += consumed[i]
                else:
                    reward -= consumed[i]

        return reward


class BalancedCollectionRewarder(object):
    """BalancedCollectionRewarder."""

    def get_reward(self, state, consumed):
        """Get reward."""

        object_counts = sum(list(state[2].values()) + [np.zeros(len(consumed))])

        reward = 0.0
        if np.sum(consumed) > 0:
            for i in range(len(consumed)):
                if (object_counts[i] + consumed[i]) >= np.max(object_counts):
                    reward += consumed[i]
                else:
                    reward -= consumed[i]

        return reward
