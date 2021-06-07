"""
This file contains a trainer and evaluation wrapper for models.
"""
import itertools
import os
import platform
import random
import time

import numpy
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from fastrl_training.utils.utils import select_optimizer, generate_video_from_images

GLOBAL_LOG_PATH = "/cache/RL/training_with_policy"


class ModelWrapperInterface(object):

    def __init__(self):
        super(ModelWrapperInterface, self).__init__()
        self.model = NotImplementedError

    def one_episode(self, log_path):
        """
        This function runs one episode in the environment and returns the episode results.
        :return: dict of the parameters for tensorboard log.
        """
        return NotImplementedError("ModelWrapper's one_episode function is not implemented")

    def update(self, optimizer):
        """
        This function runs one model update.
        :return: dict of the parameters for tensorboard log.
        """
        return NotImplementedError("ModelWrapper's update_model function is not implemented")


class ModelTrainer:

    def __init__(self,
                 model_wrapper,
                 optimizer="adam",
                 learning_rate=0.0001,
                 weight_decay=0.001,
                 scheduler=None,
                 average_after=50,
                 log_after=50,
                 early_stopping=False,
                 use_tensorboard=True,
                 continue_training=None,
                 render_video_freq=None,
                 save_path=None,
                 log_env=False,
                 seed=42):

        torch.manual_seed(seed=seed)
        numpy.random.seed(seed)
        random.seed(seed)

        self.render_video_freq = render_video_freq
        self.average_after = average_after
        self.log_after = log_after
        self.early_stopping = early_stopping
        self.model_wrapper = model_wrapper
        # count_parameters(self.model_wrapper.model)
        # Saving path creation if needed
        self.save_path = save_path \
            if save_path is not None \
            else os.path.join(GLOBAL_LOG_PATH,
                              f'{self.model_wrapper.name}/{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}')

        if not os.path.exists(self.save_path) and (use_tensorboard or render_video_freq is not None):
            os.makedirs(self.save_path)
        self.writer = SummaryWriter(self.save_path) if use_tensorboard else None
        self.log_env = log_env
        if continue_training is not None:
            self.load_weights(continue_training, req_grad=True)

        self.optimizer = select_optimizer(model=self.model_wrapper.model,
                                          optimizer_=optimizer,
                                          learning_rate=learning_rate,
                                          weight_decay=weight_decay)
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler)

    def train(self, max_train_steps):
        """
        This function starts the training.
        :param max_train_steps: How many episodes to do.
        :return: None
        """

        # Initiating global variables
        running_reward = 0
        done_average = 0
        loss_average = 0
        max_reward = -100000
        max_done = 0
        min_loss = 10000000
        stopping_counter = 0
        self.model_wrapper.train()
        # Running the episodes
        for episode in itertools.count():
            tb_info = self.run_one_episode(episode)
            running_reward += tb_info["reward"]
            done_average += tb_info["success"]

            # Updating the network
            tb_info.update(self.model_wrapper.update(self.optimizer))
            loss_average += tb_info["loss"] if tb_info["loss"] is not None else 1000

            self.handle_tensorboard(tb_info, episode)

            # Calculating average based on a bunch of episodes episodes
            if episode % self.average_after == 0 and episode != 0:
                running_reward = running_reward / self.average_after
                done_average = done_average / self.average_after
                loss_average = loss_average / self.average_after

                if self.writer is not None:
                    self.writer.add_scalar('training/reward', running_reward, episode + 1)
                    self.writer.add_scalar('training/success', done_average, episode + 1)
                    # network_weight_plot(self.model_wrapper.model, writer=self.writer, epoch=episode)

                if self.scheduler is not None:
                    self.scheduler.step(loss_average)
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    self.writer.add_scalar('training/learning_rate', lr, episode)

                # Checking if reward has improved
                if running_reward > max_reward:
                    max_reward = running_reward
                    stopping_counter = 0
                    self.save_best_weights(episode)

                elif done_average > max_done:
                    max_done = done_average
                    stopping_counter = 0
                    self.save_best_weights(episode)

                elif abs(loss_average) < min_loss:
                    min_loss = abs(loss_average)
                    stopping_counter = 0
                    self.save_best_weights(episode)

                if stopping_counter > max_train_steps * 0.001 and self.early_stopping:
                    print(f"The rewards did not improve since {self.average_after * (stopping_counter - 1)} steps")
                    self.model_wrapper.env.stop()
                    break

                else:
                    stopping_counter += 1

                # Clearing averaging variables
                running_reward = 0
                done_average = 0
                loss_average = 0

                if max_train_steps < self.model_wrapper.steps_done:
                    break
        # Saving final weights
        torch.save(self.model_wrapper.model.state_dict(), os.path.join(self.save_path, 'model_final.weight'))

    def run_one_episode(self, episode):
        print(f"Episode {episode + 1}:")

        # Setting the renderer
        if self.render_video_freq is not None and (
                episode % self.render_video_freq == 0):
            self.model_wrapper.env.set_render(mode="human", save_path=f"{self.save_path}/videos/{episode}")

        # Run an episode
        tb_info = self.model_wrapper.one_episode(
            log_path=f"{self.save_path}/env/{episode}.pkl" if self.log_env else None)

        # Turn of the render
        if self.render_video_freq is not None and (
                episode % self.render_video_freq == 0):
            self.model_wrapper.env.set_render(mode="none")
            generate_video_from_images(img_dir_path=f"{self.save_path}/videos/",
                                       video_name=f"episode_{episode}_p{getattr(self.model_wrapper, 'j', 0)}.avi")
        return tb_info

    def save_best_weights(self, episode, delete_previous=True):
        """
        This function saves the current weights.
        It also removes the previously saved ones if delete_previous is True.
        """
        if delete_previous:
            current_weigth_list = os.listdir(self.save_path)
            for item in current_weigth_list:
                if item.endswith(".weight"):
                    os.remove(os.path.join(self.save_path, item))
        # Saving weights with better results
        torch.save(self.model_wrapper.model.state_dict(),
                   os.path.join(self.save_path, f'model_{episode + 1}.weight'))

    def evaluate(self, episodes, path):
        """
        This function loads the weights of the trained model, and runs the evaluation.
        """
        self.load_weights(path)
        self.model_wrapper.eval()
        rewards = 0
        average_steps = 0
        average_done = 0
        with torch.no_grad():
            for episode in range(episodes):
                if episode % self.render_video_freq == 0:
                    self.model_wrapper.env.set_render(mode="human", save_path=f"{self.save_path}/videos/{episode}")
                else:
                    self.model_wrapper.env.set_render(mode="none")
                print(f"Episode {episode + 1}:")
                tb_info = self.model_wrapper.one_episode(
                    log_path=f"{self.save_path}/env/{episode}.pkl" if self.log_env else None)

                self.handle_tensorboard(tb_info, episode)

                rewards += tb_info["reward"]
                average_steps += tb_info["steps"]
                average_done += tb_info['success']
                generate_video_from_images(img_dir_path=f"{self.save_path}/videos/",
                                           video_name=f"episode_{episode}.avi")
        if self.writer is not None:
            self.writer.add_scalar("Evaluation/average_reward", rewards / episodes, episodes)
            self.writer.add_scalar("Evaluation/average_step", average_steps / episodes, episodes)
            self.writer.add_scalar("Evaluation/success", average_done / episodes, episodes)
            self.writer.flush()

        print(f"Evaluation rewards: {rewards / episodes}, average steps: {average_steps / episodes}")

    def handle_tensorboard(self, tb_info, episode):
        """
        This function writes to the tensorboard with the defined dict.
        :param: tb_info: Dict, containing the data to be written.
        :return: None
        """
        # If needed writing episode details to tensorboard
        if self.writer is not None and episode % self.log_after == 0:
            for key, value in tb_info.items():
                if value is not None:
                    self.writer.add_scalar(f"episode/{key}", value, episode)

    def load_weights(self, path_, req_grad=False):
        """
        This function loads the defined weights from the path_ and loads it to the model.
        :param: path_: the concrete path of the model weights.
        :return: None
        """
        state_dicts = torch.load(path_, map_location=torch.device('cpu')
        if "Windows" in platform.system() else torch.device('cuda'))
        print(self.model_wrapper.model.load_state_dict(state_dicts), f"from {path_}")
        for param in self.model_wrapper.model.parameters():
            if param.requires_grad:
                param.requires_grad = req_grad
