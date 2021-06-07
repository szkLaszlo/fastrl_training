"""
This file contains the script to train dqn for FastRL
"""
from fastrl_training.FastRL.deepmind_env import Scavenger
from fastrl_training.QN.q_agent import QWrapper
from fastrl_training.utils.models import SimpleMLP
from fastrl_training.utils.utils import ReplayMemory

from fastrl_training.utils.wrapper import ModelTrainer

if __name__ == "__main__":
    # creating environment
    env = Scavenger(arena_size=10,
                    num_channels=2,
                    max_num_steps=30,
                    num_init_objects=10,
                    default_w=None
                    )

    # creating composed model
    model = SimpleMLP(input_size=env.observation_space.n, output_size=env.action_space.n,
                      hidden_size=1024)

    # defining the training episodes
    max_episodes = 8000000
    # creating the memory replay object
    memory = ReplayMemory(max_episodes // 10)
    # creating the model wrapper
    wrapper = QWrapper(model=model,
                       env=env,
                       memory=memory,
                       gamma=0.9,
                       timesteps_observed=1,
                       use_gpu=True,
                       batch_size=1024,
                       sumo_wrapper=False,
                       target_after=10,
                       eps_decay=max_episodes // 2
                       )
    # creating the model trainer
    trainer = ModelTrainer(model_wrapper=wrapper,
                           optimizer="adam",
                           learning_rate=0.00001,
                           weight_decay=0,
                           scheduler=None,
                           # {'mode': 'max', 'factor': 0.5, 'patience': 70, 'verbose': True,
                           #            'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 100,
                           #            'min_lr': 0.00001, 'eps': 1e-08},
                           average_after=100,
                           use_tensorboard=True,
                           continue_training=None,
                           save_path=None,
                           render_video_freq=5000)
    # training the agent
    trainer.train(max_train_steps=max_episodes)
