"""
@author "Laszlo Szoke" szoke.laszlo@kjk.bme.hu
"""
import copy
import json
import os
import time

import numpy

from fastrl_training.FastRL.deepmind_env import Scavenger
from fastrl_training.FastRL.fastRL_model import FastRLv1, FastRLv1TrainerWrapper
from fastrl_training.QN.q_agent import QWrapper
from fastrl_training.utils.models import SimpleMLP
from fastrl_training.utils.utils import find_latest_weight, ReplayMemory, Transitionv1, Transition
from fastrl_training.utils.wrapper import ModelTrainer, GLOBAL_LOG_PATH
from continuousSUMO.sumoGym.environment import makeContinuousSumoEnv


def model_train(args):
    """
    Function to setup training env and model
    :param args: contains all parameters needed for the train
    :return: none
    """
    # Using google deepmind's modified scavanger
    if not args.is_sumo_wrapper:

        if args.num_object_types is None:
            setattr(args, "num_object_types", 2)

        env = Scavenger(arena_size=args.arena_size,
                        num_channels=args.num_object_types,
                        max_num_steps=args.max_num_steps,
                        num_init_objects=args.num_init_objects,
                        default_w=args.default_w
                        )
        if args.w is None:
            setattr(args, "w", numpy.eye(args.num_object_types).tolist())
    # Using SUMO env
    else:
        save_log_path = os.path.join(GLOBAL_LOG_PATH,
                                     f'SUMOEnvironment/{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}/TIME') \
            if args.save_log_path is None else args.save_log_path

        env = makeContinuousSumoEnv('SUMOEnvironment-v0',
                                    simulation_directory=args.simulation_directory,
                                    type_os=args.type_os,
                                    type_as=args.type_as,
                                    reward_type=args.reward_type,
                                    mode=args.mode,
                                    save_log_path=save_log_path,
                                    change_speed_interval=args.change_speed_interval,
                                    default_w=args.default_w,
                                    seed=args.seed,
                                    )
        if args.w is None:
            setattr(args, "w", numpy.eye(env.get_max_reward(1).shape[0]).tolist())

        if args.num_object_types is None:
            setattr(args, "num_object_types", env.get_max_reward(1).shape[0])

    # Using FastRL model
    if args.model_version == "v1":
        # creating composed model
        model = FastRLv1(input_size=env.observation_space.n, actions=env.action_space.n,
                         d=args.num_object_types, hidden_size=args.model_hidden_size, num_policies=len(args.w))
        modelwrapper = FastRLv1TrainerWrapper
        # creating the memory replay object
        memory = ReplayMemory(args.max_train_steps * args.replay_memory_size // 100, trans=Transitionv1)

    # Using Q-learning model
    elif args.model_version == "q":
        model = SimpleMLP(input_size=env.observation_space.n, output_size=env.action_space.n,
                          hidden_size=args.num_object_types * args.model_hidden_size)
        modelwrapper = QWrapper
        # creating the memory replay object
        memory = ReplayMemory(args.max_train_steps * args.replay_memory_size // 100, trans=Transition)

    else:
        raise RuntimeError("Not implemented model version")

    # creating the model wrapper
    wrapper = modelwrapper(model=model,
                           env=env,
                           memory=memory,
                           w=args.w,
                           gamma=args.gamma,
                           timesteps_observed=args.observed_timesteps,
                           use_gpu=args.use_gpu,
                           batch_size=args.batch_size,
                           sumo_wrapper=args.is_sumo_wrapper,
                           target_after=args.update_target_after,
                           eps_start=args.eps_start,
                           eps_decay=1 / args.eps_decay * args.max_train_steps,
                           eps_stop=args.eps_stop,
                           )

    # creating the model trainer
    trainer = ModelTrainer(model_wrapper=wrapper,
                           optimizer=args.optimizer,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           scheduler=args.lr_scheduler,  # None,
                           average_after=args.average_after,
                           use_tensorboard=args.use_tensorboard,
                           continue_training=args.continue_training,
                           save_path=args.save_path,
                           log_env=args.log_env,
                           render_video_freq=args.render_video_freq,
                           )
    # Saving hyperparameters
    with open(f'{trainer.save_path}/args.txt', 'w') as f:
        del args.func
        json.dump(args.__dict__, f, indent=2)

    # training the agent
    trainer.train(max_train_steps=args.max_train_steps)


def model_test(args):
    """
    Function to evaluate model
    :param args: contains all parameters for the evaluation
    :return: None
    """

    args2 = copy.deepcopy(args)

    # Loading saved argparse elements
    with open(f'{args.load_path}/args.txt', 'r') as f:
        args2.__dict__ = json.load(f)

    # Setting the deepmind Scavanger gym
    if not args2.is_sumo_wrapper:
        env = Scavenger(arena_size=args2.arena_size,
                        num_channels=args2.num_object_types,
                        max_num_steps=args2.max_num_steps,
                        num_init_objects=args2.num_init_objects,
                        default_w=args2.default_w
                        )

    # Setting SUMO environment
    else:
        save_log_path = os.path.join(GLOBAL_LOG_PATH,
                                     f'SUMOEnvironment/{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}/TIME') \
            if args2.save_log_path is None else args2.save_log_path

        env = makeContinuousSumoEnv('SUMOEnvironment-v0',
                                    simulation_directory=args2.simulation_directory,
                                    type_os=args2.type_os,
                                    type_as=args2.type_as,
                                    reward_type=getattr(args, "reward_type", args2.reward_type),
                                    mode=args2.mode,
                                    save_log_path=save_log_path,
                                    change_speed_interval=args2.change_speed_interval,
                                    default_w=args2.default_w,
                                    seed=args.seed,
                                    )
    if args2.num_object_types is None:
        setattr(args2, "num_object_types", env.get_max_reward(1).shape[0])
    # Selecting the model type
    if args2.model_version == "v1":
        # creating composed model
        model = FastRLv1(input_size=env.observation_space.n, actions=env.action_space.n,
                         d=args2.num_object_types, hidden_size=args2.model_hidden_size, num_policies=len(args2.w))
        # Creating the memory replay
        memory = ReplayMemory(1, trans=Transitionv1)

        modelwrapper = FastRLv1TrainerWrapper

    elif args2.model_version == "q":
        model = SimpleMLP(input_size=env.observation_space.n, output_size=env.action_space.n,
                          hidden_size=args2.num_object_types * args2.model_hidden_size)
        modelwrapper = QWrapper
        setattr(args, "w", args2.default_w)
        # creating the memory replay object
        memory = ReplayMemory(1, trans=Transition)

    else:
        raise NotImplementedError
    # creating the model wrapper
    wrapper = modelwrapper(model=model,
                           env=env,
                           memory=memory,
                           w=[args.w],
                           gamma=args2.gamma,
                           timesteps_observed=args2.observed_timesteps,
                           use_gpu=args2.use_gpu,
                           batch_size=args2.batch_size,
                           sumo_wrapper=args2.is_sumo_wrapper,
                           target_after=args2.update_target_after,
                           eps_start=0,
                           eps_decay=1,
                           eps_stop=0,
                           evaluate=True
                           )
    # creating the model trainer
    trainer = ModelTrainer(model_wrapper=wrapper,
                           optimizer=args2.optimizer,
                           learning_rate=args2.learning_rate,
                           weight_decay=args2.weight_decay,
                           scheduler=None,
                           average_after=10,
                           use_tensorboard=args.use_tensorboard,
                           log_after=1,
                           continue_training=None,
                           save_path=os.path.join(args.load_path,
                                                  f'eval_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}'),
                           log_env=args.log_env,
                           render_video_freq=args.max_episodes // 11,
                           )
    # Saving eval parameters
    with open(f'{trainer.save_path}/args_eval.txt', 'w') as f:
        del args.func
        setattr(args, "model", args2.model_version)
        json.dump(args.__dict__, f, indent=2)
    # Finding the last weight from the saved ones
    weigth_path = find_latest_weight(path=args.load_path)
    # Evaluate the model
    trainer.evaluate(episodes=args.max_episodes, path=weigth_path)
