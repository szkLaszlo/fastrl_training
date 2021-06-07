"""
@author "Laszlo Szoke" <szoke.laszlo@kjk.bme.hu>
This file contains script to run the fast rl trainings.
"""
import argparse
import sys

from numpy.distutils.fcompiler import str2bool

from fastrl_training.FastRL.train_test_functions_deepmind import model_test, model_train

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                        __        __   __   ___  __    #
    #   |\/|  /\  | |\ |    |__)  /\  |__) /__` |__  |__)   #
    #   |  | /~~\ | | \|    |    /~~\ |  \ .__/ |___ |  \   #
    #                                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # the subparsers are created from this
    main_parser = argparse.ArgumentParser(description='Fast RL trainer.', add_help=False)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Creating subparser element for different execution in case of different arguments
    subparser = main_parser.add_subparsers(required=False)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Constructing the env parser
    env_parser = argparse.ArgumentParser(description='Environment parser', add_help=False)

    env_parser.add_argument("--env_size", default=5,
                            help="This defines an nxn matrix for the environment")
    env_parser.add_argument("--arena_size", type=int, default=5,
                            help="Size of the arena we want to use")
    env_parser.add_argument("--num_object_types", type=int, default=None,
                            help="Defines how many different object we want.")
    env_parser.add_argument("--max_num_steps", type=int, default=15,
                            help="Defines the max step the agent can take in an episode")
    env_parser.add_argument("--num_init_objects", type=int, default=10,
                            help="Defines how many objects to have at the beginning of each episode.")
    env_parser.add_argument("--default_w", type=list, default=None,
                            help="If None, all the objects will get reward 1. "
                                 "Note: this should be 1 for all, and the model wrapper's w should be used.")
    # env_parser.add_argument("--default_w", type=list, default=[5,2,0,1,1,1],
    #                         help="If None, all the objects will get reward 1. "
    #                              "Note: this should be 1 for all, and the model wrapper's w should be used.")
    ## SUMO related parameters.
    env_parser.add_argument("--simulation_directory", type=str, default=None,
                            help="This is where the simulations are loaded from.")
    env_parser.add_argument("--type_os", type=str, default="structured",
                            help="The observation space type. It can be image or structured")
    env_parser.add_argument("--type_as", type=str, default="discrete",
                            help="The action space type. It can be discrete or continuous")
    env_parser.add_argument("--reward_type", type=str, default="positive",
                            help="Defines how the rewarding is done. See the environment")
    env_parser.add_argument("--mode", type=str, default="none",
                            help="Defines if we want to render the environment. Can be none or human")
    env_parser.add_argument("--change_speed_interval", type=int, default=30,
                            help="Defines how often to change the desired speed of the ego")
    env_parser.add_argument("--save_log_path", type=str, default=None,
                            help="Defines where to save the simulation data., if None, a timestamp will be used.")
    env_parser.add_argument("--seed", type=int, default=None)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    model_parser = argparse.ArgumentParser(description='Model parser', add_help=False)

    model_parser.add_argument("--model_hidden_size", type=int, default=128)
    model_parser.add_argument("--model_version", type=str, default="v1")

    model_parser.add_argument("--w", type=float, nargs='+', default=None,
                              help="Successor feature weights.")

    wrapper_parser = argparse.ArgumentParser(description='Model parser', add_help=False)

    wrapper_parser.add_argument("--replay_memory_size", type=float, default=0.3,
                                help="This is the percentage of the max_train_steps")
    wrapper_parser.add_argument("--gamma", type=float, default=0.95,
                                help="Defines the discount factor for the training")
    wrapper_parser.add_argument("--observed_time steps", type=int, default=1,
                                help="Defines how many time steps to give to the network")
    wrapper_parser.add_argument("--use_gpu", default=True,
                                help="If true we use the gpu")
    wrapper_parser.add_argument("--batch_size", type=int, default=1024)
    wrapper_parser.add_argument("--is_sumo_wrapper", type=str2bool, default=True,
                                help="This is true if we use the SUMO environment")
    wrapper_parser.add_argument("--update_target_after", type=int, default=2)
    wrapper_parser.add_argument("--eps_decay", type=float, default=20,
                                help="Defines the percentage of the max_episode where the decay ends")
    wrapper_parser.add_argument("--eps_start", type=float, default=0.5,
                                help="Defines the epsilon decay start value.")
    wrapper_parser.add_argument("--eps_stop", type=float, default=0.01,
                                help="Defines the minimum value of the eps decay.")

    trainer_parser = subparser.add_parser("train", description='Fast RL wrapper parser',
                                          parents=[env_parser, model_parser, wrapper_parser], add_help=False,
                                          conflict_handler='resolve',
                                          aliases=['t'])
    trainer_parser.add_argument("--optimizer", default='AdamW')
    trainer_parser.add_argument("--learning_rate", type=float, default=0.001)
    trainer_parser.add_argument("--weight_decay", default=0.0, type=float,
                                help="Defines the coefficient (weight) of the used weight decay regularization.")
    trainer_parser.add_argument("--lr_scheduler", type=dict, default={'mode': 'min',
                                                                      'factor': 0.5,
                                                                      'patience': 50,
                                                                      'verbose': True,
                                                                      'threshold': 0.01,
                                                                      'threshold_mode': 'rel',
                                                                      'cooldown': 50,
                                                                      'min_lr': 0.000001,
                                                                      'eps': 1e-08})
    trainer_parser.add_argument("--average_after", type=int, default=100,
                                help="How many episodes to average in logging")
    trainer_parser.add_argument("--use_tensorboard", type=str2bool, default=True)
    trainer_parser.add_argument("--log_env", type=str2bool, default=False)
    trainer_parser.add_argument("--continue_training", type=str,
                                default=None,
                                help="If not None, the weights are loaded from the given path, "
                                     "and the training is continued.")
    trainer_parser.add_argument("--save_path", type=str, default=None,
                                help="If None then it will be created based on the time.")
    trainer_parser.add_argument("--render_video_freq", type=int, default=5000,
                                help="Defines after how many episodes we want to take videos.")
    trainer_parser.add_argument("--max_train_steps", type=int, default=20000000,
                                help="Defines how many episodes to run.")
    trainer_parser.add_argument("--comment", type=str, default="really negative rewards",
                                help="Defines how many episodes to run.")

    trainer_parser.set_defaults(func=model_train)

    eval_parser = subparser.add_parser("eval", description='Fast RL wrapper parser',
                                       parents=[], add_help=False,
                                       conflict_handler='resolve',
                                       aliases=['e'])
    eval_parser.add_argument("--w", type=float, nargs='+', default=[1.0, 0.0, 0.1, 0.5, 1.0])
    eval_parser.add_argument("--reward_type", type=str, default="positive",
                             help="Defines how the rewarding is done. See the environment")
    eval_parser.add_argument("--seed", type=int, default=None)
    eval_parser.add_argument("--max_episodes", type=int, default=100,
                             help="Defines how many episodes to run.")
    eval_parser.add_argument("--use_tensorboard", type=str2bool, default=True)
    eval_parser.add_argument("--log_env", type=str2bool, default=True)
    eval_parser.add_argument("--load_path", type=str, help="Must contain a .weight file",
                             default="/cache/RL/training_with_policy/Qnetwork_SimpleMLP_SuMoGyM_discrete/20210409_145619")

    eval_parser.set_defaults(func=model_test)
    # Adding default run mode, especially it is for the pycharm execution.
    if len(sys.argv) == 1:
        sys.argv.insert(1, 'eval')
    # Parsing the args from the command line
    args = main_parser.parse_args()
    # Running desired script based on the input.
    args.func(args)
