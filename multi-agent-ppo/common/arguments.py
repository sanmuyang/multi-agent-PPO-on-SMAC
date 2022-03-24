import argparse
import torch

"""
Here are the param for the training

"""

def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    parser.add_argument('--alg', type=str, default='ippo', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='the optimizer')
    parser.add_argument('--model_dir', type=str, default='./model', help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./model', help='the result directory of the policy')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--threshold', type=int, default=19, help='the threshold to judge whether win')
    parser.add_argument('--evaluate_cycle', type=int, default=10000, help='how often to evaluate the model')
    parser.add_argument('--n_steps', type=int, default=2050000, help='total time steps')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')

    args = parser.parse_args()
    return args

def get_mixer_args(args):
    args.use_gpu = torch.cuda.is_available()

    args.rnn_hidden_dim = 64
    args.lr = 5e-4
    args.lr_actor = 5e-4
    args.lr_critic = 5e-4
    args.train_steps = 1

    # how often to save the model
    args.save_cycle = 9000

    # how often to update the target_net
    # args.target_update_cycle = 200
    args.grad_norm_clip = 10

    args.n_episodes = 32
    args.ppo_n_epochs = 15
    args.lamda = 0.95
    args.clip_param = 0.2
    args.entropy_coeff = 0.01

    return args

