import numpy as np
import torch
from policy.mappo import MAPPO
from policy.ippo import IPPO
from torch.distributions import Categorical

class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'mappo':
            self.policy = MAPPO(args)
        elif args.alg == 'ippo':
            self.policy = IPPO(args)
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        policy_hidden_state = self.policy.policy_hidden[:, agent_num, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.use_gpu:
            inputs = inputs.cuda()
            policy_hidden_state = policy_hidden_state.cuda()

        policy_q_value, self.policy.policy_hidden[:, agent_num, :] = self.policy.policy_rnn.forward(inputs, policy_hidden_state)
        action_prob = torch.nn.functional.softmax(policy_q_value.cpu(), dim=-1)
        action_prob[avail_actions == 0.0] = 0.0
        if evaluate:
            action = torch.argmax(action_prob)
        else:
            action = Categorical(action_prob).sample().long()

        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step=0, time_steps=0, epsilon=None):
        max_episode_len = self._get_max_episode_len(batch)

        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, time_steps)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
