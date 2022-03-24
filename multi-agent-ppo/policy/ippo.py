import torch
import os
import torch.functional as F
from network.ppo_net import PPOActor
from network.ppo_net import PPOCritic

from torch.distributions import Categorical


class IPPO:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        self.policy_rnn = PPOActor(actor_input_shape, args)
        self.eval_critic = PPOCritic(critic_input_shape, self.args)
        # self.target_critic = PPOCritic(critic_input_shape, self.args)

        if self.args.use_gpu:
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            # self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        # if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
        #     if os.path.exists(self.model_dir + '/rnn_params.pkl'):
        #         path_rnn = self.model_dir + '/rnn_params.pkl'
        #         path_coma = self.model_dir + '/critic_params.pkl'
        #         map_location = 'cuda:0' if self.args.use_gpu else 'cpu'
        #         self.policy_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
        #         self.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
        #         print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
        #     else:
        #         raise Exception("No model!")

        # self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.ac_parameters = list(self.policy_rnn.parameters()) + list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.ac_optimizer = torch.optim.RMSprop(self.ac_parameters, lr=args.lr)
        elif args.optimizer == "Adam":
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=args.lr)

        self.args = args

        self.policy_hidden = None
        self.eval_critic_hidden = None
        # self.target_critic_hidden = None

    def _get_critic_input_shape(self):
        # obs
        input_shape = self.obs_shape
        # agent_id
        input_shape += self.n_agents
        # input_shape += self.n_actions * self.n_agents * 2  # 54

        return input_shape

    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated, s = batch['u'], batch['r'], batch['avail_u'], batch['terminated'], batch['s']

        mask = (1 - batch["padded"].float())

        if self.args.use_gpu:
            u = u.cuda()
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            s = s.cuda()

        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        old_values, _ = self._get_values(batch, max_episode_len)
        old_values = old_values.squeeze(dim=-1)
        old_action_prob = self._get_action_prob(batch, max_episode_len)

        old_dist = Categorical(old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0

        for _ in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)

            values, target_values = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)

            returns = torch.zeros_like(r)
            deltas = torch.zeros_like(r)
            advantages = torch.zeros_like(r)

            prev_return = 0.0
            prev_value = 0.0
            prev_advantage = 0.0
            for transition_idx in reversed(range(max_episode_len)):
                returns[:, transition_idx] = r[:, transition_idx] + self.args.gamma * prev_return * (
                            1 - terminated[:, transition_idx]) * mask[:, transition_idx]
                deltas[:, transition_idx] = r[:, transition_idx] + self.args.gamma * prev_value * (
                            1 - terminated[:, transition_idx]) * mask[:, transition_idx] \
                                            - values[:, transition_idx]
                advantages[:, transition_idx] = deltas[:,
                                                transition_idx] + self.args.gamma * self.args.lamda * prev_advantage * (
                                                            1 - terminated[:, transition_idx]) * mask[:, transition_idx]

                prev_return = returns[:, transition_idx]
                prev_value = values[:, transition_idx]
                prev_advantage = advantages[:, transition_idx]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()

            if self.args.use_gpu:
                advantages = advantages.cuda()

            action_prob = self._get_action_prob(batch, max_episode_len)
            dist = Categorical(action_prob)
            log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            log_pi_taken[mask == 0] = 0.0

            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages

            entropy = dist.entropy()
            entropy[mask == 0] = 0.0

            policy_loss = torch.min(surr1, surr2) + self.args.entropy_coeff * entropy

            policy_loss = - (policy_loss * mask).sum() / mask.sum()

            error_clip = torch.clamp(values - old_values.detach(), -self.args.clip_param,
                                     self.args.clip_param) + old_values.detach() - returns
            error_original = values - returns

            value_loss = 0.5 * torch.max(error_original ** 2, error_clip ** 2)
            value_loss = (mask * value_loss).sum() / mask.sum()

            loss = policy_loss + value_loss

            self.ac_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_parameters, self.args.grad_norm_clip)
            self.ac_optimizer.step()

        # if train_step > 0 and train_step % self.args.target_update_cycle == 0:
        #     self.target_critic.load_state_dict(self.eval_critic.state_dict())

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx], \
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        # u_onehot = batch['u_onehot'][:, transition_idx]
        # if transition_idx != max_episode_len - 1:
        #     u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        # else:
        #     u_onehot_next = torch.zeros(*u_onehot.shape)
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        # u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        # u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        #
        # if transition_idx == 0:
        #     u_onehot_last = torch.zeros_like(u_onehot)
        # else:
        #     u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
        #     u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        inputs, inputs_next = [], []

        inputs.append(obs)
        inputs_next.append(obs_next)

        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                # inputs_next = inputs_next.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()
                # self.target_critic_hidden = self.target_critic_hidden.cuda()

            v_eval, self.eval_critic_hidden = self.eval_critic(inputs, self.eval_critic_hidden)
            # v_target, self.target_critic_hidden = self.eval_critic(inputs_next, self.target_critic_hidden)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            # v_target = v_target.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
            # v_targets.append(v_target)

        v_evals = torch.stack(v_evals, dim=1)
        # v_targets = torch.stack(v_targets, dim=1)
        return v_evals, v_targets

    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_action_prob(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.policy_hidden = self.policy_hidden.cuda()
            outputs, self.policy_hidden = self.policy_rnn(inputs, self.policy_hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_prob = action_prob + 1e-10

        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0

        action_prob = action_prob + 1e-10

        if self.args.use_gpu:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        self.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        # self.target_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.policy_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_params.pkl')