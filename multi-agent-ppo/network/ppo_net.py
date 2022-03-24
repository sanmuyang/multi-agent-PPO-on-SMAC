import torch.nn as nn
import torch
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, critic_input_shape, args, layer_norm=True):
        super(PPOCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(critic_input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

        if layer_norm:
            self.layer_norm(self.fc1, std=1.0)
            self.layer_norm(self.fc2, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, hidden):
        x = F.relu(self.fc1(inputs))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        v = self.fc2(h)
        return v, h

class PPOActor(nn.Module):
    def __init__(self, input_shape, args):
        super(PPOActor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
