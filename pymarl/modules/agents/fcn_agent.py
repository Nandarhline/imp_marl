import torch.nn as nn
import torch.nn.functional as F


class FCNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_layer = nn.Linear(input_shape, args.rnn_hidden_dim[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hd1, hd2)
                for hd1, hd2 in zip(args.rnn_hidden_dim[:-1], args.rnn_hidden_dim[1:])])
        self.output_layer = nn.Linear(hidden_dim[-1], args.n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.input_layer(inputs))
        for index, hidden_layer in enumerate (self.hidden_layers):
            x = F.relu(hidden_layer(x))
        q = F.relu(self.output_layer(x))
        h = None
        return q, h
