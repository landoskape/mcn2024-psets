import torch
from torch import nn


class GainRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_rank=1, recurrent_rank=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_rank = input_rank
        self.recurrent_rank = recurrent_rank

        # Input layer
        self.input_background = torch.randn((hidden_dim, input_dim)) / input_dim
        self.input_projective = torch.nn.Parameter(torch.randn((hidden_dim, input_rank)))
        self.input_receptive = torch.nn.Parameter(torch.randn((input_dim, input_rank)))

        # Recurrent layer intrinsic properties
        self.hidden_gain = torch.nn.Parameter(torch.randn(hidden_dim))
        self.hidden_threshold = torch.nn.Parameter(torch.randn(hidden_dim))

        # Recurrent layer connections
        self.recurrent_background = torch.randn((hidden_dim, hidden_dim)) / hidden_dim
        self.reccurent_projective = torch.nn.Parameter(torch.randn((hidden_dim, recurrent_rank)))
        self.reccurent_receptive = torch.nn.Parameter(torch.randn((hidden_dim, recurrent_rank)))

        # Readout layer
        self.readout = nn.Linear(hidden_dim, output_dim)

    def to(self, device):
        super().to(device)
        self.input_background = self.input_background.to(device)
        self.recurrent_background = self.recurrent_background.to(device)
        return self

    def activation(self, x):
        return self.hidden_gain * torch.relu(x - self.hidden_threshold)

    def input_weight(self):
        return self.input_background + self.input_projective @ self.input_receptive.t()

    def J(self):
        return self.recurrent_background + self.reccurent_projective @ self.reccurent_receptive.t()

    def forward(self, x, return_hidden=False):
        # Initialize hidden states and outputs
        batch_size = x.size(0)
        seq_length = x.size(1)
        if return_hidden:
            hidden = torch.zeros((batch_size, seq_length, self.hidden_dim), device=x.device)
        out = torch.zeros((batch_size, seq_length, self.output_dim), device=x.device)

        # Recurrent loop
        h = torch.zeros((batch_size, self.hidden_dim), device=x.device)
        for step in range(seq_length):
            x_in = (self.input_weight() @ x[:, step].T).T
            r_in = (self.J() @ h.T).T
            dh = -h + self.activation(r_in + x_in) / self.hidden_dim
            h = h + dh
            if return_hidden:
                hidden[:, step] = h
            out[:, step] = self.readout(h)

        if return_hidden:
            return out, hidden
        return out
