from abc import ABC, abstractmethod
import torch
from torch import nn


class RNN(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, input_rank=1, recurrent_rank=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_rank = input_rank
        self.recurrent_rank = recurrent_rank

        # Input layer
        self.input_projective = torch.nn.Parameter(torch.randn((hidden_dim, input_rank)))
        self.input_receptive = torch.nn.Parameter(torch.randn((input_dim, input_rank)))

        # Recurrent layer intrinsic properties
        self.set_recurrent_intrinsic()

        # Recurrent layer connections
        self.reccurent_projective = torch.nn.Parameter(torch.randn((hidden_dim, recurrent_rank)) / hidden_dim)
        self.reccurent_receptive = torch.nn.Parameter(torch.randn((hidden_dim, recurrent_rank)) / hidden_dim)

        # Readout layer
        self.readout = nn.Linear(hidden_dim, output_dim)

    @abstractmethod
    def set_recurrent_intrinsic(self):
        """required for setting the relevant intrinsic parameters"""

    @abstractmethod
    def activation(self, x):
        """required for setting the relevant activation function"""

    @abstractmethod
    def update_hidden(self, h, dh):
        """required for updating the hidden state"""

    def input_weight(self):
        return self.input_projective @ self.input_receptive.t()

    def J(self):
        return self.reccurent_projective @ self.reccurent_receptive.t()

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
            dh = -h + self.activation(r_in + x_in)
            h = self.update_hidden(h, dh)
            if return_hidden:
                hidden[:, step] = h
            out[:, step] = self.readout(h)

        if return_hidden:
            return out, hidden
        return out


class GainRNN(RNN):
    def set_recurrent_intrinsic(self):
        """required for setting the relevant intrinsic parameters"""
        self.hidden_gain = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.hidden_threshold = torch.nn.Parameter(torch.randn(self.hidden_dim))

    def activation(self, x):
        """required for setting the relevant activation function"""
        return self.hidden_gain * torch.relu(x - self.hidden_threshold)

    def update_hidden(self, h, dh):
        """required for updating the hidden state"""
        return h + dh


class TauRNN(RNN):
    def set_recurrent_intrinsic(self):
        """required for setting the relevant intrinsic parameters"""
        self.hidden_gain = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.hidden_tau = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)

    def activation(self, x):
        """required for setting the relevant activation function"""
        return self.hidden_gain * torch.relu(x)

    def update_hidden(self, h, dh):
        """required for updating the hidden state"""
        return h + dh * torch.exp(self.hidden_tau)
