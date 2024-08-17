from abc import ABC, abstractmethod
import torch
from torch import nn
from math import sqrt


class FullRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.1, nlfun="relu", gainfun="sigmoid", taufun="sigmoid", tauscale=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        if nlfun == "relu":
            self.nlfun = torch.relu
        elif nlfun == "tanh":
            self.nlfun = torch.tanh
        elif nlfun == "linear":
            self.nlfun = lambda x: x
        else:
            raise ValueError(f"nlfun ({nlfun}) not recognized, permitted are: ['relu', 'tanh']")

        prmfun_dict = dict(
            sigmoid=torch.sigmoid,
            exp=torch.exp,
            linear=lambda x: x,
        )
        if gainfun not in prmfun_dict:
            raise ValueError(f"gainfun ({gainfun}) not recognized, permitted are: {list(prmfun_dict.keys())}")
        self.gainfun = prmfun_dict[gainfun]
        if taufun not in prmfun_dict:
            raise ValueError(f"taufun ({taufun}) not recognized, permitted are: {list(prmfun_dict.keys())}")
        self.taufun = prmfun_dict[taufun]
        self.tauscale = tauscale

        assert tauscale > 0, "tauscale must be positive"

        # Input layer
        self.input_weights = torch.nn.Parameter(torch.randn((hidden_dim, input_dim)) / sqrt(hidden_dim) / sqrt(input_dim))

        # Recurrent layer intrinsic properties
        self.hidden_gain = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)
        self.hidden_tau = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)
        self.hidden_threshold = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)

        # Recurrent layer connections
        self.recurrent_weights = torch.nn.Parameter(torch.randn((hidden_dim, hidden_dim)) / hidden_dim)

        # Readout layer
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.readout_scale = nn.Parameter(torch.tensor(1.0))

    def activation(self, x):
        """required for setting the relevant activation function"""
        return self.gainfun(self.hidden_gain) * self.nlfun(x - self.hidden_threshold)

    def update_hidden(self, h, dh):
        """required for updating the hidden state"""
        return h + dh * self.alpha * self.taufun(self.hidden_tau) * self.tauscale

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
            x_in = (self.input_weights @ x[:, step].T).T
            r_in = (self.recurrent_weights @ h.T).T
            dh = -h + self.activation(r_in + x_in)
            h = self.update_hidden(h, dh)
            if return_hidden:
                hidden[:, step] = h
            out[:, step] = self.readout_scale * self.readout(h)

        if return_hidden:
            return out, hidden
        return out


class RNN(nn.Module, ABC):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        input_rank=1,
        recurrent_rank=1,
        alpha=0.1,
        nlfun="relu",
        gainfun="sigmoid",
        taufun="sigmoid",
        tauscale=10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_rank = input_rank
        self.recurrent_rank = recurrent_rank
        self.alpha = alpha
        if nlfun == "relu":
            self.nlfun = torch.relu
        elif nlfun == "tanh":
            self.nlfun = torch.tanh
        elif nlfun == "linear":
            self.nlfun = lambda x: x
        else:
            raise ValueError(f"nlfun ({nlfun}) not recognized, permitted are: ['relu', 'tanh']")

        prmfun_dict = dict(
            sigmoid=torch.sigmoid,
            exp=torch.exp,
            linear=lambda x: x,
        )
        if gainfun not in prmfun_dict:
            raise ValueError(f"gainfun ({gainfun}) not recognized, permitted are: {list(prmfun_dict.keys())}")
        self.gainfun = prmfun_dict[gainfun]
        if taufun not in prmfun_dict:
            raise ValueError(f"taufun ({taufun}) not recognized, permitted are: {list(prmfun_dict.keys())}")
        self.taufun = prmfun_dict[taufun]
        self.tauscale = tauscale

        assert tauscale > 0, "tauscale must be positive"

        # Input layer
        self.input_projective = torch.nn.Parameter(torch.randn((hidden_dim, input_rank)) / sqrt(hidden_dim))
        self.input_receptive = torch.nn.Parameter(torch.randn((input_dim, input_rank)) / sqrt(input_dim))

        # Recurrent layer intrinsic properties
        self.set_recurrent_intrinsic()

        # Recurrent layer connections
        self.reccurent_projective = torch.nn.Parameter(torch.randn((hidden_dim, recurrent_rank)) / sqrt(hidden_dim))
        self.reccurent_receptive = torch.nn.Parameter(torch.randn((hidden_dim, recurrent_rank)) / sqrt(hidden_dim))

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
        self.hidden_gain = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)
        self.hidden_threshold = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)

    def activation(self, x):
        """required for setting the relevant activation function"""
        return self.gainfun(self.hidden_gain) * self.nlfun(x - self.hidden_threshold)

    def update_hidden(self, h, dh):
        """required for updating the hidden state"""
        return h + dh * self.alpha


class TauRNN(RNN):
    def set_recurrent_intrinsic(self):
        """required for setting the relevant intrinsic parameters"""
        self.hidden_gain = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)
        self.hidden_tau = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)

    def activation(self, x):
        """required for setting the relevant activation function"""
        return self.gainfun(self.hidden_gain) * self.nlfun(x)

    def update_hidden(self, h, dh):
        """required for updating the hidden state"""
        return h + dh * self.taufun(self.hidden_tau) * self.alpha * self.tauscale
    

class IntrinsicRNN(RNN):
    def set_recurrent_intrinsic(self):
        """required for setting the relevant intrinsic parameters"""
        self.hidden_gain = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)
        self.hidden_tau = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)
        self.hidden_threshold = torch.nn.Parameter(torch.randn(self.hidden_dim) / 10)

    def activation(self, x):
        """required for setting the relevant activation function"""
        return self.gainfun(self.hidden_gain) * self.nlfun(x - self.hidden_threshold)

    def update_hidden(self, h, dh):
        """required for updating the hidden state"""
        return h + dh * self.taufun(self.hidden_tau) * self.alpha * self.tauscale
