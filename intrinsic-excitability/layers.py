import torch
from torch import nn
from snntorch import Leaky, Synaptic
from torch.nn.functional import sigmoid


class LeakyLayer(Leaky):
    def __init__(
        self,
        num_units,
        beta=0.5,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=True,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        self.num_units = num_units
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
            reset_delay,
        )

    def _base_state_function(self, input_):
        base_fn = sigmoid(self.beta) * self.mem + input_
        return base_fn

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta * torch.ones(self.num_units))
        else:
            self.register_buffer("beta", beta)


class SynapticLayer(Synaptic):
    def __init__(
        self,
        num_units,
        alpha=1.0,
        beta=1.0,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=True,
        learn_beta=True,
        learn_threshold=True,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        reset_delay=True,
    ):
        self.num_units = num_units
        super().__init__(
            alpha,
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_alpha,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            reset_delay,
        )

    def _base_state_function(self, input_):
        base_fn_syn = sigmoid(self.alpha) * self.syn + input_
        base_fn_mem = sigmoid(self.beta) * self.mem + base_fn_syn
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        base_fn_syn = sigmoid(self.alpha) * self.syn + input_
        base_fn_mem = sigmoid(self.beta) * self.mem + base_fn_syn
        return 0, base_fn_mem

    def _alpha_register_buffer(self, alpha, learn_alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha * torch.ones(self.num_units))
        else:
            self.register_buffer("alpha", alpha)
