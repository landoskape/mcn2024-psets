import torch
from torch import nn
from snntorch import surrogate
from layers import LeakyLayer, SynapticLayer


class SpikingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=1.0, synaptic=False, alpha=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input layer
        self.input_weight = torch.nn.Linear(input_size, hidden_size)

        # Spiking recurrent layer
        self.synaptic = synaptic
        layer = SynapticLayer if synaptic else LeakyLayer
        args = dict(alpha=alpha) if synaptic else {}
        self.rnn = layer(self.hidden_size, beta=beta, spike_grad=surrogate.fast_sigmoid(), learn_beta=True, **args)
        self.recurrent_weight = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Readout layer
        self.readout = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, start_integration=0):
        # Initialize hidden states and outputs
        batch_size = x.size(0)
        mem = self.rnn.mem_reset(torch.zeros((batch_size, self.hidden_size)))
        if self.synaptic:
            syn = torch.zeros((batch_size, self.hidden_size))
        spk = torch.zeros((batch_size, self.hidden_size))

        spk_rec = []
        mem_rec = []
        if self.synaptic:
            syn_rec = []
        out = torch.zeros((batch_size, 2))

        # Recurrent loop
        for step in range(x.size(1)):
            x_in = self.input_weight(x[:, step])
            r_in = self.recurrent_weight(spk)
            if self.synaptic:
                spk, syn, mem = self.rnn(x_in + r_in, syn=syn, mem=mem)
            else:
                spk, mem = self.rnn(x_in + r_in, mem=mem)
            spk_rec.append(spk)
            if self.synaptic:
                syn_rec.append(syn)
            mem_rec.append(mem)
            if step >= start_integration:
                c_out = self.readout(spk)
                out += c_out

        if self.synaptic:
            return self.softmax(out), torch.stack(spk_rec, dim=1), torch.stack(syn_rec, dim=1), torch.stack(mem_rec, dim=1)
        else:
            return self.softmax(out), torch.stack(spk_rec, dim=1), torch.stack(mem_rec, dim=1)
