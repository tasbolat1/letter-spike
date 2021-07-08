import torch
import slayerSNN as snn
from slayerSNN import loihi as spikeLayer

class SlayerMLP(torch.nn.Module):
    """1-layer MLP built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, output_size):
        super(SlayerMLP, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc = self.slayer.dense(input_size, output_size)

    def forward(self, spike_input):
        spike_output = self.slayer.spike(self.slayer.psp(self.fc(spike_input)))
        return spike_output


class Slayer2LayerMLP(torch.nn.Module):
    """2-layer MLP built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerMLP, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))
        return spike_output