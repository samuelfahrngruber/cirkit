from abc import ABC
from typing import Generic, TypeVar, override

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchSumLayer, TorchGaussianLayer, TorchLayer, TorchCPTLayer

numerical_epsilon = 1e-10


AnyTorchSumLayer = TorchSumLayer | TorchCPTLayer


L = TypeVar("L", bound=TorchLayer)


class AbstractLayerEM(Generic[L], ABC):

    def __init__(self, layer: L):
        self.layer = layer
        self.sufficient_statistics: dict[str, Tensor] = {}

    def layer_fn(self, layer, *inputs):
        return layer(*inputs)

    def expectation(self):
        raise NotImplementedError()

    def maximization(self):
        raise NotImplementedError()


class NoopLayerEM(AbstractLayerEM[TorchLayer]):

    def __init__(self, layer: TorchLayer):
        super().__init__(layer)

    @override
    def expectation(self):
        pass

    @override
    def maximization(self):
        pass


class SumLayerEM(AbstractLayerEM[AnyTorchSumLayer]):

    def __init__(self, layer: AnyTorchSumLayer):
        super().__init__(layer)
        self.weight_grads = None
        self.layer.weight.register_full_backward_hook(self.weight_grads_hook)

    def weight_grads_hook(self, module, grad_input, grad_output):
        self.weight_grads = grad_output[0]

    def update_weight(self, new_weight):
        self.layer.weight.nodes[0]._ptensor.copy_(new_weight)

    @override
    def expectation(self):
        self.sufficient_statistics["n"] = self.weight_grads * self.layer.weight()

    @override
    def maximization(self):
        n = self.sufficient_statistics["n"]
        new_weight = n / n.sum()
        self.update_weight(new_weight)


class GaussianLayerEM(AbstractLayerEM[TorchGaussianLayer]):

    def __init__(self, layer: TorchGaussianLayer):
        super().__init__(layer)
        self.inputs = None
        self.outputs = None
        self.output_grads = None
        self.layer.register_full_backward_hook(self.output_grads_hook)

    def output_grads_hook(self, module, grad_input, grad_output):
        self.output_grads = grad_output[0]

    def update_mean(self, new_mean):
        self.layer.params["mean"].nodes[0]._ptensor.copy_(new_mean)

    def update_stddev(self, new_stddev):
        self.layer.params["stddev"].nodes[0]._ptensor.copy_(new_stddev)

    @override
    def layer_fn(self, layer, *inputs):
        self.inputs = torch.stack(inputs)
        self.outputs = layer(*inputs)
        return self.outputs

    @override
    def expectation(self):
        p_l = self.output_grads.unsqueeze(0)  # [1, Features, Samples, Outputs]

        x = self.inputs  # [1, Features, Samples, 1]
        x_2 = x ** 2  # [1, Features, Samples, 1]

        self.sufficient_statistics["x"] = (p_l * x).sum(2, keepdim=True)  # [1, Features, 1, Outputs]
        self.sufficient_statistics["x^2"] = (p_l * x_2).sum(2, keepdim=True)  # [1, Features, 1, Outputs]
        self.sufficient_statistics["p_l"] = p_l.sum(2, keepdim=True)  # [1, Features, 1, Outputs]

    @override
    def maximization(self):
        p_l = self.sufficient_statistics["p_l"]  # [1, Features, 1, Outputs]
        x = self.sufficient_statistics["x"] / p_l  # [1, Features, 1, Outputs]
        x_2 = self.sufficient_statistics["x^2"] / p_l  # [1, Features, 1, Outputs]

        mean = x  # [1, Features, 1, Outputs]
        var = x_2 - x ** 2  # [1, Features, 1, Outputs]
        stddev = var.sqrt() # [1, Features, 1, Outputs]
        stddev = stddev.log() # TODO: investigate why this is needed - maybe the notebooks is wrong?

        self.update_mean(mean.squeeze())
        self.update_stddev(stddev.squeeze())


def create_layer_em(layer: TorchLayer) -> AbstractLayerEM[TorchLayer]:
    if layer.num_parameters == 0:
        return NoopLayerEM(layer)
    match layer:
        case TorchSumLayer() | TorchCPTLayer():
            return SumLayerEM(layer)
        case TorchGaussianLayer():
            return GaussianLayerEM(layer)
    raise ValueError(f"EM is not supported for layer: {layer}")


class FullBatchEM:

    def __init__(self, circuit: TorchCircuit):
        self.circuit = circuit
        self.layer_ems: dict[TorchLayer, AbstractLayerEM[TorchLayer]] = {}
        for layer in circuit.layers:
            self.layer_ems[layer] = create_layer_em(layer)
        self.log_likelihoods_per_sample = None

    def zero_grad(self):
        self.circuit.zero_grad()

    def forward(self, data):
        self.log_likelihoods_per_sample = self.circuit.evaluate(data, module_fn=self.layer_fn)
        return self.log_likelihoods_per_sample

    def layer_fn(self, layer, *inputs):
        return self.layer_ems[layer].layer_fn(layer, *inputs)

    def backward_latent_posterior(self):
        self.log_likelihoods_per_sample.sum().backward()

    def expectation(self):
        with torch.no_grad():
            for layer, em in self.layer_ems.items():
                if em:
                    em.expectation()

    def maximization(self):
        with torch.no_grad():
            for layer, em in self.layer_ems.items():
                if em:
                    em.maximization()
