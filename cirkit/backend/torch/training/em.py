from abc import ABC
from typing import Generic, TypeVar, override

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchSumLayer, TorchGaussianLayer, TorchLayer, TorchHadamardLayer
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter

AnyTorchSumLayer = TorchSumLayer


def update_params_nested(params: TorchParameter, new_values: torch.Tensor):
    for p in params.outputs:
        if type(p) == TorchTensorParameter:
            p.update_params(new_values.squeeze())


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
        self.weight = self.layer.weight.outputs[0]._ptensor

    @override
    def expectation(self):
        self.sufficient_statistics["n"] = self.weight.grad.sum(dim=2)

    @override
    def maximization(self):
        n = self.sufficient_statistics["n"]
        new_weight = self.weight * n / n.sum()

        update_params_nested(self.layer.weight, new_weight)
        self.weight.copy_(new_weight)


class GaussianLayerEM(AbstractLayerEM[TorchGaussianLayer]):

    def __init__(self, layer: TorchGaussianLayer):
        super().__init__(layer)
        self.mean = self.layer.params["mean"]()
        self.stddev = self.layer.params["stddev"]()
        self.inputs = None
        self.outputs = None

    @override
    def layer_fn(self, layer, *inputs):
        self.inputs = torch.stack(inputs)
        self.outputs = layer(*inputs)
        self.outputs.retain_grad()
        return self.outputs

    @override
    def expectation(self):
        p_l = self.outputs.grad.permute(2, 1, 0)  # [Inputs, Samples, Outputs] -> [Outputs, Samples, Inputs]

        x = self.inputs  # [Samples, Features]
        x_2 = x ** 2  # [Samples, Features]

        self.sufficient_statistics["x"] = torch.sum(p_l * x, dim=2)  # [Outputs, Inputs]
        self.sufficient_statistics["x^2"] = torch.sum(p_l * x_2, dim=2)  # [Outputs, Inputs]
        self.sufficient_statistics["p_l"] = torch.sum(p_l, dim=1)  # [Outputs, Inputs]

    @override
    def maximization(self):
        p_l = self.sufficient_statistics["p_l"]  # [Outputs, Inputs]
        x = self.sufficient_statistics["x"] / p_l  # [Outputs, Inputs]
        x_2 = self.sufficient_statistics["x^2"] / p_l  # [Outputs, Inputs]

        mean = x  # [Outputs, Inputs]
        var = x_2 - x ** 2  # [Outputs, Inputs]
        stddev = torch.sqrt(var)  # [Outputs, Inputs]

        update_params_nested(self.layer.params["mean"], mean)
        # update_params_nested(self.layer.params["stddev"], stddev)

        self.mean.copy_(mean.squeeze())
        #  self.stddev.copy_(stddev.squeeze())


def create_layer_em(layer: TorchLayer) -> AbstractLayerEM[TorchLayer]:
    if layer.num_parameters == 0:
        return NoopLayerEM(layer)
    match layer:
        case TorchSumLayer():
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
