from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar, override

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchSumLayer, TorchGaussianLayer, TorchLayer, TorchCPTLayer
from cirkit.backend.torch.parameters.nodes import TorchParameterNode, TorchScaledSigmoidParameter, TorchSoftmaxParameter, TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter

AnyTorchSumLayer = TorchSumLayer | TorchCPTLayer


L = TypeVar("L", bound=TorchLayer)


@dataclass
class EMConfig:
    eps_clamp_min_normalization: float = 1e-10
    eps_clamp_inv_sigmoid: float = 1e-7
    eps_clamp_variance: float = 1e-7


class TorchParameterInteractions:

    def __init__(self, param: TorchParameter, config: EMConfig):
        self.param = param
        self.config = config

    def update(self, new_param_out: Tensor):
        # TODO: check if the parameter is a single chain of reparameterizations, otherwise throw an error
        for node in reversed(self.param.nodes):
            if isinstance(node, TorchTensorParameter):
                node._ptensor.copy_(new_param_out)
                return
            new_param_out = self.invert_node(node, new_param_out)
        raise ValueError("Did not reach final tensor parameter. Could not update.")

    def invert_node(self, node: TorchParameterNode, out: Tensor) -> Tensor:
        match node:
            case TorchScaledSigmoidParameter():
                # forward taken from implementation: torch.sigmoid(x) * (self.vmax - self.vmin) + self.vmin
                normalized_out = (out - node.vmin) / (node.vmax - node.vmin)
                input = torch.logit(normalized_out, eps=self.config.eps_clamp_inv_sigmoid)
                return input
            case TorchSoftmaxParameter():
                # forward taken from implementation: torch.softmax(x, dim=self.dim + 1)
                # input is proportional to log output
                return out.log()
        raise ValueError(f"Cannot invert reparameterization for: {node}")


class AbstractLayerEM(Generic[L], ABC):

    def __init__(self, layer: L, config: EMConfig = EMConfig()):
        self.layer = layer
        self.config = config
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

    @override
    def expectation(self):
        self.sufficient_statistics["n"] = self.weight_grads * self.layer.weight()

    @override
    def maximization(self):
        n = self.sufficient_statistics["n"]
        new_weight = n / n.sum()
        weight_handler = TorchParameterInteractions(self.layer.weight, self.config)
        weight_handler.update(new_weight)


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

        normalization_constant = p_l.sum(2, keepdim=True)
        normalization_constant = normalization_constant.clamp_min(self.config.eps_clamp_min_normalization)
        self.sufficient_statistics["x"] = (p_l * x).sum(2, keepdim=True) / normalization_constant # [1, Features, 1, Outputs]
        self.sufficient_statistics["x^2"] = (p_l * x_2).sum(2, keepdim=True) / normalization_constant # [1, Features, 1, Outputs]

    @override
    def maximization(self):
        x = self.sufficient_statistics["x"]
        x_2 = self.sufficient_statistics["x^2"]

        mean = x  # [1, Features, 1, Outputs]
        var = x_2 - x ** 2  # [1, Features, 1, Outputs]
        var = var.clamp_min(self.config.eps_clamp_variance)
        stddev = var.sqrt() # [1, Features, 1, Outputs]

        stddev_handler = TorchParameterInteractions(self.layer.params["stddev"], self.config)
        stddev_handler.update(stddev.squeeze())
        mean_handler = TorchParameterInteractions(self.layer.params["mean"], self.config)
        mean_handler.update(mean.squeeze())


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
