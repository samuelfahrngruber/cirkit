from abc import ABC
from typing import Mapping, Generic, TypeVar

import torch
from black.trans import defaultdict
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchSumLayer, TorchGaussianLayer, TorchLayer
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter

L = TypeVar("L", bound=TorchLayer)

def update_params_nested(params: TorchParameter, new_values: torch.Tensor):
    for p in params.outputs:
        if type(p) == TorchTensorParameter:
            p.update_params(new_values.squeeze())

class AbstractEMCell(Generic[L], ABC):
    def __init__(self, layer: L):
        self.layer = layer
        self.sufficient_statistics: dict[str, Tensor] = {}

    def layer_fn(self, layer, *inputs):
        self.inputs = torch.stack(inputs)
        self.outputs = layer(*inputs)
        self.outputs.retain_grad()
        return self.outputs

    def expectation(self):
        raise NotImplementedError()

    def maximization(self):
        raise NotImplementedError()


class SumEMCell(AbstractEMCell[TorchSumLayer]):
    def __init__(self, layer: TorchSumLayer):
        super().__init__(layer)
        self.weight = self.layer.weight.outputs[0]._ptensor

    def expectation(self):
        self.sufficient_statistics["n"] = self.weight.grad.sum(dim=2)

    def maximization(self):
        n = self.sufficient_statistics["n"]
        new_weight = self.weight * n / n.sum()
        print(f"{new_weight=}")

        update_params_nested(self.layer.weight, new_weight)
        self.weight.copy_(new_weight)


class GaussianEMCell(AbstractEMCell[TorchGaussianLayer]):
    def __init__(self, layer: TorchGaussianLayer):
        super().__init__(layer)
        self.mean = self.layer.params["mean"]()
        self.stddev = self.layer.params["stddev"]()

    def expectation(self):
        p_l = self.outputs.grad # [Inputs, Samples, Outputs]
        p_l = p_l.permute(2, 1, 0)  # [Outputs, Samples, Inputs]

        # old impl:
        # x = data_full[:,leaf.scope_idx[-1]] # [Samples, Features]
        data_full = self.inputs
        x = self.inputs # data_full[..., self.layer.scope_idx].permute(1, 0, 2)  # [???] taken from cirkit/backend/torch/circuits.py:66
        x_2 = x ** 2  # [Samples, Features]

        sum_x_suff_stats_x = torch.sum(p_l * x, dim=2)  # [Outputs, Inputs]
        sum_x_suff_stats_x_2 = torch.sum(p_l * x_2, dim=2)  # [Outputs, Inputs]

        sum_x_p_l = torch.sum(p_l, dim=1)  # [Outputs, Inputs]

        self.sufficient_statistics["x"] = sum_x_suff_stats_x
        self.sufficient_statistics["x^2"] = sum_x_suff_stats_x_2
        self.sufficient_statistics["p_l"] = sum_x_p_l

    def maximization(self):
        sum_x_suff_stats_x = self.sufficient_statistics["x"] # [Outputs, Inputs]
        sum_x_suff_stats_x_2 = self.sufficient_statistics["x^2"]  # [Outputs, Inputs]
        sum_x_p_l = self.sufficient_statistics["p_l"]  # [Outputs, Inputs]

        x = sum_x_suff_stats_x / sum_x_p_l  # [Outputs, Inputs]
        x_2 = sum_x_suff_stats_x_2 / sum_x_p_l  # [Outputs, Inputs]

        mean = x  # [Outputs, Inputs]
        var = x_2 - x ** 2  # [Outputs, Inputs]
        var = var.clamp(min=1e-3)  # [Outputs, Inputs]

        stddev = torch.sqrt(var)  # [Outputs, Inputs]

        print(f"{mean=} {var=}")

        update_params_nested(self.layer.params["mean"], mean)
        update_params_nested(self.layer.params["stddev"], stddev)

        self.mean.copy_(mean.squeeze())
        self.stddev.copy_(stddev.squeeze())


def create_em_cell(layer: TorchLayer) -> AbstractEMCell[TorchLayer] | None:
    match layer:
        case TorchSumLayer():
            return SumEMCell(layer)
        case TorchGaussianLayer():
            return GaussianEMCell(layer)




class FullBatchEM:
    def __init__(self, circuit: TorchCircuit):
        self.circuit = circuit
        self.em_cells: dict[TorchLayer, AbstractEMCell[TorchLayer]] = {}
        for layer in circuit.layers:
            self.em_cells[layer] = create_em_cell(layer)
        self.log_likelihoods_per_sample = None

    def zero_grad(self):
        self.circuit.zero_grad()

    def forward(self, data):
        self.log_likelihoods_per_sample = self.circuit.evaluate(data, module_fn=self.layer_fn)
        return self.log_likelihoods_per_sample

    def layer_fn(self, layer, *inputs):
        em_cell = self.em_cells.get(layer)
        if em_cell:
            return em_cell.layer_fn(layer, *inputs)
        return layer(*inputs)

    def backward(self):
        self.log_likelihoods_per_sample.sum().backward()

    def expectation(self):
        with torch.no_grad():
            for layer, cell in self.em_cells.items():
                if cell:
                    cell.expectation()

    def maximization(self):
        with torch.no_grad():
            for layer, cell in self.em_cells.items():
                if cell:
                    cell.maximization()
