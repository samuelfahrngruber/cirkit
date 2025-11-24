
import itertools
import time

from sklearn.covariance import log_likelihood
from torch.onnx.ops import symbolic

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.symbolic.io import plot_circuit
from cirkit.templates import data_modalities, utils
from cirkit.pipeline import compile
# from notebooks.datasets import sample_rings
import matplotlib.pyplot as plt
import torch
import numpy as np

from cirkit.backend.torch.layers import TorchInnerLayer, TorchSumLayer, TorchGaussianLayer

torch.manual_seed(42)

from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.utils.scope import Scope
from cirkit.symbolic.layers import GaussianLayer, HadamardLayer, SumLayer, KroneckerLayer


device = torch.device("cpu")
def new_circuit():

    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        initialization='uniform'
    ))
    mean_factory = utils.parameterization_to_factory(utils.Parameterization(
        initialization='normal',
        initialization_kwargs={"mean": -2, "stddev": 1}
    ))
    stddev_factory = utils.parameterization_to_factory(utils.Parameterization(
        initialization='normal',
        initialization_kwargs={"mean": 2, "stddev": 1}
    ))

    g0 = GaussianLayer(Scope((0,)), 3, mean_factory=mean_factory, stddev_factory=stddev_factory)
    g1 = GaussianLayer(Scope((1,)), 3, mean_factory=mean_factory, stddev_factory=stddev_factory)
    prod = HadamardLayer(num_input_units=3, arity=2)
    sl = SumLayer(3, 1, 1, weight_factory=weight_factory)

    symbolic_circuit = Circuit(
        layers=[g0, g1, prod, sl],
        in_layers={
            g0: [],
            g1: [],
            prod: [g0, g1],
            sl: [prod],
        },
        outputs=[sl]
    )
    pctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=False)
    circuit = compile(symbolic_circuit, pctx)
    circuit = circuit.to(device)
    return symbolic_circuit, circuit

symbolic_circuit, circuit = new_circuit()

def plot_circuit_distribution_2d(circuit, data=None, ax=plt):
    x = np.float32(np.linspace(-8, 8, 100))
    y = np.float32(np.linspace(-8, 8, 100))

    X, Y = np.meshgrid(x, y)

    xy = itertools.product(x, y)
    xy = torch.tensor(list(xy))
    # Z = circuit(xy).detach().numpy().reshape(100, 100)
    Z = circuit(xy).detach().numpy().reshape(100, 100)


    ax.contourf(X, Y, Z)
    # ax.colorbar()

    if data is not None:
        ax.scatter(data[:, 0], data[:, 1], color="red", marker="x")

# plot_circuit_distribution_2d(circuit)

class AutoGridPlotter:
    def __init__(self, cols=3, rows=3):
        self.fig = plt.figure(figsize=(12, 12))
        self.axs = self.fig.subplots(rows, cols).flatten()
        self.idx = 0
        self.n = len(self.axs)
        self.rows = rows
        self.cols = cols

    def next_ax(self):
        ax = self.axs[self.idx]
        self.idx = self.idx + 1
        ax.title.set_text(f"i={self.idx}")
        if self.idx > self.n:
            self.show()
            self.fig = plt.figure()
            self.axs = self.fig.subplots(self.rows, self.cols).flatten()
            self.idx = 0
            self.n = len(self.axs)
        return ax

    def show(self):
        plt.show()



def update_params_nested(params: TorchParameter, new_values: torch.Tensor):
    for p in params.outputs:
        if type(p) == TorchTensorParameter:
            p.update_params(new_values.squeeze())

def get_params_nested(params: TorchParameter):
    for p in params.outputs:
        if type(p) == TorchTensorParameter:
            return p()

class FullBatchEM:
    def __init__(self, circuit):
        self.circuit = circuit

        self.sum_layer_weights = {}
        self.leaf_layer_outputs = {}

        self.sum_x_n_sn = {}
        self.total_counts = {}
        self.sum_x_suff_stats_x = {}
        self.sum_x_suff_stats_x_2 = {}
        self.sum_x_p_l = {}

        for l in circuit.modules():
            if isinstance(l, TorchSumLayer):
                self.sum_layer_weights[l] = l.weight
            if isinstance(l, TorchGaussianLayer):
                self.leaf_layer_outputs[l] = None


    def _layer_fn(self, layer, *inputs):
        output = layer(*inputs)
        if isinstance(layer, TorchGaussianLayer):
            output.retain_grad()
            self.leaf_layer_outputs[layer] = output
        return output

    def e_weights(self, log_likelihoods_per_sample):
        for sum_layer, weights in self.sum_layer_weights.items():
            with torch.no_grad():
                n_sn = get_params_nested(weights).grad
                sum_x_n_sn = torch.sum(n_sn, dim=2)
                self.sum_x_n_sn[sum_layer] = sum_x_n_sn

    def m_weights(self):
        for sum_layer, weights in self.sum_layer_weights.items():
            with torch.no_grad():
                w_sn = get_params_nested(weights)
                sum_x_n_sn = self.sum_x_n_sn[sum_layer]

                w_sn_new = w_sn * sum_x_n_sn / sum_x_n_sn.sum()
                w_sn_new_reparam = w_sn_new # torch.softmax(w_sn_new, dim=2)

                print(f"{w_sn_new=}")

                update_params_nested(weights, w_sn_new_reparam)

    def e_leaves(self, log_likelihoods_per_sample, data_full):
        """
        Args:
            data_full: Train data; Shape [Samples, Features]
        """
        for leaf, outputs in self.leaf_layer_outputs.items():
            with torch.no_grad():
                p_l = outputs.grad # [Inputs, Samples, Outputs]
                p_l = p_l.permute(2, 1, 0) # [Outputs, Samples, Inputs]

                x = data_full[:,leaf.scope_idx].squeeze() # [Samples, Features]
                x_2 = x ** 2 # [Samples, Features]

                sum_x_suff_stats_x = torch.sum(p_l * x, dim=1) # [Outputs, Inputs]
                sum_x_suff_stats_x_2 = torch.sum(p_l * x_2, dim=1) # [Outputs, Inputs]

                sum_x_p_l = torch.sum(p_l, dim=1) # [Outputs, Inputs]

                self.sum_x_suff_stats_x[leaf] = sum_x_suff_stats_x
                self.sum_x_suff_stats_x_2[leaf] = sum_x_suff_stats_x_2
                self.sum_x_p_l[leaf] = sum_x_p_l

    def m_leaves(self):
        for leaf in self.leaf_layer_outputs.keys():
            with torch.no_grad():
                sum_x_suff_stats_x = self.sum_x_suff_stats_x[leaf] # [Outputs, Inputs]
                sum_x_suff_stats_x_2 = self.sum_x_suff_stats_x_2[leaf] # [Outputs, Inputs]
                sum_x_p_l = self.sum_x_p_l[leaf] # [Outputs, Inputs]

                x = sum_x_suff_stats_x / sum_x_p_l
                x_2 = sum_x_suff_stats_x_2 / sum_x_p_l

                mean = x
                var = x_2 - x ** 2
                var = var.clamp(min=1e-3)

                stddev = torch.sqrt(var)

                mean = mean.permute(1, 0)
                stddev = stddev.permute(1, 0)

                print(f"{mean=} {var=}")

                update_params_nested(leaf.params["mean"], mean)
                update_params_nested(leaf.params["stddev"], stddev)


    def step(self, data):
        self.circuit.zero_grad()
        log_likelihoods_per_sample = self.circuit.evaluate(data, module_fn=self._layer_fn)
        log_likelihoods_per_sample.sum().backward()
        self.e_weights(log_likelihoods_per_sample)
        self.e_leaves(log_likelihoods_per_sample, data)
        self.m_weights()
        self.m_leaves()
        return log_likelihoods_per_sample.sum()

em = FullBatchEM(circuit)
data = torch.tensor([[1.0, 1.0], [0.8, 1.2], [2.0, 2.0], [2.1, 2.2], [2.2, 2.2], [2.1, 2.1], [-3, -3], [-3.1, -2.8]])
em_losses = []
grid_plotter = AutoGridPlotter()
for i in range(9):
    # em.e_step(torch.from_numpy(ring_samples))

    # data = torch.tensor([[1.0, 1.0]])
    # data = torch.from_numpy(ring_samples)
    ll = em.step(data)
    plot_circuit_distribution_2d(circuit, data, ax=grid_plotter.next_ax())
    em_losses.append(ll.detach().numpy())
grid_plotter.show()

plt.plot(range(len(em_losses)), em_losses)
plt.show()

