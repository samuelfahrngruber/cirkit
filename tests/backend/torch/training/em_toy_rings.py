
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.training.em import FullBatchEM
from cirkit.pipeline import PipelineContext
from cirkit.pipeline import compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, HadamardLayer, SumLayer
from cirkit.templates import utils
from cirkit.utils.scope import Scope
from notebooks.datasets import sample_rings

device = torch.device("cpu")

def new_circuit():

    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        initialization='uniform'
    ))
    mean_factory = utils.parameterization_to_factory(utils.Parameterization(
        initialization='normal',
        initialization_kwargs={"mean": -2, "stddev": 1}
    ))


    g0 = GaussianLayer(Scope((0,)), 4, mean_factory=mean_factory)
    g1 = GaussianLayer(Scope((1,)), 4, mean_factory=mean_factory)
    prod = HadamardLayer(num_input_units=4, arity=2)
    sl = SumLayer(4, 1, 1, weight_factory=weight_factory)

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
    pctx = PipelineContext(backend="torch", semiring="lse-sum", fold=False, optimize=False)
    circuit = compile(symbolic_circuit, pctx)
    circuit = circuit.to(device)
    return symbolic_circuit, circuit

symbolic_circuit, circuit = new_circuit()

def plot_circuit_distribution_2d(circuit, data=None, ax=plt, exp=False):
    x = np.float32(np.linspace(-10, 10, 100))
    y = np.float32(np.linspace(-10, 10, 100))

    X, Y = np.meshgrid(x, y)

    xy = itertools.product(x, y)
    xy = torch.tensor(list(xy))
    Z = circuit(xy).detach()
    if exp:
        Z = Z.exp()
    Z = Z.numpy().reshape(100, 100)


    ax.contourf(X, Y, Z, levels=50)
    # ax.colorbar()

    if data is not None:
        ax.scatter(data[:, 1], data[:, 0], color="red", marker="+", alpha=0.5)

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
        if self.idx >= self.n:
            self.show()
            self.fig = plt.figure()
            self.axs = self.fig.subplots(self.rows, self.cols).flatten()
            self.idx = 0
            self.n = len(self.axs)
        return ax

    def show(self):
        plt.show()

em = FullBatchEM(circuit)

ring_samples = sample_rings(500, dim=2, radia=[1, 3, 7], sigma=0.5)
ring_samples = np.float32(ring_samples)

data = torch.from_numpy(ring_samples)
data = torch.tensor([[1.0, 1.0], [0.8, 1.2], [2.0, 2.0], [2.1, 2.2], [2.2, 2.2], [2.1, 2.1], [-5, -3], [-3.1, -2.8], [-4, -3], [-1, -3], [0, 5]])

em_losses = []
grid_plotter = AutoGridPlotter()
for i in range(8):
    em.zero_grad()
    lls = em.forward(data)
    em_losses.append(lls.detach().mean().numpy())
    em.backward_latent_posterior()
    em.expectation()
    em.maximization()
    plot_circuit_distribution_2d(circuit, data, ax=grid_plotter.next_ax())


plt.figure()
plt.plot(range(len(em_losses)), em_losses)

plt.figure()
plot_circuit_distribution_2d(circuit)

plt.figure()
plot_circuit_distribution_2d(circuit, exp=True)

plt.show()

