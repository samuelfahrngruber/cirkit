from typing import Mapping, Sequence

import numpy as np
import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.pipeline import PipelineContext, compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, HadamardLayer, SumLayer, Layer
from cirkit.templates import utils
from cirkit.utils.scope import Scope
import matplotlib.pyplot as plt


def get_torch_device():
    return torch.device("cpu")


def make_gmm(n_components: int, n_dims = 2, seed = 42) -> Circuit:
    torch.manual_seed(seed)
    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        activation='softmax',
        initialization='uniform',
    ))

    gaussians = [GaussianLayer(Scope((dim,)), n_components) for dim in range(n_dims)]

    prod = HadamardLayer(num_input_units=n_components, arity=n_dims)
    suml = SumLayer(n_components, 1, 1, weight_factory=weight_factory)

    in_layers: dict[Layer, Sequence[Layer]] = {
        prod: gaussians,
        suml: [prod]
    }
    for g in gaussians:
        in_layers[g] = []

    return Circuit(layers=[prod, suml] + gaussians, in_layers=in_layers, outputs=[suml])

def compile_circuit(sym_circuit: Circuit) -> TorchCircuit:
    ctx = PipelineContext(
        backend='torch',
        semiring='lse-sum',
        fold=True,
        optimize=False
    )
    with ctx:
        circuit = compile(sym_circuit)
    return circuit

def create_and_compile_gmm(n_components: int, n_dims: int, seed = 42):
    sym_gmm = make_gmm(n_components=n_components, n_dims=n_dims, seed=seed)
    torch_gmm = compile_circuit(sym_gmm)
    return torch_gmm

def detect_decreasing_likelihood(likelihood_curve):
    likelihood_curve = np.array(likelihood_curve)
    deltas = likelihood_curve[1:] - likelihood_curve[:-1]
    decreased = deltas < 0
    by = deltas * decreased
    if np.any(decreased):
        print(f"""WARNING: log likelihood increased:
        {likelihood_curve=}
        {deltas=}
        {decreased=}
        {by=}""")

def plot_avg_ll_curves(avg_ll_curves, labels, markers, colors, title, xlabel, ylabel):
    curve_count = len(avg_ll_curves)
    if not (len(labels) == curve_count and len(markers) == curve_count and len(colors) == curve_count):
        raise ValueError("Cannot plot LL curves")

    plt.figure(figsize=(12, 6))
    ticks = range(len(avg_ll_curves[0]))
    for avg_lls, label, marker, color in zip(avg_ll_curves, labels, markers, colors):
        detect_decreasing_likelihood(avg_lls)
        plt.plot(ticks, avg_lls, label=label, marker=marker, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
