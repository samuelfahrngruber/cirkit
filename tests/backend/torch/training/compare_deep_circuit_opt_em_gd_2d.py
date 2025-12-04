import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from cirkit.backend.torch.training.em import FullBatchEM
from cirkit.pipeline import PipelineContext, compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.io import plot_circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer, HadamardLayer
from cirkit.templates import utils
from cirkit.templates.data_modalities import tabular_data
from cirkit.utils.scope import Scope
from tests.backend.torch.training.utils import get_torch_device, create_and_compile_gmm, compile_circuit, \
    detect_decreasing_likelihood

# setup data
n_samples = 1_000
n_samples_train = 250
data_train, _ = make_moons(n_samples=n_samples, noise=0.1)
data_test, _ = make_moons(n_samples=n_samples_train, noise=0.1)
# data_train, _ = make_blobs(centers=3, cluster_std=0.5, random_state=0)
# data_test, _ = make_blobs(centers=3, cluster_std=0.5, random_state=0)
dataset_train = TensorDataset(torch.tensor(data_train).to(torch.float32))
dataset_test = TensorDataset(torch.tensor(data_test).to(torch.float32))
batch_size = n_samples
train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

# setup torch
device = get_torch_device()

def make_circuit():
    torch.manual_seed(3)
    symbolic_circuit = tabular_data(
        region_graph="random-binary-tree",
        num_features=2,
        input_layers={"name": "gaussian", "args": {}},
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        sum_weight_param=utils.Parameterization(
            activation="softmax", initialization="normal"
        ),
        use_mixing_weights=True,
    )
    plot_circuit(symbolic_circuit)
    return symbolic_circuit

def run_experiment(num_epochs = 50, use_em = True):
    sym_circuit = make_circuit()
    circuit = compile_circuit(sym_circuit)

    em = FullBatchEM(circuit)
    gd = optim.Adam(circuit.parameters(), lr=0.1)

    avg_lls = []

    for epoch_idx in range(num_epochs):
        for batch_idx, (batch,) in enumerate(train_dataloader):
            batch = batch.to(device)

            avg_ll = None

            if use_em:
                log_likelihoods = em.forward(batch)
                avg_ll = log_likelihoods.detach().mean()
                em.backward_latent_posterior()
                em.expectation()
                em.maximization()
                em.zero_grad()
            else:
                log_likelihoods = circuit(batch)
                loss = -log_likelihoods.mean()
                avg_ll = -loss.detach()
                loss.backward()
                gd.step()
                gd.zero_grad()

            if epoch_idx % 5 == 0:
                print(f"{epoch_idx=} {batch_idx=} {avg_ll=}")
            avg_lls.append(avg_ll)

    return circuit, avg_lls


final_model_em, avg_lls_em = run_experiment(use_em=True)
detect_decreasing_likelihood(avg_lls_em)
final_model_gd, avg_lls_gd = run_experiment(use_em=False)

plt.figure(figsize=(12, 6))
ticks = range(len(avg_lls_em))
plt.plot(ticks, avg_lls_em, label="EM", marker="s")
plt.plot(ticks, avg_lls_gd, label="GD", marker="v")
plt.title("Gradient Descent (GD) vs. Expectation-Maximization (EM)")
plt.xlabel("Full-Batch Epochs")
plt.ylabel("Average Log-Likelihood")
plt.legend()
plt.show()

def plot_circuit_distribution_2d(circuit, samples, plot_samples=True):
    plt.figure()

    x = np.linspace(samples[:, 0].min() - 0.2, samples[:, 0].max() + 0.2, 100, dtype=np.float32)
    y = np.linspace(samples[:, 1].min() - 0.2, samples[:, 1].max() + 0.2, 100, dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy = torch.tensor(xy, dtype=torch.float32)

    Z = circuit(xy).detach().exp().numpy().reshape(100, 100)

    plt.contourf(X, Y, Z)

    if plot_samples:
        plt.scatter(samples[:, 0], samples[:, 1], color="red", marker="+", alpha=0.5)

    plt.show()

plot_circuit_distribution_2d(final_model_em, samples=data_train)
