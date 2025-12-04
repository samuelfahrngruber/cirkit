import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from cirkit.backend.torch.training.em import FullBatchEM
from cirkit.pipeline import PipelineContext, compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer, HadamardLayer
from cirkit.templates import utils
from cirkit.utils.scope import Scope
from tests.backend.torch.training.utils import get_torch_device, create_and_compile_gmm, detect_decreasing_likelihood, \
    plot_avg_ll_curves

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

def run_experiment(n_components = 8, num_epochs = 50, use_em = True):
    circuit = create_and_compile_gmm(n_components=n_components, n_dims=2)

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
final_model_gd, avg_lls_gd = run_experiment(use_em=False)

plot_avg_ll_curves(
    [avg_lls_em, avg_lls_gd],
    labels=["EM", "GD"],
    markers=["o", "v"],
    colors=["r", "b"],
    title="Full-Batch Gradient Descent (GD) vs. Expectation-Maximization (EM)",
    xlabel="Epochs",
    ylabel="Average Log-Likelihood",
)

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
