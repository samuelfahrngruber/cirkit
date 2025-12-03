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
device = torch.device("cpu")

def gmm_2d(n_components: int, seed = 42) -> Circuit:
    torch.manual_seed(seed)
    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        activation='softmax',
        initialization='uniform',
    ))

    g0 = GaussianLayer(Scope((0,)), n_components)
    g1 = GaussianLayer(Scope((1,)), n_components)
    prod = HadamardLayer(num_input_units=n_components, arity=2)
    sl = SumLayer(n_components, 1, 1, weight_factory=weight_factory)

    return Circuit(
        layers=[g0, g1, prod, sl],
        in_layers={
            g0: [],
            g1: [],
            prod: [g0, g1],
            sl: [prod],
        },
        outputs=[sl]
    )

def create_and_compile_gmm(n_components: int):
    sym_gmm = gmm_2d(n_components=n_components)
    ctx = PipelineContext(
        backend='torch',
        semiring='lse-sum',
        fold=True,
        optimize=False
    )
    with ctx:
        torch_gmm = compile(sym_gmm)
    return torch_gmm

def run_experiment(n_components = 8, num_epochs = 50, use_em = True):
    circuit = create_and_compile_gmm(n_components)

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
