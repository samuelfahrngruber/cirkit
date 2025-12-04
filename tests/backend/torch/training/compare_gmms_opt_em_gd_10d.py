import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons, make_blobs, load_diabetes
from sklearn.model_selection import train_test_split
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
data_train = load_diabetes()["data"]
data_train, data_test = train_test_split(data_train, train_size=0.8)
dataset_train = TensorDataset(torch.tensor(data_train).to(torch.float32))
dataset_test = TensorDataset(torch.tensor(data_test).to(torch.float32))
batch_size = n_samples
train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

device = get_torch_device()

def run_experiment(n_components = 20, num_epochs = 100, use_em = True):
    circuit = create_and_compile_gmm(n_components=n_components, n_dims=10)

    em = FullBatchEM(circuit)
    gd = optim.SGD(circuit.parameters(), lr=0.001, momentum=0.9)

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
    title="Full-Batch Gradient Descent (GD) vs. Expectation-Maximization (EM) on Diabetes dataset (10-Dimensional)",
    xlabel="Epochs",
    ylabel="Average Log-Likelihood",
)
