import torch
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from cirkit.backend.torch.training.em import FullBatchEM
from cirkit.symbolic.io import plot_circuit
from cirkit.templates import utils
from cirkit.templates.data_modalities import tabular_data
from tests.backend.torch.training.utils import get_torch_device, compile_circuit, \
    plot_avg_ll_curves

# setup data
n_samples = 1_000
n_samples_train = 250
data_train = load_breast_cancer()["data"]
data_train, data_test = train_test_split(data_train, train_size=0.8)
dataset_train = TensorDataset(torch.tensor(data_train).to(torch.float32))
dataset_test = TensorDataset(torch.tensor(data_test).to(torch.float32))
batch_size = n_samples
train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)
# setup torch
device = get_torch_device()

def make_circuit(seed = 42):
    torch.manual_seed(seed)
    symbolic_circuit = tabular_data(
        region_graph="random-binary-tree",
        num_features=30,
        input_layers={"name": "gaussian", "args": {}},
        num_input_units=64,
        sum_product_layer="cp",
        num_sum_units=64,
        sum_weight_param=utils.Parameterization(
            activation="softmax", initialization="normal"
        ),
        use_mixing_weights=True,
    )
    plot_circuit(symbolic_circuit)
    return symbolic_circuit

def run_experiment(num_epochs = 200, use_em = True):
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
final_model_gd, avg_lls_gd = run_experiment(use_em=False)

plot_avg_ll_curves(
    [avg_lls_em, avg_lls_gd],
    labels=["EM", "GD"],
    markers=["o", "v"],
    colors=["r", "b"],
    title="Full-Batch Gradient Descent vs. Expectation-Maximization (EM)",
    xlabel="Epochs",
    ylabel="Average Log-Likelihood",
)
