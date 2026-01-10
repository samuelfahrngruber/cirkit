import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator


def plot_2D(*fns, title=None, xmin=-2.5, xmax=2.5, nbins=15):
    x_min, x_max = xmin, xmax
    y_min, y_max = xmin, xmax

    dx, dy = 0.01, 0.01

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(x_min, x_max + dy, dy),
    slice(y_min, y_max + dx, dx)]
    xy = torch.from_numpy(np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))).float()

    fns = [fn for fn in fns if fn is not None]

    ncols = len(fns)
    if ncols == 0:
        return

    fig, axs = plt.subplots(ncols=ncols, figsize=(5 * ncols, 5))

    if ncols == 1:
        axs = [axs]

    for fn, ax in zip(fns, axs):
        with torch.no_grad():
            z = fn(xy)
        z = z.view(y.shape).numpy()
        z = z[:-1, :-1]

        cmap = plt.colormaps['PiYG']

        levels = MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
        cf = ax.contourf(
            x[:-1, :-1] + dx / 2.,
            y[:-1, :-1] + dy / 2.,
            z,
            levels=levels, cmap=cmap
        )
        ax.set_aspect('equal', 'box')

    fig.colorbar(cf, ax=axs)
    if title is not None:
        if ncols == 1:
            axs[-1].set_title(title)
        else:
            fig.suptitle(title)

    plt.show()

import torch.distributions as D
import math

radius = 2  # Distance of the centers from the origin
K = 8  # Number of clusters
mus = torch.tensor([
    [math.cos(2*math.pi*n / K) for n in range(K)],
    [math.sin(2*math.pi*n / K) for n in range(K)]
]).T * radius
sigma = .2  # Standard deviation

mix = D.Categorical(torch.ones(K,))
comp = D.Independent(D.Normal(mus, sigma), 1)
gmm = D.MixtureSameFamily(mix, comp)

def sample_points(n_points):
    return gmm.sample((n_points,))

#plt.scatter(*sample_points(1000).unbind(-1))
#plt.gca().set_aspect('equal', 'box')
#plt.title('Original samples')
#plt.show()

def true_density(xy):
    return gmm.log_prob(xy).exp()

# plot_2D(true_density, title='Original density')

from cirkit.symbolic.circuit import Circuit, Scope
from cirkit.symbolic.layers import GaussianLayer, SumLayer, HadamardLayer
from cirkit.templates import utils


def build_symbolic_circuit() -> Circuit:
    # This parametrizes the mixture weights such that they add up to one.
    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        # activation='softmax',  # Parameterize the sum weights by using a softmax activation
        initialization='uniform'  # Initialize the sum weights by sampling from a standard normal distribution
    ))

    # We introduce one more mixture than in the original model
    # Again, SGD/Adam is not the best way to fit a (shallow) Gaussian mixture model
    units = K # + 1

    g0 = GaussianLayer(Scope((0,)), units)
    g1 = GaussianLayer(Scope((1,)), units)
    prod = HadamardLayer(num_input_units=units, arity=2)
    sl = SumLayer(units, 1, 1, weight_factory=weight_factory)

    return Circuit(
        layers=[g0, g1, prod, sl],  # Layers that appear in the circuit (i.e. nodes in the graph)
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            g0: [],
            g1: [],
            prod: [g0, g1],
            sl: [prod],
        },
        outputs=[sl]  # Nodes that are returned by the circuit
    )

# Build a symbolic complex circuit by overparameterizing a Quad-Tree (4) region graph, which is structured-decomposable
symbolic_circuit = build_symbolic_circuit()

# Print which structural properties the circuit satisfies
print(f'Structural properties:')
print(f'  - Smoothness: {symbolic_circuit.is_smooth}')
print(f'  - Decomposability: {symbolic_circuit.is_decomposable}')
print(f'  - Structured-decomposability: {symbolic_circuit.is_structured_decomposable}')

from cirkit.symbolic.io import plot_circuit

plot_circuit(symbolic_circuit)

import random
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

dataset_size = 10000

# Set some seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the torch device to use
device = torch.device('cpu')

# Load the MNIST data set and data loaders
data_train = TensorDataset(sample_points(dataset_size))
data_test = TensorDataset(sample_points(dataset_size//10))

# Instantiate the training and testing data loaders
train_dataloader = DataLoader(data_train, shuffle=True, batch_size=10000)
test_dataloader = DataLoader(data_test, shuffle=False, batch_size=10000)

from cirkit.pipeline import PipelineContext, compile

# Instantiate the pipeline context
ctx = PipelineContext(
    backend='torch',  # Choose PyTorch as compilation backend
    # ---- Use the evaluation semiring (R, +, x), where + is the numerically stable LogSumExp and x is the sum ---- #
    semiring='lse-sum',
    # ------------------------------------------------------------------------------------------------------------- #
    fold=True,     # Fold the circuit to better exploit GPU parallelism
    optimize=False  # Optimize the layers of the circuit
)

with ctx:  # Compile the circuits computing log |c(X)| and log |Z|
    circuit = compile(symbolic_circuit)

def model_density(xy):
    return circuit(xy).exp()

plot_2D(model_density, title='Model density (before training)', xmin=-3, xmax=3)


from cirkit.backend.torch.training.em import FullBatchEM

circuit = circuit.to(device)

em = FullBatchEM(circuit)

num_epochs = 30
step_idx = 0
running_loss = 0.0
running_samples = 0

optimizer = optim.Adam(circuit.parameters(), lr=0.01)

use_em = True

avg_lls = []

for epoch_idx in range(num_epochs):
    for i, (batch,) in enumerate(train_dataloader):
        # The circuit expects an input of shape (batch_dim, num_variables)
        batch = batch.to(device)

        if use_em:
            log_likelihoods = em.forward(batch)
            avg_ll = log_likelihoods.detach().mean().numpy()
            avg_lls.append(avg_ll)
            loss = -avg_ll
            em.backward_latent_posterior()
            em.expectation()
            em.maximization()
            em.zero_grad()
        else:
            log_likelihoods = circuit(batch)
            loss = -torch.mean(log_likelihoods)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss * len(batch)
        running_samples += len(batch)
        step_idx += 1
        if step_idx % 5 == 0:
            average_nll = running_loss / running_samples
            print(f"Step {step_idx}: Average NLL: {average_nll:.3f}")
            running_loss = 0.0
            running_samples = 0

def model_density(xy):
    return circuit(xy).exp()

final_weights = circuit.layers._modules['2'].weight()
print(f"final weights: {final_weights}")
print(f"total weights: {final_weights.sum()}")

plt.plot(range(len(avg_lls)), avg_lls)

plot_2D(model_density, true_density, title='Model / True density (after training)', xmin=-3, xmax=3)
plot_2D(model_density, title='Model density (after training)', xmin=-3, xmax=3)
plot_2D(true_density, title='True density (after training)', xmin=-3, xmax=3)

