"""
Neural network building blocks for SGNM.
"""
from __future__ import annotations
import itertools
import torch
import torch.nn as nn


class RadialBasisFunctions(nn.Module):
    """
    Compute one or more radial basis functions with learnable parameters.
    """

    def __init__(
        self: RadialBasisFunctions,
        num_functions: int,
    ) -> None:

        super().__init__()

        self.mu = nn.Parameter(
            torch.randn(num_functions),
            requires_grad=True,
        )

        self.sigma = nn.Parameter(
            torch.randn(num_functions),
            requires_grad=True,
        )

    def forward(
        self: RadialBasisFunctions,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass each element of x though all the radial basis functions.
        """

        exp = (x[..., None] - self.mu) * self.sigma ** 2
        return torch.exp(-exp ** 2) * self.sigma.abs()


class DenseNetwork(nn.Module):
    """
    A configurable multi-layer fully-connected network.
    """

    def __init__(
        self: DenseNetwork,
        in_size: int,
        out_size: int,
        hidden_sizes: list = [],
        bias: bool = True,
        dropout: float = 0.0,
        activation: nn.Module = nn.LeakyReLU(0.2),
    ) -> None:

        super().__init__()

        features = [in_size] + hidden_sizes + [out_size]

        layers = []
        for l1, l2 in itertools.pairwise(features):
            layers.append(nn.Linear(l1, l2, bias))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)
