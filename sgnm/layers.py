from __future__ import annotations
import itertools
import torch
import torch.nn as nn
import ciffy
from ciffy.enum import Adenosine, Cytosine, Guanosine, Uridine
from .gnm import _gnm_variances, _orientation_score, _local_frame

FRAME1 = torch.tensor([
    Adenosine.C2.value,
    Cytosine.C2.value,
    Guanosine.C2.value,
    Uridine.C2.value,
])
FRAME2 = torch.tensor([
    Adenosine.C4.value,
    Cytosine.C4.value,
    Guanosine.C4.value,
    Uridine.C4.value,
])
FRAME3 = torch.tensor([
    Adenosine.C6.value,
    Cytosine.C6.value,
    Guanosine.C6.value,
    Uridine.C6.value,
])
FRAMES = torch.cat([FRAME1, FRAME2, FRAME3])


def _normalize(x: torch.Tensor) -> torch.Tensor:

    min = x.min()
    max = (x - min).max()

    return (x - min) / max


def _base_frame(poly: ciffy.Polymer) -> torch.Tensor:
    """
    Get pairwise base orientations based on the C2-C4-C6 frame.
    """

    poly1 = poly.get_by_name(FRAME1)
    poly2 = poly.get_by_name(FRAME2)
    poly3 = poly.get_by_name(FRAME3)

    diff1 = poly1.coordinates - poly2.coordinates
    diff2 = poly1.coordinates - poly3.coordinates
    return _local_frame(diff1, diff2)


class RadialBasisFunctions(nn.Module):
    """
    Compute one or more radial basis functions.
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


class BaseSGNM(nn.Module):

    def __init__(
        self: SGNM,
    ) -> None:

        super().__init__()

    def forward(
        self: SGNM,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:

        dists = torch.cdist(coords, coords)
        emb = _gnm_variances(dists)

        return _normalize(emb)

    def ciffy(
        self: SGNM,
        poly: ciffy.Polymer,
    ) -> torch.Tensor:

        poly = poly.get_by_name(FRAMES)
        _, coords = poly.center(ciffy.RESIDUE)

        return self(coords)


class SGNM(BaseSGNM):

    def __init__(
        self: SGNM,
        dim: int,
        out_dim: int = 1,
        layers: int = 1
    ) -> None:

        super().__init__()

        self.rbf1 = RadialBasisFunctions(dim)
        self.linear1 = DenseNetwork(dim, out_dim, [dim] * layers)

        self.rbf2 = RadialBasisFunctions(dim)
        self.linear2 = DenseNetwork(dim, out_dim, [dim] * layers)

    def embed1(
        self: SGNM,
        dists: torch.Tensor,
    ) -> torch.Tensor:

        adj = self.rbf1(dists)
        adj = self.linear1(adj).squeeze(-1)

        return _normalize(adj)

    def embed2(
        self: SGNM,
        scores: torch.Tensor,
    ) -> torch.Tensor:

        adj = self.rbf2(scores)
        adj = self.linear2(adj).squeeze(-1)

        return _normalize(adj)

    def forward(
        self: SGNM,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:

        dists = torch.cdist(coords, coords)
        emb = self.embed1(dists)

        if frames is not None:
            scores = _orientation_score(frames)
            emb = emb * self.embed2(scores)

        emb = _gnm_variances(emb)
        return _normalize(emb)

    def ciffy(
        self: SGNM,
        poly: ciffy.Polymer,
    ) -> torch.Tensor:

        poly = poly.get_by_name(FRAMES)
        poly, coords = poly.center(ciffy.RESIDUE)
        frames = _base_frame(poly)

        return self(coords, frames)

    @classmethod
    def load(
        cls: type[SGNM],
        path: str | None = None,
    ) -> BaseSGNM:
        """
        Load the module with the appropriate sizes given the weights file.
        """

        if path is None:
            return BaseSGNM()

        weights = torch.load(path, weights_only=True)
        dim = weights['rbf1.mu'].size(0)

        module = SGNM(dim)
        module.load_state_dict(weights)

        return module


class Score(nn.Module):

    def __init__(
        self: Score,
        weights: str | None = None,
    ) -> None:

        super().__init__()

        self.module = SGNM.load(weights)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad = False

    def forward(
        self: SGNM,
        profile: torch.Tensor,
        coords: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the MAE between the provided and predicted SHAPE profiles.
        Also return the predicted profile itself.
        """

        pred = self.module(coords, frames)
        ix = ~torch.isnan(profile)

        return torch.abs(pred[ix] - profile[ix]).mean(), pred

    def ciffy(
        self: SGNM,
        profile: torch.Tensor,
        poly: ciffy.Polymer,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        pred = self.module.ciffy(poly)
        ix = ~torch.isnan(profile)

        return torch.abs(pred[ix] - profile[ix]).mean(), pred
