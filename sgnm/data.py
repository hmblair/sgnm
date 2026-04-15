"""
Dataset abstractions for SGNM training.

Uses ciffy's PolymerDataset for structure loading and ReactivityIndex
for matching reactivity profiles to structures by sequence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import h5py
import torch
import ciffy
from ciffy.biochemistry import Scale, Molecule
from ciffy.nn import PolymerDataset
from ciffy.rna import ReactivityIndex

from .config import DataConfig


@dataclass
class Sample:
    """A single training sample."""

    name: str
    """Sample identifier (e.g., PDB chain ID)."""

    polymer: ciffy.Polymer
    """Structural data (preprocessed)."""

    reactivity: torch.Tensor
    """Target reactivity values, shape (L,) or (L, num_channels)."""

    mask: torch.Tensor
    """Boolean mask for valid positions."""

    def to(self, device: torch.device | str) -> "Sample":
        """Move all tensors and polymer to a device."""
        polymer = self.polymer
        if str(device).startswith("cuda"):
            polymer = polymer.cuda()
        return Sample(
            name=self.name,
            polymer=polymer,
            reactivity=self.reactivity.to(device),
            mask=self.mask.to(device),
        )


def _read_fasta_sequences(path: str) -> list[str]:
    """Read sequences from a FASTA file, in order."""
    sequences: list[str] = []
    current = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append(current)
                current = ""
            else:
                current += line
    if current:
        sequences.append(current)
    return sequences


def _parse_h5_spec(spec: str) -> tuple[str, str]:
    """Parse ``path/to/file.h5:dataset_key`` into (path, key).

    If no ``:key`` suffix is given, defaults to ``"reactivity"``.
    """
    # Split on last colon that isn't part of a Windows drive letter
    # and where the right side doesn't start with / (i.e. not a path sep)
    idx = spec.rfind(":")
    if idx > 0 and not spec[idx + 1:].startswith(("/", "\\")):
        return spec[:idx], spec[idx + 1:]
    return spec, "reactivity"


def load_reactivity_index(
    reactivity_paths: str | list[str],
    fasta_path: str,
) -> ReactivityIndex:
    """Build a ReactivityIndex from HDF5 reactivity file(s) and a FASTA.

    Each entry is a path to an HDF5 file optionally suffixed with
    ``:dataset_key`` (defaults to ``"reactivity"``). Each entry provides
    one channel of reactivity data with shape ``(N, L)``. When multiple
    entries are given the channels are stacked to produce ``(N, L, C)``.

    Examples::

        # Single file, default key
        load_reactivity_index("2a3.h5", "seqs.fasta")

        # Two channels from separate files
        load_reactivity_index(["2a3.h5", "dms.h5"], "seqs.fasta")

        # Two channels from the same file, different keys
        load_reactivity_index(
            ["profiles.h5:PDB130-2A3/reactivity",
             "profiles.h5:PDB130-DMS/reactivity"],
            "seqs.fasta",
        )

    Args:
        reactivity_paths: Path(s) to HDF5 datasets, one per condition.
        fasta_path: Path to FASTA file with probed sequences.

    Returns:
        Populated ReactivityIndex.
    """
    if isinstance(reactivity_paths, str):
        reactivity_paths = [reactivity_paths]

    sequences = _read_fasta_sequences(fasta_path)

    channels = []
    for spec in reactivity_paths:
        path, key = _parse_h5_spec(spec)
        with h5py.File(path, "r") as f:
            channels.append(f[key][:])

    if len(channels) == 1:
        reactivities = channels[0]  # (N, L)
    else:
        reactivities = np.stack(channels, axis=-1)  # (N, L, C)

    index = ReactivityIndex()
    for i, seq in enumerate(sequences):
        index.add(str(i), seq, reactivities[i])

    return index


class ReactivityDataset:
    """Dataset pairing RNA structures with reactivity profiles.

    Uses ciffy's PolymerDataset for structure loading/filtering/splitting
    and ReactivityIndex for sequence-based reactivity matching.
    """

    def __init__(
        self,
        structures: PolymerDataset,
        index: ReactivityIndex,
    ) -> None:
        self.structures = structures
        self.index = index
        self.skip_counts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> Sample | None:
        polymer = self.structures[idx]
        if polymer is None:
            self._skip("load_failed")
            return None

        try:
            stripped = polymer.strip()
            if stripped.empty():
                self._skip("empty")
                return None

            match = self.index.match(stripped)
            if match is None:
                self._skip("no_reactivity_match")
                return None

            reactivity = match.reactivity

            # Mask for valid (non-NaN) positions
            if reactivity.dim() == 1:
                mask = ~torch.isnan(reactivity)
            else:
                mask = ~torch.isnan(reactivity).any(dim=-1)

            return Sample(
                name=match.name,
                polymer=stripped,
                reactivity=reactivity,
                mask=mask,
            )

        except Exception as e:
            self._skip(f"error:{type(e).__name__}")
            return None

    def _skip(self, reason: str) -> None:
        self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1

    def __iter__(self) -> Iterator[Sample]:
        import random
        self.skip_counts.clear()
        yielded = 0
        indices = list(range(len(self)))
        random.shuffle(indices)
        for i in indices:
            sample = self[i]
            if sample is not None:
                yielded += 1
                yield sample
        self._yielded = yielded

    def summary(self) -> str:
        total = len(self)
        yielded = getattr(self, "_yielded", "?")
        parts = [f"{yielded}/{total} samples yielded"]
        for reason, count in sorted(self.skip_counts.items(), key=lambda x: -x[1]):
            parts.append(f"  {reason}: {count}")
        return "\n".join(parts)


class ProfileLoader:
    """Utility class for loading profiles from various sources."""

    def __init__(self, path: str, format: str = "auto") -> None:
        self.path = path
        self.format = format
        self._profiles: dict[str, torch.Tensor] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        with h5py.File(self.path, "r") as f:
            if self.format == "auto":
                self.format = "v2" if "ids" in f else "v1"

            if self.format == "v1":
                names = f["id_strings"][0].astype(str)
                reacs = torch.from_numpy(f["r_norm"][:])
                for name, reac in zip(names, reacs):
                    self._profiles[name] = reac

            elif self.format == "v2":
                names = list(f["ids"][:].astype(str))
                if "PDB130-2A3/reactivity" in f:
                    reacs = torch.from_numpy(f["PDB130-2A3/reactivity"][:])
                    for i, name in enumerate(names):
                        self._profiles[name] = reacs[i]

        self._loaded = True

    def get(self, name: str) -> torch.Tensor | None:
        self._load()
        return self._profiles.get(name)

    def __getitem__(self, name: str) -> torch.Tensor:
        self._load()
        return self._profiles[name]

    def __contains__(self, name: str) -> bool:
        self._load()
        return name in self._profiles

    def keys(self) -> list[str]:
        self._load()
        return list(self._profiles.keys())

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        self._load()
        return iter(self._profiles.items())

    def to_dict(self) -> dict[str, torch.Tensor]:
        self._load()
        return self._profiles.copy()
