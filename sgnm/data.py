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


def load_reactivity_index(
    reactivity_path: str,
    fasta_path: str,
    data_format: str = "v2",
) -> ReactivityIndex:
    """Build a ReactivityIndex from an HDF5 reactivity file and FASTA sequences.

    The HDF5 entries and FASTA entries must be in the same order.

    Args:
        reactivity_path: Path to HDF5 file with reactivity profiles.
        fasta_path: Path to FASTA file with probed sequences.
        data_format: HDF5 format ("v1" or "v2").

    Returns:
        Populated ReactivityIndex.
    """
    sequences = _read_fasta_sequences(fasta_path)

    index = ReactivityIndex()
    with h5py.File(reactivity_path, "r") as f:
        if data_format == "v1":
            names = f["id_strings"][0].astype(str)
            reacs = f["r_norm"][:]
            for name, seq, reac in zip(names, sequences, reacs):
                index.add(name, seq, reac)

        elif data_format == "v2":
            names = list(f["ids"][:].astype(str))
            reacs_2a3 = f["PDB130-2A3/reactivity"][:]
            reacs_dms = f["PDB130-DMS/reactivity"][:]
            for i, (name, seq) in enumerate(zip(names, sequences)):
                reac = np.stack([reacs_2a3[i], reacs_dms[i]], axis=-1)
                index.add(name, seq, reac)

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
        self.skip_counts.clear()
        yielded = 0
        for i in range(len(self)):
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
