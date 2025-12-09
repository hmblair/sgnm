"""
Dataset abstractions for SGNM training.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import os
import h5py
import torch
import ciffy

from .config import DataConfig, NUCLEOTIDE_DICT


@dataclass
class Sample:
    """A single training sample."""

    name: str
    """Sample identifier (e.g., PDB chain ID)."""

    polymer: ciffy.Polymer
    """Structural data (preprocessed)."""

    sequence: torch.Tensor
    """Tokenized sequence (long tensor)."""

    reactivity: torch.Tensor
    """Target reactivity values, shape (L,) or (L, num_channels)."""

    mask: torch.Tensor
    """Boolean mask for valid positions."""


def tokenize(seq: str) -> torch.Tensor:
    """
    Convert sequence string to token tensor.

    Args:
        seq: RNA sequence string (A, C, G, U/T)

    Returns:
        Long tensor of token indices
    """
    return torch.tensor([NUCLEOTIDE_DICT.get(x.upper(), 0) for x in seq]).long()


class HDF5Dataset:
    """
    Dataset for HDF5 reactivity files paired with structure directories.

    Supports two HDF5 formats:
    - v1: Keys 'id_strings', 'r_norm', 'sequences'
    - v2: Keys 'ids', 'PDB130-2A3/reactivity', 'PDB130-DMS/reactivity'
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
    ) -> None:
        """
        Initialize dataset.

        Args:
            config: Data configuration
            split: One of "train", "val", "test"
        """
        self.config = config
        self.split = split

        # Load reactivity data
        self._load_reactivity()

        # Get structure file list
        self._load_structures()

        # Apply split
        self._apply_split()

    def _load_reactivity(self) -> None:
        """Load reactivity data from HDF5 file."""
        self.reactivity_data: dict[str, tuple[str, torch.Tensor]] = {}

        if not self.config.reactivity_path:
            return

        with h5py.File(self.config.reactivity_path, 'r') as f:
            if self.config.data_format == "v1":
                names = f['id_strings'][0].astype(str)
                reacs = torch.from_numpy(f['r_norm'][:])
                seqs = f['sequences'][0].astype(str)
                for name, seq, reac in zip(names, seqs, reacs):
                    self.reactivity_data[name] = (seq, reac)

            elif self.config.data_format == "v2":
                names = list(f['ids'][:].astype(str))
                reacs_2a3 = torch.from_numpy(f['PDB130-2A3/reactivity'][:])
                reacs_dms = torch.from_numpy(f['PDB130-DMS/reactivity'][:])
                for i, name in enumerate(names):
                    reac = torch.stack([reacs_2a3[i], reacs_dms[i]], dim=-1)
                    self.reactivity_data[name] = ("", reac)

    def _load_structures(self) -> None:
        """Enumerate available structure files."""
        if not self.config.structures_dir:
            self.structure_files = []
            return

        self.structure_files = [
            f for f in os.listdir(self.config.structures_dir)
            if f.endswith(('.cif', '.pdb', '.cifpy'))
        ]

    def _apply_split(self) -> None:
        """Partition data into train/val/test splits."""
        import random
        random.seed(self.config.seed)

        n = len(self.structure_files)
        indices = list(range(n))
        random.shuffle(indices)

        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        if self.split == "train":
            self.indices = indices[:train_end]
        elif self.split == "val":
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]

    def __len__(self) -> int:
        """Return number of samples in split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Sample | None:
        """
        Load and preprocess a single sample.

        Returns None if sample fails validation.
        """
        file_idx = self.indices[idx]
        filename = self.structure_files[file_idx]
        path = os.path.join(self.config.structures_dir, filename)

        try:
            poly = ciffy.load(path)

            # Filter by chain count
            if poly.size(ciffy.CHAIN) > self.config.max_chains:
                return None

            # Process each RNA chain
            for chain in poly.chains(ciffy.RNA):
                sample = self._process_chain(chain)
                if sample is not None:
                    return sample

            return None

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def _process_chain(self, chain: ciffy.Polymer) -> Sample | None:
        """Process a single RNA chain into a Sample."""
        cid = chain.id(0)

        if cid not in self.reactivity_data:
            return None

        stripped = chain.frame().strip()
        if stripped.empty():
            return None

        n_residues = stripped.size(ciffy.RESIDUE)
        if n_residues * 3 != stripped.coordinates.size(0):
            return None

        seq_str, reactivity = self.reactivity_data[cid]

        # Align reactivity to structure
        offset = self.config.offset
        low = reactivity.size(0) - offset - n_residues
        high = reactivity.size(0) - offset

        if low < 0:
            return None

        reactivity = reactivity[low:high]

        # Sequence validation (if available)
        if seq_str:
            seq_slice = seq_str[low:high]
            if seq_slice.lower() != stripped.str():
                return None

        # Trim ends
        trim = self.config.trim_ends
        if trim > 0 and reactivity.size(0) > 2 * trim:
            reactivity = reactivity[trim:-trim]

        # Tokenize sequence
        sequence = tokenize(stripped.str())
        if trim > 0 and sequence.size(0) > 2 * trim:
            sequence = sequence[trim:-trim]

        # Create mask for valid positions
        if reactivity.dim() == 1:
            mask = ~torch.isnan(reactivity)
        else:
            mask = ~torch.isnan(reactivity).any(dim=-1)

        return Sample(
            name=cid,
            polymer=stripped,
            sequence=sequence,
            reactivity=reactivity,
            mask=mask,
        )

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over valid samples (skipping None)."""
        for i in range(len(self)):
            sample = self[i]
            if sample is not None:
                yield sample


class ProfileLoader:
    """
    Utility class for loading profiles from various sources.
    """

    def __init__(self, path: str, format: str = "auto") -> None:
        """
        Initialize profile loader.

        Args:
            path: Path to profile file
            format: Format ("v1", "v2", or "auto" to detect)
        """
        self.path = path
        self.format = format
        self._profiles: dict[str, torch.Tensor] = {}
        self._loaded = False

    def _load(self) -> None:
        """Load profiles from file."""
        if self._loaded:
            return

        with h5py.File(self.path, 'r') as f:
            # Auto-detect format
            if self.format == "auto":
                if 'ids' in f:
                    self.format = "v2"
                else:
                    self.format = "v1"

            if self.format == "v1":
                names = f['id_strings'][0].astype(str)
                reacs = torch.from_numpy(f['r_norm'][:])
                for name, reac in zip(names, reacs):
                    self._profiles[name] = reac

            elif self.format == "v2":
                names = list(f['ids'][:].astype(str))
                if 'PDB130-2A3/reactivity' in f:
                    reacs = torch.from_numpy(f['PDB130-2A3/reactivity'][:])
                    for i, name in enumerate(names):
                        self._profiles[name] = reacs[i]

        self._loaded = True

    def get(self, name: str) -> torch.Tensor | None:
        """Get profile by name."""
        self._load()
        return self._profiles.get(name)

    def __getitem__(self, name: str) -> torch.Tensor:
        """Get profile by name (raises KeyError if not found)."""
        self._load()
        return self._profiles[name]

    def __contains__(self, name: str) -> bool:
        """Check if profile exists."""
        self._load()
        return name in self._profiles

    def keys(self) -> list[str]:
        """Get all profile names."""
        self._load()
        return list(self._profiles.keys())

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over (name, profile) pairs."""
        self._load()
        return iter(self._profiles.items())

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Get all profiles as a dictionary."""
        self._load()
        return self._profiles.copy()
