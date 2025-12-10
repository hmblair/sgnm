"""
Data loading utilities for the Structure VAE.

Provides structure-only datasets (no reactivity data needed).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import os
import torch
import ciffy

from ..data import tokenize
from .training import StructureSample, AllAtomStructureSample


@dataclass
class StructureDataConfig:
    """Configuration for structure-only datasets."""

    structures_dir: str = ""
    """Directory containing structure files (.cif, .pdb)."""

    max_length: int = 500
    """Maximum number of residues (filter out longer structures)."""

    min_length: int = 10
    """Minimum number of residues (filter out shorter structures)."""

    max_chains: int = 1
    """Maximum number of chains per structure."""

    train_split: float = 0.8
    """Fraction of data for training."""

    val_split: float = 0.1
    """Fraction of data for validation."""

    test_split: float = 0.1
    """Fraction of data for testing."""

    seed: int = 42
    """Random seed for reproducible splits."""


class StructureOnlyDataset:
    """
    Dataset for loading RNA structures without reactivity data.

    Loads structure files from a directory and extracts coordinates,
    sequence information, and optionally local coordinate frames.
    """

    def __init__(
        self,
        config: StructureDataConfig,
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

        self._load_files()
        self._apply_split()

    def _load_files(self) -> None:
        """Enumerate structure files in directory."""
        if not self.config.structures_dir:
            self.files = []
            return

        structures_dir = Path(self.config.structures_dir)
        if not structures_dir.exists():
            raise ValueError(f"Directory not found: {structures_dir}")

        self.files = [
            f.name
            for f in structures_dir.iterdir()
            if f.suffix in (".cif", ".pdb", ".cifpy")
        ]
        self.files.sort()  # Deterministic ordering

    def _apply_split(self) -> None:
        """Partition files into train/val/test splits."""
        import random

        random.seed(self.config.seed)

        n = len(self.files)
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
        """Return number of files in split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> StructureSample | None:
        """
        Load and process a single structure.

        Returns None if structure fails validation.
        """
        file_idx = self.indices[idx]
        filename = self.files[file_idx]
        path = os.path.join(self.config.structures_dir, filename)

        try:
            poly = ciffy.load(path)

            # Filter by chain count
            if poly.size(ciffy.CHAIN) > self.config.max_chains:
                return None

            # Process RNA chains
            for chain in poly.chains(ciffy.RNA):
                sample = self._process_chain(chain, filename)
                if sample is not None:
                    return sample

            return None

        except Exception as e:
            # Silently skip problematic files during iteration
            return None

    def _process_chain(
        self, chain: ciffy.Polymer, filename: str
    ) -> StructureSample | None:
        """Process a single RNA chain into a StructureSample."""
        stripped = chain.frame().strip()

        if stripped.empty():
            return None

        n_residues = stripped.size(ciffy.RESIDUE)

        # Length filters
        if n_residues < self.config.min_length:
            return None
        if n_residues > self.config.max_length:
            return None

        # Extract residue centers
        _, coords = stripped.center(ciffy.RESIDUE)

        # Tokenize sequence
        seq_str = stripped.str()
        node_types = tokenize(seq_str)

        return StructureSample(
            name=f"{filename}:{chain.id(0)}",
            coords=coords,
            node_types=node_types,
            frames=None,
        )

    def __iter__(self) -> Iterator[StructureSample]:
        """Iterate over valid samples (skipping None)."""
        for i in range(len(self)):
            sample = self[i]
            if sample is not None:
                yield sample


class StructureListDataset:
    """
    Dataset from a list of structure file paths.

    Useful for custom data loading scenarios.
    """

    def __init__(
        self,
        file_paths: list[str],
        max_length: int = 500,
        min_length: int = 10,
    ) -> None:
        """
        Args:
            file_paths: List of paths to structure files
            max_length: Maximum residue count
            min_length: Minimum residue count
        """
        self.file_paths = file_paths
        self.max_length = max_length
        self.min_length = min_length

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> StructureSample | None:
        path = self.file_paths[idx]

        try:
            poly = ciffy.load(path)

            for chain in poly.chains(ciffy.RNA):
                stripped = chain.frame().strip()

                if stripped.empty():
                    continue

                n = stripped.size(ciffy.RESIDUE)
                if n < self.min_length or n > self.max_length:
                    continue

                _, coords = stripped.center(ciffy.RESIDUE)
                node_types = tokenize(stripped.str())

                return StructureSample(
                    name=Path(path).stem,
                    coords=coords,
                    node_types=node_types,
                    frames=None,
                )

            return None

        except Exception:
            return None

    def __iter__(self) -> Iterator[StructureSample]:
        for i in range(len(self)):
            sample = self[i]
            if sample is not None:
                yield sample


# Nucleotide type mapping
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


class AllAtomDataset:
    """
    Dataset for loading all-atom RNA structures.

    Extracts full atomic coordinates and builds mappings between
    atoms and residues for hierarchical VAE processing.
    """

    def __init__(
        self,
        config: StructureDataConfig,
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

        self._load_files()
        self._apply_split()

    def _load_files(self) -> None:
        """Enumerate structure files in directory."""
        if not self.config.structures_dir:
            self.files = []
            return

        structures_dir = Path(self.config.structures_dir)
        if not structures_dir.exists():
            raise ValueError(f"Directory not found: {structures_dir}")

        self.files = [
            f.name
            for f in structures_dir.iterdir()
            if f.suffix in (".cif", ".pdb", ".cifpy")
        ]
        self.files.sort()

    def _apply_split(self) -> None:
        """Partition files into train/val/test splits."""
        import random

        random.seed(self.config.seed)

        n = len(self.files)
        indices = list(range(n))
        random.shuffle(indices)

        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        if self.split == "train":
            self.indices = indices[:train_end]
        elif self.split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> AllAtomStructureSample | None:
        """
        Load and process a single all-atom structure.

        Returns None if structure fails validation.
        """
        file_idx = self.indices[idx]
        filename = self.files[file_idx]
        path = os.path.join(self.config.structures_dir, filename)

        try:
            poly = ciffy.load(path)

            if poly.size(ciffy.CHAIN) > self.config.max_chains:
                return None

            for chain in poly.chains(ciffy.RNA):
                sample = self._process_chain_allatom(chain, filename)
                if sample is not None:
                    return sample

            return None

        except Exception:
            return None

    def _process_chain_allatom(
        self, chain: ciffy.Polymer, filename: str
    ) -> AllAtomStructureSample | None:
        """Process a single RNA chain into an AllAtomStructureSample."""
        stripped = chain.frame().strip()

        if stripped.empty():
            return None

        n_residues = stripped.size(ciffy.RESIDUE)

        if n_residues < self.config.min_length:
            return None
        if n_residues > self.config.max_length:
            return None

        # Get all-atom coordinates
        atom_coords = stripped.coordinates  # (A, 3)
        n_atoms = atom_coords.shape[0]

        # Get atom types from ciffy
        atom_types = stripped.atoms  # ciffy enum tensor

        # Build residue indices and atoms_per_residue
        residue_indices = []
        atoms_per_residue = []

        # ciffy provides residue sizes
        residue_sizes = stripped.sizes(ciffy.RESIDUE)
        offset = 0
        for res_idx, n_res_atoms in enumerate(residue_sizes):
            residue_indices.extend([res_idx] * n_res_atoms)
            atoms_per_residue.append(n_res_atoms)
            offset += n_res_atoms

        residue_indices = torch.tensor(residue_indices, dtype=torch.long)
        atoms_per_residue = torch.tensor(atoms_per_residue, dtype=torch.long)

        # Get residue types from sequence
        seq_str = stripped.str()
        residue_types = torch.tensor(
            [NUC_TO_IDX.get(s, 0) for s in seq_str],
            dtype=torch.long,
        )

        return AllAtomStructureSample(
            name=f"{filename}:{chain.id(0)}",
            atom_coords=atom_coords,
            atom_types=atom_types,
            residue_indices=residue_indices,
            residue_types=residue_types,
            atoms_per_residue=atoms_per_residue,
            polymer=stripped,
        )

    def __iter__(self) -> Iterator[AllAtomStructureSample]:
        for i in range(len(self)):
            sample = self[i]
            if sample is not None:
                yield sample
