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

from ..config import FRAME1, FRAME2, FRAME3, NUCLEOTIDE_DICT
from ..data import tokenize
from .training import StructureSample


def _base_frame(poly: ciffy.Polymer) -> torch.Tensor:
    """
    Extract local coordinate frames from nucleobase C2-C4-C6 atoms.

    Args:
        poly: ciffy Polymer object

    Returns:
        (N, 3, 3) local frame matrices
    """
    from ..gnm import _local_frame

    seq = poly.str()
    nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

    N = poly.size(ciffy.RESIDUE)
    coords = poly.coordinates.view(N, -1, 3)

    frame1_idx = torch.tensor([FRAME1[nuc_map.get(s, 0)] for s in seq])
    frame2_idx = torch.tensor([FRAME2[nuc_map.get(s, 0)] for s in seq])
    frame3_idx = torch.tensor([FRAME3[nuc_map.get(s, 0)] for s in seq])

    c2 = coords[torch.arange(N), frame1_idx]
    c4 = coords[torch.arange(N), frame2_idx]
    c6 = coords[torch.arange(N), frame3_idx]

    v1 = c4 - c2
    v2 = c6 - c2

    return _local_frame(v1, v2)


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

    include_frames: bool = True
    """Whether to compute local coordinate frames."""

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

        # Check coordinate consistency
        if n_residues * 3 != stripped.coordinates.size(0):
            return None

        # Extract residue centers
        _, coords = stripped.center(ciffy.RESIDUE)

        # Tokenize sequence
        seq_str = stripped.str()
        node_types = tokenize(seq_str)

        # Optionally compute frames
        frames = None
        if self.config.include_frames:
            try:
                frames = _base_frame(stripped)
            except Exception:
                # Frames may fail for some structures
                pass

        return StructureSample(
            name=f"{filename}:{chain.id(0)}",
            coords=coords,
            node_types=node_types,
            frames=frames,
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
        include_frames: bool = True,
    ) -> None:
        """
        Args:
            file_paths: List of paths to structure files
            max_length: Maximum residue count
            min_length: Minimum residue count
            include_frames: Whether to compute local frames
        """
        self.file_paths = file_paths
        self.max_length = max_length
        self.min_length = min_length
        self.include_frames = include_frames

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

                frames = None
                if self.include_frames:
                    try:
                        frames = _base_frame(stripped)
                    except Exception:
                        pass

                return StructureSample(
                    name=Path(path).stem,
                    coords=coords,
                    node_types=node_types,
                    frames=frames,
                )

            return None

        except Exception:
            return None

    def __iter__(self) -> Iterator[StructureSample]:
        for i in range(len(self)):
            sample = self[i]
            if sample is not None:
                yield sample
