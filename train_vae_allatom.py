#!/usr/bin/env python
"""
Train an all-atom VAE on RNA structures with ball sampling visualization.

Usage:
    python train_vae_allatom.py

This script:
1. Trains an all-atom VAE on RNA structures from PDB130
2. At each epoch, samples 10 structures in a ball around a reference
3. Saves sampled structures as PDB files using ciffy
"""
import os
import random
from pathlib import Path

import torch

from sgnm.vae import (
    AllAtomStructureVAE,
    VAEConfig,
    AllAtomVAETrainer,
    VAETrainConfig,
    AllAtomDataset,
    StructureDataConfig,
)
from sgnm.vae.losses import kabsch_rmsd


def main():
    # Configuration
    structures_dir = "/home/hmblair/data/pdb130"
    output_dir = Path("./vae_allatom_outputs")
    output_dir.mkdir(exist_ok=True)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # 1. Setup data
    # =========================================================================
    print("\n=== Loading data ===")
    data_config = StructureDataConfig(
        structures_dir=structures_dir,
        max_length=150,  # Smaller for all-atom (more atoms per residue)
        min_length=20,
        max_chains=1,
        train_split=0.9,
        val_split=0.1,
        test_split=0.0,
        seed=42,
    )

    train_dataset = AllAtomDataset(data_config, split="train")
    val_dataset = AllAtomDataset(data_config, split="val")

    print(f"Training files: {len(train_dataset)}")
    print(f"Validation files: {len(val_dataset)}")

    # =========================================================================
    # 2. Setup model
    # =========================================================================
    print("\n=== Setting up model ===")
    vae_config = VAEConfig(
        hidden_dim=128,
        latent_dim=32,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_rbf=16,
        cutoff=10.0,
        kl_weight=0.001,
        kl_warmup_epochs=10,
        # All-atom specific
        num_atom_types=149,
        num_atom_encoder_layers=3,
        num_residue_decoder_layers=3,
        num_atom_decoder_layers=2,
        atom_cutoff=5.0,
        local_rmsd_weight=0.1,
    )

    model = AllAtomStructureVAE(vae_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # =========================================================================
    # 3. Select reference structure for ball sampling
    # =========================================================================
    print("\n=== Selecting reference structure for sampling ===")

    # Find a valid sample from validation set
    reference_sample = None
    for sample in val_dataset:
        if sample is not None and sample.polymer is not None:
            reference_sample = sample
            print(f"Selected reference: {sample.name}")
            print(f"  Residues: {sample.residue_types.size(0)}")
            print(f"  Atoms: {sample.atom_coords.size(0)}")
            break

    if reference_sample is None:
        print("No valid reference sample found!")
        return

    # Move reference to device
    reference_on_device = None

    # Create sampling output directory
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Save original reference structure
    reference_sample.polymer.write(str(samples_dir / "reference.cif"))
    print(f"Saved reference to: {samples_dir / 'reference.cif'}")

    # =========================================================================
    # 4. Define epoch callback for ball sampling
    # =========================================================================
    def sampling_callback(model, epoch, metrics):
        """Sample structures in a ball around reference at end of each epoch."""
        nonlocal reference_on_device

        model.eval()

        # Move reference to device on first call
        if reference_on_device is None:
            from sgnm.vae.training import AllAtomStructureSample
            reference_on_device = AllAtomStructureSample(
                name=reference_sample.name,
                atom_coords=reference_sample.atom_coords.to(device),
                atom_types=reference_sample.atom_types.to(device),
                residue_indices=reference_sample.residue_indices.to(device),
                residue_types=reference_sample.residue_types.to(device),
                atoms_per_residue=reference_sample.atoms_per_residue.to(device),
                polymer=reference_sample.polymer,
            )

        with torch.no_grad():
            # Sample 10 structures in ball around reference
            polymers = model.sample_around_reference(
                reference_on_device,
                num_samples=10,
                radius=1.0,
            )

            # Also get reconstruction
            recon_poly = model.reconstruct_polymer(reference_on_device)

            # Compute reconstruction RMSD
            recon_coords = model.reconstruct(reference_on_device)
            rmsd = kabsch_rmsd(
                recon_coords, reference_on_device.atom_coords
            ).item()

        # Save samples for this epoch
        epoch_dir = samples_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)

        # Save reconstruction
        recon_poly.write(str(epoch_dir / "reconstruction.cif"))

        # Save ball samples
        for i, poly in enumerate(polymers):
            poly.write(str(epoch_dir / f"sample_{i:02d}.cif"))

        print(f"  Ball sampling saved | Recon RMSD: {rmsd:.3f} Å")

        model.train()

    # =========================================================================
    # 5. Train with callback
    # =========================================================================
    print("\n=== Training ===")
    train_config = VAETrainConfig(
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_epochs=50,
        gradient_clip=1.0,
        patience=15,
        device=device,
        checkpoint_dir=str(output_dir / "checkpoints"),
        save_every=10,
        save_best=True,
        log_every=50,
    )

    trainer = AllAtomVAETrainer(
        model=model,
        vae_config=vae_config,
        train_config=train_config,
        train_data=train_dataset,
        val_data=val_dataset,
        epoch_callback=sampling_callback,
    )

    results = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Final epoch: {results.final_epoch}")
    print(f"  Best val loss: {results.best_val_loss:.4f}")
    print(f"  Total steps: {results.total_steps}")

    # Save final model
    model_path = output_dir / "vae_allatom_final.pth"
    model.save(str(model_path))
    print(f"  Model saved to: {model_path}")

    print("\n=== Done! ===")
    print(f"All outputs saved to: {output_dir}")
    print(f"Ball samples per epoch in: {samples_dir}")


if __name__ == "__main__":
    main()
