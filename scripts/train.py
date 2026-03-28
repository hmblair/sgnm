"""Launch SGNM training."""
from sgnm.config import ModelConfig, DataConfig, TrainConfig
from sgnm.training import train_sgnm

results = train_sgnm(
    model_config=ModelConfig(dim=32, layers=2),
    data_config=DataConfig(
        reactivity_path="/oak/stanford/groups/rhiju/sherlock/home/hmblair/data/analysis/map-seq/pdb/pdb130/profiles.h5",
        fasta_path="/oak/stanford/groups/rhiju/sherlock/home/hmblair/data/analysis/map-seq/pdb/pdb130/data/ref.fasta",
        structures_dir="/scratch/users/hmblair/data/rna",
        data_format="v2",
    ),
    train_config=TrainConfig(
        learning_rate=1e-2,
        max_epochs=100,
        warmup_epochs=5.0,
        device="cuda",
        checkpoint_dir="./checkpoints",
        wandb_project="sgnm",
        wandb_run="dim32-layers2",
    ),
)

print(f"Best val loss: {results.best_val_loss:.4f}")
print(f"Final epoch: {results.final_epoch}")
