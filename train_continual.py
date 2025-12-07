"""
Continual learning experiment: train on 2-digit addition, then switch to 3-digit.
Compares Adam vs WorkAdam on distribution shift handling.
"""

import csv
import os
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf

from gpt import GPT, GPTConfig
from adder.dataset import ArithmeticDataset
from thermo_adam import WorkAdam


@dataclass
class Phase:
    ndigit: int
    iters: int


def get_work_stats(optimizer) -> Optional[dict]:
    """Extract work/mass statistics from WorkAdam optimizer."""
    if not isinstance(optimizer, WorkAdam):
        return None

    total_mass = 0.0
    total_params = 0
    max_mass = 0.0

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state and "mass" in optimizer.state[p]:
                mass = optimizer.state[p]["mass"]
                total_mass += mass.sum().item()
                total_params += mass.numel()
                max_mass = max(max_mass, mass.max().item())

    return {
        "mean_mass": total_mass / total_params if total_params > 0 else 0,
        "max_mass": max_mass,
    }


def eval_accuracy(model, dataset, device, max_batches=None):
    """Evaluate accuracy on a dataset (uses dataset's max_ndigit for decoding)."""
    model.eval()
    max_ndigit = dataset.max_ndigit
    factors = torch.tensor([[10**i for i in range(max_ndigit + 1)][::-1]]).to(device)
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

    correct, total = 0, 0
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            x = x.to(device)
            # Input is first 2*max_ndigit positions (both operands, zero-padded)
            d1d2 = x[:, : max_ndigit * 2]
            # Generate max_ndigit+1 answer tokens
            d1d2d3 = model.generate(d1d2, max_ndigit + 1, sample=False)
            d3 = d1d2d3[:, -(max_ndigit + 1) :].flip(1)

            # Decode the padded operands and prediction
            d1i = (d1d2[:, :max_ndigit] * factors[:, 1:]).sum(1)
            d2i = (d1d2[:, max_ndigit : max_ndigit * 2] * factors[:, 1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i

            correct += (d3i_pred == d3i_gt).sum().item()
            total += x.size(0)

            if max_batches and b + 1 >= max_batches:
                break

    model.train()
    return correct / total if total > 0 else 0


def run_continual_experiment(
    optimizer_name: str = "adamw",
    phases: list[Phase] = None,
    seed: int = 42,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    eval_every: int = 100,
    work_dir: str = None,
    # WorkAdam-specific
    rho: float = 0.99,
    mass_init: float = 1e-3,
):
    """
    Run continual learning experiment.

    Args:
        optimizer_name: "adamw" or "workadam"
        phases: List of Phase(ndigit, iters) defining the curriculum
        seed: Random seed
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay
        eval_every: Evaluate every N iterations
        work_dir: Output directory
    """
    if phases is None:
        phases = [Phase(ndigit=2, iters=3000), Phase(ndigit=3, iters=5000)]

    # Setup
    random.seed(seed)
    torch.manual_seed(seed)

    if work_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = f"./out/continual/{optimizer_name}/{timestamp}"
    os.makedirs(work_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # Create model sized for largest task
    max_ndigit = max(p.ndigit for p in phases)
    model_config = GPTConfig(
        block_size=3 * max_ndigit + 1,
        vocab_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        dropout=0.0,
        bias=True,
    )
    model = GPT(model_config).to(device)

    # Setup optimizer
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
        )
    elif optimizer_name == "workadam":
        optimizer = WorkAdam(
            model.parameters(),
            lr=lr,
            beta1=0.9,
            rho=rho,
            weight_decay=weight_decay,
            mass_init=mass_init,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Pre-create test datasets for all phases (for measuring forgetting)
    # All datasets padded to max_ndigit for consistent positional encoding
    test_datasets = {}
    for phase in phases:
        cfg = OmegaConf.create({"ndigit": phase.ndigit})
        test_datasets[phase.ndigit] = ArithmeticDataset(
            cfg, split="test", max_ndigit=max_ndigit
        )

    # CSV logging
    fieldnames = ["iter", "phase", "ndigit", "train_loss", "train_acc"]
    fieldnames += [f"test_acc_{p.ndigit}d" for p in phases]
    if optimizer_name == "workadam":
        fieldnames += ["mean_mass", "max_mass"]

    log_path = os.path.join(work_dir, "metrics.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    # Training loop
    global_iter = 0
    train_losses = []

    for phase_idx, phase in enumerate(phases):
        print(f"\n{'='*60}")
        print(
            f"Phase {phase_idx}: {phase.ndigit}-digit addition for {phase.iters} iters"
        )
        print(f"{'='*60}\n")

        # Create dataset for current phase (padded to max_ndigit)
        cfg = OmegaConf.create({"ndigit": phase.ndigit})
        train_dataset = ArithmeticDataset(cfg, split="train", max_ndigit=max_ndigit)

        train_loader = DataLoader(
            train_dataset,
            sampler=torch.utils.data.RandomSampler(
                train_dataset, replacement=True, num_samples=int(1e10)
            ),
            batch_size=batch_size,
            num_workers=0,
        )
        data_iter = iter(train_loader)

        model.train()
        for phase_iter in range(phase.iters):
            # Get batch
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            # Forward + backward
            _, loss = model(x, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # Logging
            if global_iter % 10 == 0:
                print(f"[Phase {phase_idx}] iter {global_iter}: loss {loss.item():.4f}")

            if global_iter % eval_every == 0:
                avg_loss = sum(train_losses) / len(train_losses) if train_losses else 0
                train_losses = []

                # Evaluate on current task
                train_acc = eval_accuracy(model, train_dataset, device, max_batches=5)

                # Evaluate on all tasks (current + previous)
                row = {
                    "iter": global_iter,
                    "phase": phase_idx,
                    "ndigit": phase.ndigit,
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                }
                for p in phases:
                    acc = eval_accuracy(model, test_datasets[p.ndigit], device)
                    row[f"test_acc_{p.ndigit}d"] = acc

                # Work stats (WorkAdam only)
                work_stats = get_work_stats(optimizer)
                if work_stats:
                    row.update(work_stats)

                log_writer.writerow(row)
                log_file.flush()

                # Print summary
                accs = [
                    f"{p.ndigit}d: {row[f'test_acc_{p.ndigit}d']*100:.1f}%"
                    for p in phases
                ]
                print(f"  -> Test accs: {', '.join(accs)}")
                if work_stats:
                    print(
                        f"  -> Mass: mean={work_stats['mean_mass']:.2e}, max={work_stats['max_mass']:.2e}"
                    )

            global_iter += 1

    log_file.close()
    print(f"\nSaved metrics to {log_path}")

    # Plot results
    plot_continual_metrics(work_dir, phases)

    return work_dir


def plot_continual_metrics(work_dir: str, phases: list[Phase]):
    """Plot continual learning metrics."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(work_dir, "metrics.csv"))

    n_plots = 2 + ("mean_mass" in df.columns)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Mark phase transitions
    phase_boundaries = []
    cumsum = 0
    for p in phases[:-1]:
        cumsum += p.iters
        phase_boundaries.append(cumsum)

    # Loss plot
    ax = axes[0]
    ax.plot(df["iter"], df["train_loss"], label="train")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    for b in phase_boundaries:
        ax.axvline(
            b,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="phase switch" if b == phase_boundaries[0] else "",
        )
    ax.legend()

    # Accuracy plot (all tasks)
    ax = axes[1]
    for p in phases:
        col = f"test_acc_{p.ndigit}d"
        ax.plot(df["iter"], df[col], label=f"{p.ndigit}-digit")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy (all tasks)")
    for b in phase_boundaries:
        ax.axvline(b, color="red", linestyle="--", alpha=0.7)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # Mass plot (WorkAdam only)
    if "mean_mass" in df.columns:
        ax = axes[2]
        ax.plot(df["iter"], df["mean_mass"], label="mean")
        ax.plot(df["iter"], df["max_mass"], label="max", alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mass")
        ax.set_title("WorkAdam Mass (work EMA)")
        ax.set_yscale("log")
        for b in phase_boundaries:
            ax.axvline(b, color="red", linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "metrics.png"), dpi=150)
    plt.close()
    print(f"Saved plot to {os.path.join(work_dir, 'metrics.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "workadam"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--rho", type=float, default=0.99, help="WorkAdam: EMA decay for mass"
    )
    parser.add_argument(
        "--mass_init", type=float, default=1e-3, help="WorkAdam: initial mass"
    )
    args = parser.parse_args()

    # Define curriculum: 2-digit -> 3-digit
    phases = [
        Phase(ndigit=2, iters=3000),
        Phase(ndigit=3, iters=5000),
    ]

    run_continual_experiment(
        optimizer_name=args.optimizer,
        phases=phases,
        seed=args.seed,
        lr=args.lr,
        rho=args.rho,
        mass_init=args.mass_init,
    )
