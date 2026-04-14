"""
Grid search over GatedThermoAdamW hyperparameters on Split MNIST.

Sweeps lr × gate_scale × gate_c, runs N_SEEDS seeds each, reports
mean ± std final average accuracy. Saves a summary CSV and prints a
ranked table at the end.

Usage:
    python sweep_gated.py
    python sweep_gated.py --seeds 3 --iters 1500
"""

import argparse
import csv
import itertools
import os
import statistics
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from gated_thermo_adamw import GatedThermoAdamW

# ── Grid ──────────────────────────────────────────────────────────────────────
LR_GRID         = [1e-3, 3e-4]
GATE_SCALE_GRID = [50, 200, 500, 2000]
GATE_C_GRID     = [0.05, 0.3, 1.0]

# ── Model / Data (duplicated from train_split_mnist for self-containment) ─────

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class MLPBackbone(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class SplitMNISTModel(nn.Module):
    def __init__(self, hidden_dim=256, n_tasks=5):
        super().__init__()
        self.backbone = MLPBackbone(hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_tasks)])
    def forward(self, x, task_id):
        return self.heads[task_id](self.backbone(x))


def get_task_dataset(mnist, task_idx):
    cls_a, cls_b = TASKS[task_idx]
    mask = (mnist.targets == cls_a) | (mnist.targets == cls_b)
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    subset = Subset(mnist, indices)
    class _Relabeled(torch.utils.data.Dataset):
        def __len__(self): return len(subset)
        def __getitem__(self, i):
            x, y = subset[i]
            return x, int(y != cls_a)
    return _Relabeled()


def eval_task(model, dataset, task_id, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x, task_id).argmax(1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total else 0.0


def run_single(lr, gate_scale, gate_c, seed, iters_per_task, device,
               train_tasks, test_tasks, batch_size=64):
    torch.manual_seed(seed)
    model = SplitMNISTModel().to(device)
    optimizer = GatedThermoAdamW(model.parameters(), lr=lr,
                                  gate_scale=gate_scale, c=gate_c)

    for task_idx in range(5):
        train_ds = train_tasks[task_idx]
        loader = DataLoader(
            train_ds,
            sampler=torch.utils.data.RandomSampler(
                train_ds, replacement=True, num_samples=int(1e9)),
            batch_size=batch_size, num_workers=0,
        )
        data_iter = iter(loader)
        model.train()
        for _ in range(iters_per_task):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            logits = model(x, task_idx)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(loss=loss.item())

    accs = [eval_task(model, test_tasks[t], t, device) for t in range(5)]
    return sum(accs) / 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_mnist = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_mnist  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_tasks = [get_task_dataset(train_mnist, t) for t in range(5)]
    test_tasks  = [get_task_dataset(test_mnist,  t) for t in range(5)]

    grid = list(itertools.product(LR_GRID, GATE_SCALE_GRID, GATE_C_GRID))
    total_runs = len(grid) * args.seeds
    print(f"\nSweeping {len(grid)} configs × {args.seeds} seeds = {total_runs} runs\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"./out/sweep_gated/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")

    results = []  # list of dicts

    run_idx = 0
    for lr, gate_scale, gate_c in grid:
        seed_accs = []
        for seed in range(args.seeds):
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] lr={lr}  scale={gate_scale}  c={gate_c}  seed={seed}", end="  ", flush=True)
            acc = run_single(lr, gate_scale, gate_c, seed,
                             args.iters, device, train_tasks, test_tasks, args.batch_size)
            seed_accs.append(acc)
            print(f"avg={acc*100:.1f}%")

        mean = statistics.mean(seed_accs)
        std  = statistics.stdev(seed_accs) if len(seed_accs) > 1 else 0.0
        results.append(dict(lr=lr, gate_scale=gate_scale, gate_c=gate_c,
                            mean=mean, std=std,
                            seed_accs=",".join(f"{a:.4f}" for a in seed_accs)))

    # Sort by mean desc
    results.sort(key=lambda r: -r["mean"])

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr","gate_scale","gate_c","mean","std","seed_accs"])
        writer.writeheader()
        writer.writerows(results)

    # Print ranked table
    print(f"\n{'='*65}")
    print(f"{'Rank':<5} {'lr':<8} {'scale':<8} {'c':<6} {'Mean%':<9} {'Std%':<7}")
    print(f"{'-'*65}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r['lr']:<8} {r['gate_scale']:<8} {r['gate_c']:<6} "
              f"{r['mean']*100:<9.2f} {r['std']*100:<7.2f}")

    print(f"\nTop config: lr={results[0]['lr']}  gate_scale={results[0]['gate_scale']}  "
          f"gate_c={results[0]['gate_c']}  → {results[0]['mean']*100:.2f}% ± {results[0]['std']*100:.2f}%")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
