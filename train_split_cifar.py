"""
Split CIFAR-10 continual learning benchmark — task-incremental setting.

5 sequential binary classification tasks:
  Task 0: airplane  vs automobile  (0 vs 1)
  Task 1: bird      vs cat         (2 vs 3)
  Task 2: deer      vs dog         (4 vs 5)
  Task 3: frog      vs horse       (6 vs 7)
  Task 4: ship      vs truck       (8 vs 9)

Model: small CNN backbone (3→32→64→128 conv) + 5 task-specific heads.
Task-incremental protocol: correct head selected at test time.

EWC baseline: ~88-93% average accuracy.
AdamW fine-tuning baseline: ~60-75% (significant catastrophic forgetting).
"""

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from thermo_adam import WorkAdam
from gated_thermo_adamw import GatedThermoAdamW
from snr_adam import SNRAdam
from adabelief import AdaBelief


# ── Model ─────────────────────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """Small CNN: 32×32×3 → 256-dim features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Flatten → FC
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.out_dim = 256

    def forward(self, x):
        return self.net(x)


class SplitCIFARModel(nn.Module):
    def __init__(self, n_tasks=5):
        super().__init__()
        self.backbone = CNNBackbone()
        self.heads = nn.ModuleList(
            [nn.Linear(self.backbone.out_dim, 2) for _ in range(n_tasks)]
        )

    def forward(self, x, task_id):
        return self.heads[task_id](self.backbone(x))


# ── Dataset helpers ───────────────────────────────────────────────────────────

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
TASK_NAMES = [
    "airplane vs automobile",
    "bird vs cat",
    "deer vs dog",
    "frog vs horse",
    "ship vs truck",
]


def get_task_dataset(cifar, task_idx):
    cls_a, cls_b = TASKS[task_idx]
    targets = torch.tensor(cifar.targets)
    mask = (targets == cls_a) | (targets == cls_b)
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    subset = Subset(cifar, indices)

    class RelabeledSubset(torch.utils.data.Dataset):
        def __init__(self, subset, cls_a):
            self.subset = subset
            self.cls_a = cls_a

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, int(y != self.cls_a)  # cls_a → 0, cls_b → 1

    return RelabeledSubset(subset, cls_a)


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_task(model, dataset, task_id, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, task_id).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def run(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_cifar = torchvision.datasets.CIFAR10(
        "./data", train=True,  download=True, transform=train_transform)
    test_cifar  = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_transform)

    train_tasks = [get_task_dataset(train_cifar, t) for t in range(5)]
    test_tasks  = [get_task_dataset(test_cifar,  t) for t in range(5)]

    model = SplitCIFARModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Config: optimizer={args.optimizer}  lr={args.lr}  "
          f"gate_scale={args.gate_scale}  gate_c={args.gate_c}")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == "workadam":
        optimizer = WorkAdam(model.parameters(), lr=args.lr,
                             beta1=0.9, rho=0.99,
                             weight_decay=args.weight_decay, mass_init=1e-3)
    elif args.optimizer == "gatedthermo":
        optimizer = GatedThermoAdamW(model.parameters(), lr=args.lr,
                                      gate_scale=args.gate_scale, c=args.gate_c,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == "snr":
        optimizer = SNRAdam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    elif args.optimizer == "adabelief":
        optimizer = AdaBelief(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"./out/split_cifar/{args.optimizer}/{timestamp}"
    os.makedirs(work_dir, exist_ok=True)

    fieldnames = (["global_iter", "task", "train_loss"]
                  + [f"test_acc_task{t}" for t in range(5)]
                  + ["avg_acc"])
    log_path = os.path.join(work_dir, "metrics.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    global_iter = 0

    for task_idx in range(5):
        cls_a, cls_b = TASKS[task_idx]
        print(f"\n{'='*55}")
        print(f"Task {task_idx}: {TASK_NAMES[task_idx]}  ({args.iters_per_task} iters)")
        print(f"{'='*55}")

        train_ds = train_tasks[task_idx]
        loader = DataLoader(
            train_ds,
            sampler=torch.utils.data.RandomSampler(
                train_ds, replacement=True, num_samples=int(1e10)),
            batch_size=args.batch_size,
            num_workers=0,
        )
        data_iter = iter(loader)

        model.train()
        for _ in range(args.iters_per_task):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            logits = model(x, task_idx)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if isinstance(optimizer, GatedThermoAdamW):
                optimizer.step(loss=loss.item())
            else:
                optimizer.step()

            if global_iter % args.eval_every == 0:
                accs = [
                    eval_task(model, test_tasks[t], t, device)
                    if t <= task_idx else 0.0
                    for t in range(5)
                ]
                avg_acc = sum(accs[:task_idx + 1]) / (task_idx + 1)

                row = {"global_iter": global_iter, "task": task_idx,
                       "train_loss": loss.item(), "avg_acc": avg_acc}
                for t in range(5):
                    row[f"test_acc_task{t}"] = accs[t]
                writer.writerow(row)
                log_file.flush()

                acc_str = "  ".join(
                    f"T{t}:{accs[t]*100:.1f}%" for t in range(task_idx + 1))
                gate_str = ""
                if isinstance(optimizer, GatedThermoAdamW):
                    gs = optimizer.get_gate_status()
                    gate_str = f"  f={gs['f']:.2f} z={gs['z']:.2f}"
                print(f"  [{global_iter:6d}] {acc_str}  avg={avg_acc*100:.1f}%{gate_str}")

            global_iter += 1

    log_file.close()

    print(f"\n{'='*55}")
    print("Final test accuracies (task-incremental):")
    accs = [eval_task(model, test_tasks[t], t, device) for t in range(5)]
    for t in range(5):
        print(f"  Task {t} ({TASK_NAMES[t]}): {accs[t]*100:.1f}%")
    print(f"  Average: {sum(accs)/5*100:.1f}%")
    print(f"\nMetrics saved to {log_path}")

    plot_results(log_path, work_dir, args.optimizer)


def plot_results(log_path, work_dir, optimizer_name):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(log_path)
    boundaries = df.groupby("task")["global_iter"].min().tolist()[1:]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Split CIFAR-10 (task-incremental) — {optimizer_name}",
                 fontweight="bold")

    ax = axes[0]
    for t in range(5):
        mask = df[f"test_acc_task{t}"] > 0
        ax.plot(df.loc[mask, "global_iter"],
                df.loc[mask, f"test_acc_task{t}"] * 100,
                color=colors[t], label=TASK_NAMES[t], alpha=0.8)
    for b in boundaries:
        ax.axvline(b, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-task Accuracy"); ax.set_ylim(-5, 105)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(df["global_iter"], df["avg_acc"] * 100, color="black", lw=2)
    for b in boundaries:
        ax.axvline(b, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Average Accuracy (tasks seen so far)"); ax.set_ylim(-5, 105)

    plt.tight_layout()
    out = os.path.join(work_dir, "results.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="adamw",
                        choices=["adamw", "workadam", "gatedthermo", "snr", "adabelief"])
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--iters_per_task", type=int,   default=2000)
    parser.add_argument("--eval_every",     type=int,   default=200)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--gate_c",         type=float, default=0.3)
    parser.add_argument("--gate_scale",     type=float, default=200.0)
    args = parser.parse_args()

    run(args)
