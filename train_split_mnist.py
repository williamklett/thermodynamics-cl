"""
Split MNIST continual learning benchmark — task-incremental setting.

5 sequential binary classification tasks:
  Task 0: digits 0 vs 1
  Task 1: digits 2 vs 3
  Task 2: digits 4 vs 5
  Task 3: digits 6 vs 7
  Task 4: digits 8 vs 9

Model: shared MLP backbone (784 → 256 → 256) + 5 task-specific heads (256 → 2).
At test time the correct head is selected (standard task-incremental protocol).
This matches EWC / SI / PackNet papers — baseline AdamW should reach ~95%+,
catastrophic forgetting drops it to ~19% without protection.
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

class MLPBackbone(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class SplitMNISTModel(nn.Module):
    def __init__(self, hidden_dim=256, n_tasks=5):
        super().__init__()
        self.backbone = MLPBackbone(hidden_dim=hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_tasks)])

    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.heads[task_id](features)


# ── Dataset helpers ───────────────────────────────────────────────────────────

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def get_task_dataset(mnist, task_idx):
    cls_a, cls_b = TASKS[task_idx]
    targets = mnist.targets
    mask = (targets == cls_a) | (targets == cls_b)
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    subset = Subset(mnist, indices)

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
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, task_id).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def run(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_mnist  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_tasks = [get_task_dataset(train_mnist, t) for t in range(5)]
    test_tasks  = [get_task_dataset(test_mnist,  t) for t in range(5)]

    model = SplitMNISTModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: optimizer={args.optimizer} lr={args.lr} gate_scale={args.gate_scale} gate_c={args.gate_c}")

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
    work_dir = f"./out/split_mnist/{args.optimizer}/{timestamp}"
    os.makedirs(work_dir, exist_ok=True)

    fieldnames = ["global_iter", "task", "train_loss"] + [f"test_acc_task{t}" for t in range(5)] + ["avg_acc"]
    log_path = os.path.join(work_dir, "metrics.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    global_iter = 0

    for task_idx in range(5):
        cls_a, cls_b = TASKS[task_idx]
        print(f"\n{'='*50}")
        print(f"Task {task_idx}: {cls_a} vs {cls_b}  ({args.iters_per_task} iters)")
        print(f"{'='*50}")

        train_ds = train_tasks[task_idx]
        loader = DataLoader(
            train_ds,
            sampler=torch.utils.data.RandomSampler(
                train_ds, replacement=True, num_samples=int(1e10)
            ),
            batch_size=args.batch_size,
            num_workers=0,
        )
        data_iter = iter(loader)

        model.train()
        for step in range(args.iters_per_task):
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

            if global_iter % 50 == 0:
                print(f"  iter {global_iter:5d}: loss {loss.item():.4f}", end="")
                if isinstance(optimizer, GatedThermoAdamW):
                    gs = optimizer.get_gate_status()
                    print(f"  f={gs['f']:.3f}  z={gs['z']:.2f}", end="")
                print()

            if global_iter % args.eval_every == 0:
                accs = [eval_task(model, test_tasks[t], t, device) for t in range(5)]
                avg_acc = sum(accs[:task_idx + 1]) / (task_idx + 1)
                row = {"global_iter": global_iter, "task": task_idx,
                       "train_loss": loss.item(), "avg_acc": avg_acc}
                for t in range(5):
                    row[f"test_acc_task{t}"] = accs[t]
                writer.writerow(row)
                log_file.flush()

                acc_str = "  ".join(f"T{t}:{accs[t]*100:.1f}%" for t in range(task_idx + 1))
                print(f"  [{global_iter:5d}] {acc_str}  | avg={avg_acc*100:.1f}%")

            global_iter += 1

    log_file.close()

    print(f"\n{'='*50}")
    print("Final test accuracies (task-incremental, correct head per task):")
    accs = [eval_task(model, test_tasks[t], t, device) for t in range(5)]
    for t, (cls_a, cls_b) in enumerate(TASKS):
        print(f"  Task {t} ({cls_a} vs {cls_b}): {accs[t]*100:.1f}%")
    print(f"  Average: {sum(accs)/5*100:.1f}%")
    print(f"\nMetrics saved to {log_path}")

    plot_results(log_path, work_dir, args.optimizer)
    return work_dir


def plot_results(log_path, work_dir, optimizer_name):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(log_path)
    boundaries = []
    for t in range(1, 5):
        switch = df[df["task"] == t]["global_iter"].min()
        if not pd.isna(switch):
            boundaries.append(int(switch))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Split MNIST (task-incremental) — {optimizer_name}", fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax = axes[0]
    for t in range(5):
        ax.plot(df["global_iter"], df[f"test_acc_task{t}"] * 100,
                color=colors[t], label=f"Task {t} ({TASKS[t][0]}v{TASKS[t][1]})", alpha=0.8)
    for b in boundaries:
        ax.axvline(b, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-task Accuracy"); ax.set_ylim(-5, 105); ax.legend(fontsize=8)

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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--iters_per_task", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate_c", type=float, default=0.3)
    parser.add_argument("--gate_scale", type=float, default=200.0)
    args = parser.parse_args()

    run(args)
