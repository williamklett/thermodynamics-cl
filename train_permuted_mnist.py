"""
Permuted MNIST continual learning benchmark.

10 sequential tasks, each applying a different fixed random permutation to
the 784 input pixels. All tasks share the same 10-class output head (no
task labels at test time). The model must retain all previous permutations
while learning new ones.

This is a harder benchmark than Split MNIST because:
- Shared head across all tasks (no task-specific escape hatch)
- 10 tasks instead of 5
- Full 10-class classification instead of binary

Standard result: AdamW collapses to ~19% by task 10 (random for 10 classes ≈ 10%).
EWC holds ~90%+ across all tasks.
"""

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

from thermo_adam import WorkAdam
from gated_thermo_adamw import GatedThermoAdamW
from snr_adam import SNRAdam
from adabelief import AdaBelief


# ── Model ─────────────────────────────────────────────────────────────────────

class PermutedMNISTModel(nn.Module):
    """Single shared MLP + single shared 10-class head. No task-specific anything."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ── Dataset ───────────────────────────────────────────────────────────────────

class PermutedMNISTDataset(Dataset):
    """Applies a fixed pixel permutation to MNIST images."""
    def __init__(self, mnist, permutation):
        self.mnist = mnist
        self.perm = permutation  # LongTensor of shape (784,)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        x, y = self.mnist[idx]
        x = x.view(-1)[self.perm]  # permute pixels
        return x, y


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_task(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def run(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_mnist = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_mnist  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Generate fixed permutations — task 0 is identity (no permutation)
    rng = torch.Generator()
    rng.manual_seed(args.seed)
    permutations = [torch.arange(784)]  # task 0: unpermuted
    for _ in range(args.n_tasks - 1):
        permutations.append(torch.randperm(784, generator=rng))

    train_tasks = [PermutedMNISTDataset(train_mnist, p) for p in permutations]
    test_tasks  = [PermutedMNISTDataset(test_mnist,  p) for p in permutations]

    model = PermutedMNISTModel(hidden_dim=args.hidden_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: optimizer={args.optimizer}  lr={args.lr}  "
          f"gate_scale={args.gate_scale}  gate_c={args.gate_c}  tasks={args.n_tasks}")

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
    work_dir = f"./out/permuted_mnist/{args.optimizer}/{timestamp}"
    os.makedirs(work_dir, exist_ok=True)

    fieldnames = (["global_iter", "task", "train_loss"]
                  + [f"test_acc_task{t}" for t in range(args.n_tasks)]
                  + ["avg_acc_seen"])
    log_path = os.path.join(work_dir, "metrics.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    global_iter = 0

    for task_idx in range(args.n_tasks):
        print(f"\n{'='*55}")
        print(f"Task {task_idx}  ({args.iters_per_task} iters)")
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

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if isinstance(optimizer, (GatedThermoAdamW, SNRAdam)):
                optimizer.step(loss=loss.item())
            else:
                optimizer.step()

            if global_iter % args.eval_every == 0:
                # Evaluate all tasks seen so far
                accs = []
                for t in range(args.n_tasks):
                    acc = eval_task(model, test_tasks[t], device) if t <= task_idx else 0.0
                    accs.append(acc)

                avg_acc = sum(accs[:task_idx + 1]) / (task_idx + 1)
                row = {"global_iter": global_iter, "task": task_idx,
                       "train_loss": loss.item(), "avg_acc_seen": avg_acc}
                for t in range(args.n_tasks):
                    row[f"test_acc_task{t}"] = accs[t]
                writer.writerow(row)
                log_file.flush()

                acc_str = "  ".join(f"T{t}:{accs[t]*100:.1f}%" for t in range(task_idx + 1))
                gate_str = ""
                if isinstance(optimizer, GatedThermoAdamW):
                    gs = optimizer.get_gate_status()
                    gate_str = f"  f={gs['f']:.2f} z={gs['z']:.2f}"
                print(f"  [{global_iter:6d}] {acc_str}  avg={avg_acc*100:.1f}%{gate_str}")

            global_iter += 1

    log_file.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Final test accuracies (all tasks):")
    accs = [eval_task(model, test_tasks[t], device) for t in range(args.n_tasks)]
    for t in range(args.n_tasks):
        print(f"  Task {t:2d}: {accs[t]*100:.1f}%")
    print(f"  Average: {sum(accs)/args.n_tasks*100:.1f}%")
    print(f"\nMetrics saved to {log_path}")

    plot_results(log_path, work_dir, args.optimizer, args.n_tasks)


def plot_results(log_path, work_dir, optimizer_name, n_tasks):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Permuted MNIST — {optimizer_name}", fontweight="bold")

    # Task boundaries
    boundaries = df.groupby("task")["global_iter"].min().tolist()[1:]

    ax = axes[0]
    colors = plt.cm.tab10.colors
    for t in range(n_tasks):
        col = f"test_acc_task{t}"
        mask = df[col] > 0
        ax.plot(df.loc[mask, "global_iter"], df.loc[mask, col] * 100,
                color=colors[t % 10], alpha=0.7, label=f"T{t}")
    for b in boundaries:
        ax.axvline(b, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-task Accuracy"); ax.set_ylim(-5, 105)
    if n_tasks <= 10:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax.plot(df["global_iter"], df["avg_acc_seen"] * 100, color="black", lw=2)
    for b in boundaries:
        ax.axvline(b, color="gray", ls="--", alpha=0.4)
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
    parser.add_argument("--n_tasks",      type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim",   type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--iters_per_task", type=int, default=2000)
    parser.add_argument("--eval_every",   type=int,   default=200)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--gate_c",       type=float, default=0.3)
    parser.add_argument("--gate_scale",   type=float, default=200.0)
    args = parser.parse_args()

    run(args)
