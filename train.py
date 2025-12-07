"""
Trains a GPT to add n-digit numbers. Based on: https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py
"""

import csv
import os
import random
from datetime import datetime
from omegaconf import OmegaConf

import torch
from torch.utils.data.dataloader import DataLoader

from gpt import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from adder.dataset import ArithmeticDataset


def setup_experiment(config_path: str = "arithmetic/configs/default.yaml"):
    # Load config
    config = OmegaConf.load(config_path)

    # System
    random.seed(config.system.seed)
    torch.manual_seed(config.system.seed)
    torch.cuda.manual_seed_all(config.system.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.system.work_dir = os.path.join(config.system.work_dir, timestamp)
    os.makedirs(config.system.work_dir, exist_ok=True)

    # Data
    config.data.ndigit = config.data.ndigit
    train_dataset = ArithmeticDataset(config.data, split="train")
    test_dataset = ArithmeticDataset(config.data, split="test")

    # Model - update config with dataset-derived values
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(GPTConfig(**config.model))

    # Trainer
    trainer = Trainer(
        TrainerConfig(**config.trainer), model, train_dataset, test_dataset
    )

    # Save updated config
    OmegaConf.save(config, os.path.join(config.system.work_dir, "config.yaml"))

    return config, trainer


# Define eval functions for Trainer callback
def eval_loss(trainer):
    """Compute average loss on test set."""
    loader = DataLoader(
        trainer.test_dataset, batch_size=100, num_workers=0, drop_last=False
    )
    losses = []
    for x, y in loader:
        x, y = x.to(trainer.device), y.to(trainer.device)
        _, loss = trainer.model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def eval_split(trainer, ndigit, split, max_batches=None):
    dataset = {"train": trainer.train_dataset, "test": trainer.test_dataset}[split]
    results = []
    mistakes = []
    mistakes_printed_already = 0
    factors = torch.tensor([[10**i for i in range(ndigit + 1)][::-1]]).to(
        trainer.device
    )
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        # isolate the first two digits of the input sequence alone
        d1d2 = x[:, : ndigit * 2]
        # let the model sample the rest of the sequence
        d1d2d3 = trainer.model.generate(
            d1d2, ndigit + 1, sample=False
        )  # using greedy argmax, not sampling
        d3 = d1d2d3[:, -(ndigit + 1) :]
        d3 = d3.flip(1)  # reverse the digits to their "normal" order
        # decode the integers from individual digits
        d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
        d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i # manually calculate the ground truth
        # evaluate the correctness of the results in this batch
        correct = (d3i_pred == d3i_gt).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i]:
                mistake = f"GPT claims that {d1i[i]} + {d2i[i]} = {d3i_pred[i]} but gt is {d3i_gt[i]}"
                mistakes.append(mistake)
                if mistakes_printed_already < 5:
                    mistakes_printed_already += 1
                    print(mistake)
        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print(
        "%s final score: %d/%d = %.2f%% correct"
        % (split, rt.sum(), len(results), 100 * rt.mean())
    )
    return rt.mean(), mistakes


def make_batch_callback(config):
    top_score = 0
    ndigit = config.data.ndigit
    train_max_batches = {1: None, 2: None, 3: 5}.get(ndigit, 5)

    # Accumulate train losses between evals
    train_losses = []

    # Initialize CSV log
    log_path = os.path.join(config.system.work_dir, "metrics.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(
        log_file, fieldnames=["iter", "train_loss", "val_loss", "train_acc", "test_acc"]
    )
    log_writer.writeheader()

    def batch_end_callback(trainer):
        nonlocal top_score, train_losses
        train_losses.append(trainer.loss.item())

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if trainer.iter_num % 500 == 0:
            # Average train loss over the interval
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_losses = []

            trainer.model.eval()
            with torch.no_grad():
                val_loss = eval_loss(trainer)
                print(f"train loss: {avg_train_loss:.5f}, val loss: {val_loss:.5f}")
                train_score, train_mistakes = eval_split(
                    trainer,
                    ndigit,
                    "train",
                    max_batches=train_max_batches,
                )
                test_score, test_mistakes = eval_split(
                    trainer, ndigit, "test", max_batches=None
                )

            # Log metrics
            log_writer.writerow(
                {
                    "iter": trainer.iter_num,
                    "train_loss": avg_train_loss,  # average training loss since last eval
                    "val_loss": val_loss,  # average loss on the test set
                    "train_acc": train_score.item(),  # accuracy on (first max_batches of) the training set
                    "test_acc": test_score.item(),  # accuracy on the test set
                }
            )
            log_file.flush()

            # Log mistakes
            mistakes_dir = os.path.join(config.system.work_dir, "mistakes")
            os.makedirs(mistakes_dir, exist_ok=True)
            mistakes_path = os.path.join(
                mistakes_dir, f"iter_{trainer.iter_num:06d}.txt"
            )
            with open(mistakes_path, "w", encoding="utf-8") as f:
                f.write(f"Train mistakes ({len(train_mistakes)}):\n")
                for m in train_mistakes:
                    f.write(f"  {m}\n")
                f.write(f"\nTest mistakes ({len(test_mistakes)}):\n")
                for m in test_mistakes:
                    f.write(f"  {m}\n")

            score = train_score + test_score
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(trainer.model.state_dict(), ckpt_path)
            trainer.model.train()

    return batch_end_callback


def plot_metrics(work_dir: str):
    """Plot and save training metrics."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(work_dir, "metrics.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(df["iter"], df["train_loss"], label="train")
    axes[0].plot(df["iter"], df["val_loss"], label="val")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Accuracy plot
    axes[1].plot(df["iter"], df["train_acc"], label="train")
    axes[1].plot(df["iter"], df["test_acc"], label="test")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "metrics.png"), dpi=150)
    plt.close()
    print(f"Saved metrics plot to {os.path.join(work_dir, 'metrics.png')}")


def run_experiment(config_path: str = "arithmetic/configs/default.yaml"):
    config, trainer = setup_experiment(config_path)
    trainer.set_callback("on_batch_end", make_batch_callback(config))
    trainer.run()
    plot_metrics(config.system.work_dir)


if __name__ == "__main__":
    run_experiment()
