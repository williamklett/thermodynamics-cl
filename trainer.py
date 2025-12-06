"""
Simple trainer class. Based on: https://github.com/karpathy/nanoGPT/blob/master/trainer.py
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data.dataloader import DataLoader
from thermo_adam import WorkAdam

OPT_FUNC = {
    "adamw": torch.optim.AdamW,
    "workadam": WorkAdam,
}

@dataclass
class TrainerConfig:
    device: str = "auto"
    num_workers: int = 4
    max_iters: Optional[int] = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0
    optimizer: [Dict[str, Any]] = None  # expects e.g. {"name": "adamw", "lr": ..., ...}


class Trainer:
    def __init__(self, config: TrainerConfig, model, train_dataset, test_dataset):
        self.config = config
        self.model = model
        # Extract optimizer class + kwargs
        opt_cfg = dict(config.optimizer)  # shallow copy (handles OmegaConf.DictConfig too)
        opt_name = opt_cfg.pop("name")   # remove 'name' so it doesn't get passed as kwarg
        if opt_name not in OPT_FUNC:
            raise ValueError(f"Unknown optimizer '{opt_name}'. Expected one of {list(OPT_FUNC.keys())}.")
        self.optimizer_cls = OPT_FUNC[opt_name]
        self.optimizer_kwargs = opt_cfg

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Running on device", self.device)

        # Variables for logging
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        # Initialize the dataloader (infinite random stream)
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True, # faster transfers to GPU
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        self.data_iter = iter(self.train_loader)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
    def get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        batch = [t.to(self.device) for t in batch]
        x, y = batch
        return x, y

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(self.optimizer_cls, self.optimizer_kwargs)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        while True:
            # Fetch the next batch (x, y)
            x, y = self.get_batch()

            # Forward the model
            logits, self.loss = model(x, y)

            # Backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break