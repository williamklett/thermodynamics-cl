# mypy: allow-untyped-defs
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


__all__ = ["GatedThermoAdamW"]


class GatedThermoAdamW(Optimizer):
    r"""
    WorkAdam + ReLU gate on the denominator.

    Identical to WorkAdam except the denominator is:
        sqrt(M_{t-1} * (1 + f_t) + eps)
    instead of:
        sqrt(M_{t-1} + eps)

    where f_t = max(0, z_t - c) is a global scalar derived from
    the EMA-smoothed loss derivative z-scored against its Welford history.

    When f_t = 0 (stable training, no loss spike), the update is
    *exactly* WorkAdam.  When f_t > 0 (loss is rising anomalously),
    the accumulated mass is amplified in the denominator, slowing
    learning on high-work coordinates.

    Args:
        params:      model parameters
        lr:          learning rate (default: 1e-3)
        beta1:       first-moment EMA decay (default: 0.9)
        rho:         work EMA decay (default: 0.99)
        gamma:       loss-EMA decay for derivative smoothing (default: 0.99)
        c:           ReLU shift in z-score units (default: 1.0)
        eps:         numerical stability floor (default: 1e-8)
        weight_decay: coefficient (default: 0.0)
        decoupled_wd: True → AdamW-style after work (default: True)
        maximize:    maximize objective (default: False)
        mass_init:   initial mass per coordinate (default: 1e-3)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta1: float = 0.3,
        rho: float = 0.99,
        gamma: float = 0.99,
        c: float = 0.3,
        gate_scale: float = 200.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        decoupled_wd: bool = True,
        maximize: bool = False,
        mass_init: float = 1e-3,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if mass_init < 0.0:
            raise ValueError(f"Invalid mass_init value: {mass_init}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_wd=decoupled_wd,
            maximize=maximize,
            mass_init=mass_init,
        )
        super().__init__(params, defaults)

        self.gamma = gamma
        self.c = c
        self.gate_scale = gate_scale

        # Global loss-tracking state
        self._loss_slow: Optional[float] = None   # long-term baseline, gamma=0.99

        # Welford online stats on (loss - loss_slow)
        self._welford_n: int = 0
        self._welford_mean: float = 0.0
        self._welford_M2: float = 1.0

        # Single slow-decay EMA for gate
        self._f_ema: float = 0.0

        # For logging
        self._f: float = 0.0
        self._z: float = 0.0

    @torch.no_grad()
    def step(self, loss: float = None, closure=None):
        """Performs a single optimization step."""
        if closure is not None and loss is None:
            with torch.enable_grad():
                loss = closure()
        if loss is not None and hasattr(loss, 'item'):
            loss = loss.item()

        # ── Gate computation ─────────────────────────────────────────────
        if loss is not None:
            if self._loss_slow is None:
                self._loss_slow = loss

            # Slow baseline tracks the long-term loss floor
            gamma_slow = 0.99
            residual = loss - self._loss_slow
            self._loss_slow = gamma_slow * self._loss_slow + (1.0 - gamma_slow) * loss

            # Welford on positive residuals only — keeps mean/sigma calibrated
            # to normal upward fluctuations, not poisoned by learning descent
            if residual > 0:
                self._welford_n += 1
                delta = residual - self._welford_mean
                self._welford_mean += delta / self._welford_n
                delta2 = residual - self._welford_mean
                self._welford_M2 += delta * delta2

        sigma = math.sqrt(self._welford_M2 / max(self._welford_n, 1) + 1e-8)
        residual_now = (0.0 if self._loss_slow is None else
                        (loss if loss is not None else self._loss_slow) - self._loss_slow)
        z_t = (residual_now - self._welford_mean) / sigma
        f_current = max(0.0, z_t - self.c)
        self._f_ema = 0.99 * self._f_ema + 0.01 * f_current
        f_t = self._f_ema * self.gate_scale
        
        self._f = f_t
        self._z = z_t

    

        # ── Per-parameter update (WorkAdam + gate) ───────────────────────
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float = group["beta1"]
            rho: float = group["rho"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            decoupled_wd: bool = group["decoupled_wd"]
            maximize: bool = group["maximize"]
            mass_init: float = group["mass_init"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GatedThermoAdamW does not support sparse gradients")

                if maximize:
                    grad = -grad

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if mass_init == 0.0:
                        state["mass"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["mass"] = torch.full_like(
                            p, fill_value=mass_init, memory_format=torch.preserve_format
                        )
                    state["prev_param"] = p.detach().clone(
                        memory_format=torch.preserve_format
                    )

                exp_avg: Tensor = state["exp_avg"]
                mass: Tensor = state["mass"]
                prev_param: Tensor = state["prev_param"]

                state["step"] += 1
                step: int = state["step"]

                # Weight decay (coupled)
                if weight_decay != 0.0 and not decoupled_wd:
                    grad = grad.add(p.data, alpha=weight_decay)

                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Bias correction
                bias_correction1 = 1.0 - beta1**step
                exp_avg_hat = exp_avg / bias_correction1

                # Gated denominator: falls back to WorkAdam when f_t = 0
                denom = (mass * (1.0 + f_t) + eps).sqrt()

                # Parameter update
                p.data.addcdiv_(exp_avg_hat, denom, value=-lr)

                # Work = |g * Δθ|
                update = -lr * exp_avg_hat / denom
                work = (grad * update).abs()

                # Mass EMA
                mass.mul_(rho).add_(work, alpha=1.0 - rho)

                # Decoupled weight decay after work measurement
                if weight_decay != 0.0 and decoupled_wd:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                prev_param.copy_(p.data)

        return loss

    def get_gate_status(self) -> dict:
        return {
            "f": self._f,
            "z": self._z,
            "f_ema": self._f_ema,
            "loss_slow": self._loss_slow,
            "welford_mean": self._welford_mean,
            "welford_M2": self._welford_M2,
            "welford_n": self._welford_n,
        }
