# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


__all__ = ["WorkAdam"]


class WorkAdam(Optimizer):
    r"""
    Work-based Adam-style optimizer with *local* information mass.

    Per-parameter coordinate :math:`i`:

    .. math::
        g_t        &= \nabla_{\theta} f_t(\theta_{t-1}) \\
        m_t        &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
        \Delta\theta_{t,i} &= \theta_{t,i} - \theta_{t-1,i} \\
        W_{t,i}    &= \big| g_{t,i}\, \Delta\theta_{t,i} \big| \\
        M_{t,i}    &= \rho M_{t-1,i} + (1 - \rho)\, W_{t,i} \\
        \hat m_t   &= m_t / (1 - \beta_1^t) \\
        \theta_t   &= \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{M_t} + \varepsilon}

    Here :math:`M_t` is a smoothed *recent work* per coordinate. Spikes in
    :math:`M_t` or its derivatives are candidates for task-switching signals.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: learning rate (default: 1e-3)
        beta1: coefficient for the first-moment EMA of gradients (default: 0.9)
        rho: coefficient for the EMA of work (default: 0.99)
        eps: term added to denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: weight decay factor (default: 0)
        decoupled_weight_decay: if True, use AdamW-style decoupled decay;
            otherwise, L2 penalty is folded into the gradient. (default: True)
        maximize: maximize the objective instead of minimizing (default: False)
        mass_init: initial mass per coordinate (default: 1e-3). Sets the
            initial scale of the denominator via :math:`\sqrt{M_0}`.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta1: float = 0.9,
        rho: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,  # keyword-only arguments
        decoupled_weight_decay: bool = True,
        maximize: bool = False,
        mass_init: float = 1e-3,
    ) -> None:
        # Basic sanity checks (mirroring torch.optim.Adam style)
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
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
            mass_init=mass_init,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float = group["beta1"]
            rho: float = group["rho"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            decoupled_wd: bool = group["decoupled_weight_decay"]
            maximize: bool = group["maximize"]
            mass_init: float = group["mass_init"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("WorkAdam does not support sparse gradients")

                if maximize:
                    grad = -grad

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment of gradients
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # EMA of work: start small but nonzero
                    if mass_init == 0.0:
                        state["mass"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["mass"] = torch.full_like(
                            p, fill_value=mass_init, memory_format=torch.preserve_format
                        )
                    # Previous parameter values for displacement
                    state["prev_param"] = p.detach().clone(
                        memory_format=torch.preserve_format
                    )

                exp_avg: Tensor = state["exp_avg"]
                mass: Tensor = state["mass"]
                prev_param: Tensor = state["prev_param"]

                state["step"] += 1
                step: int = state["step"]

                # Weight decay
                if weight_decay != 0.0:
                    if decoupled_wd:
                        # AdamW-style: separate shrink
                        p.data.add_(p.data, alpha=-lr * weight_decay)
                    else:
                        # L2 penalty folded into gradient
                        grad = grad.add(p.data, alpha=weight_decay)

                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Bias correction for m_t
                bias_correction1 = 1.0 - beta1**step
                exp_avg_hat = exp_avg / bias_correction1

                # Denominator uses *current* EMA of work (mass)
                denom = mass.sqrt().add_(eps)

                # Parameter update
                p.data.addcdiv_(exp_avg_hat, denom, value=-lr)

                # Δθ_t = θ_t - θ_{t-1}
                displacement = p.data - prev_param
                work = (grad * displacement).abs()

                # M_t = ρ M_{t-1} + (1 - ρ) W_t
                mass.mul_(rho).add_(work, alpha=1.0 - rho)

                # Store current params as previous for next step
                prev_param.copy_(p.data)

        return loss
