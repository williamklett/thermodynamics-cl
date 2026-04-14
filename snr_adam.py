# mypy: allow-untyped-defs
from __future__ import annotations

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


__all__ = ["SNRAdam"]


class SNRAdam(Optimizer):
    r"""
    Signal-to-Noise Ratio optimizer for continual learning.

    Replaces Adam's second moment EMA(g²) with true gradient variance
    Var(g) = EMA(g²) - EMA(g)², so the denominator measures gradient
    *disagreement* (noise) rather than gradient magnitude (signal + noise).

    Update rule:
        m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
        s_t   = beta2 * s_{t-1} + (1 - beta2) * g_t²
        m̂_t   = m_t / (1 - beta1^t)
        ŝ_t   = s_t / (1 - beta2^t)
        Var_t = max(ŝ_t - m̂_t², eps)
        θ_t   = θ_{t-1} - lr * m̂_t / sqrt(Var_t)
        θ_t   = θ_t * (1 - lr * weight_decay)   # decoupled WD

    CL behaviour:
        within-task stable  → Var small → large steps (fast learning / grokking)
        task switch         → conflicting gradients → Var spikes → steps shrink
        shared basin        → Var stays low across tasks → large steps maintained
        conflicting tasks   → EMA(g) cancels, Var stays high → near-zero steps

    Args:
        params:       model parameters
        lr:           learning rate (default: 3e-4)
        beta1:        first-moment EMA decay (default: 0.9)
        beta2:        second-moment EMA decay (default: 0.99).
                      Lower → faster variance adaptation → better grokking detection.
                      Higher → longer conflict memory → more CL protection.
        eps:          variance floor (default: 1e-6)
        weight_decay: decoupled weight decay (default: 0.01)
        use_sqrt:     divide by sqrt(Var) if True, raw Var if False (default: True)
        maximize:     maximize objective instead of minimizing (default: False)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = .999,
        eps: float = 1e-4,
        weight_decay: float = 0.01,
        *,
        use_sqrt: bool = True,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            use_sqrt=use_sqrt,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float = group["beta1"]
            beta2: float = group["beta2"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            use_sqrt: bool = group["use_sqrt"]
            maximize: bool = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g: Tensor = p.grad
                if g.is_sparse:
                    raise RuntimeError("SNRAdam does not support sparse gradients")

                if maximize:
                    g = -g

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["s"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m: Tensor = state["m"]
                s: Tensor = state["s"]
                state["step"] += 1
                t: int = state["step"]

                # Step 1: Update moments
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                s.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Step 2: Bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                s_hat = s / bc2

                # Step 3: True variance
                var = (s_hat - m_hat.square()).clamp_(min=eps)

                # Step 4: SNR update
                denom = var.sqrt() if use_sqrt else var
                p.addcdiv_(m_hat, denom, value=-lr)

                # Step 5: Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

        return loss
