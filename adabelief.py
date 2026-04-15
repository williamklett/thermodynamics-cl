"""
AdaBelief Optimizer — Zhuang et al., NeurIPS 2020
"AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"

Key idea: replace Adam's second moment EMA(g²) with EMA of squared gradient
*residuals* (g - m)², where m is the running first moment. This measures how
much the current gradient deviates from the optimizer's own prediction (the
momentum). Large step when gradient aligns with belief; small step when it
disagrees.

Update rule:
    m_t   = β1*m_{t-1} + (1-β1)*g_t
    s_t   = β2*s_{t-1} + (1-β2)*(g_t - m_t)² + ε   # ε keeps s_t > 0
    m̂_t   = m_t / (1 - β1^t)
    ŝ_t   = s_t / (1 - β2^t)
    θ_t   = θ_{t-1} - α * m̂_t / (√ŝ_t + ε)
    θ_t  *= (1 - α * weight_decay)                   # decoupled WD

The +ε inside the s_t update (not just in the denominator) is the official
implementation's trick to ensure s_t ≥ ε even when gradients are constant,
preventing the denominator from collapsing to ε and causing large steps.
"""

import math
import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            lr:           Step size α.
            beta1:        EMA decay for first moment (momentum). Default 0.9.
            beta2:        EMA decay for second moment (belief variance). Default 0.999.
            eps:          Added both inside the s_t update and to denominator.
                          Use 1e-8 for vision; 1e-16 for transformers/LSTMs.
            weight_decay: Decoupled weight decay coefficient (AdamW-style).
        """
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr     = group["lr"]
            beta1  = group["beta1"]
            beta2  = group["beta2"]
            eps    = group["eps"]
            wd     = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]

                # Initialise state
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)   # first moment
                    state["s"] = torch.full_like(p, eps)  # second moment (belief variance), init to eps

                state["step"] += 1
                t = state["step"]
                m, s = state["m"], state["s"]

                # First moment: m_t = β1*m + (1-β1)*g
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)

                # Belief variance: s_t = β2*s + (1-β2)*(g - m_t)² + ε
                residual = g - m
                s.mul_(beta2).addcmul_(residual, residual, value=1.0 - beta2).add_(eps)

                # Bias-corrected estimates
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                s_hat = s / bc2

                # Parameter update: θ -= α * m̂ / (√ŝ + ε)
                denom = s_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

        return loss
