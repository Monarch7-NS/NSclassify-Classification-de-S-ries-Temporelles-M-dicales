"""Fonctions de coût pour la classification binaire. Toutes prennent (logits [B], cibles {0,1} [B])."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL = -alpha_t (1 - p_t)^gamma log p_t (Lin et al., 2017). gamma = 0 et alpha = None redonne la BCE."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        loss = (1 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            loss = (self.alpha * y + (1 - self.alpha) * (1 - y)) * loss
        return loss.mean()


class SmoothedBCE(nn.Module):
    """BCE avec cibles lissées : y' = y (1 - eps) + eps / 2."""

    def __init__(self, eps=0.1, pos_weight=None):
        super().__init__()
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, y):
        y_s = y * (1 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(logits, y_s, pos_weight=self.pos_weight)


class SigmoidMSE(nn.Module):
    """Ablation pédagogique : MSE sur sigmoid(logit). Gradient atténué par sigma'(z) quand le modèle est confiant et faux."""

    def forward(self, logits, y):
        return F.mse_loss(torch.sigmoid(logits), y)


def positive_weight(y_train):
    n_pos = float((y_train == 1).sum())
    return (len(y_train) - n_pos) / max(n_pos, 1.0)


def build_loss(spec, y_train):
    name = spec["name"]
    pw = torch.tensor(positive_weight(y_train), dtype=torch.float32)
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_pw":
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if name == "focal":
        return FocalLoss(gamma=spec.get("gamma", 2.0), alpha=spec.get("alpha"))
    if name == "bce_ls":
        return SmoothedBCE(spec.get("eps", 0.1), pos_weight=pw if spec.get("weighted") else None)
    if name == "mse_unsuitable":
        return SigmoidMSE()
    raise ValueError(f"loss inconnue: {name}")
