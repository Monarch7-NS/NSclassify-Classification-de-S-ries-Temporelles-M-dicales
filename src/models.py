"""Modèles : baselines sklearn, MLP, CNN 1D, CNN-LSTM (+ presets nommés) et CNN compact historique."""
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic(C=1.0):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced", solver="liblinear", max_iter=2000
        )),
    ])


def build_random_forest(max_depth=16, min_samples_leaf=2, seed=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


class CompactCNN1D(nn.Module):
    """Entrée [B, 1, L] -> logit d'anormalité [B]."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


# ---------------------------------------------------------------------------
# Modèles profonds de l'étude. Tous : forward(x [B,1,L], rr [B,n_rr] | None) -> logit [B]
# ---------------------------------------------------------------------------

def act_layer(name):
    return {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.1), "gelu": nn.GELU}[name]()


class MultiScaleConv(nn.Module):
    """Branches Conv1d parallèles (k = 3, 5, 7) concaténées sur les canaux."""

    def __init__(self, cin, cout, stride=1, kernels=(3, 5, 7)):
        super().__init__()
        widths = [cout // len(kernels)] * len(kernels)
        widths[0] += cout - sum(widths)
        self.branches = nn.ModuleList(
            nn.Conv1d(cin, w, k, padding=k // 2, stride=stride) for w, k in zip(widths, kernels))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, kernel, bn, act, pool):
        stride = 2 if pool == "stride" else 1
        conv = (MultiScaleConv(cin, cout, stride) if kernel == "multi"
                else nn.Conv1d(cin, cout, kernel, padding=kernel // 2, stride=stride))
        layers = [conv]
        if bn:
            layers.append(nn.BatchNorm1d(cout))
        layers.append(act_layer(act))
        if pool == "max":
            layers.append(nn.MaxPool1d(2))
        super().__init__(*layers)


def _conv_stack(channels, kernel, bn, act, pool, last_pool):
    blocks, cin = [], 1
    for i, c in enumerate(channels):
        blocks.append(ConvBlock(cin, c, kernel, bn, act, pool if (i < len(channels) - 1 or last_pool) else "none"))
        cin = c
    return nn.Sequential(*blocks)


def _probe_length(module, length):
    was_training = module.training
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(1, 1, length))
    module.train(was_training)
    return out.shape[-1]


class MLP(nn.Module):
    """Perceptron multicouche sur la fenêtre aplatie (+ RR optionnel)."""

    def __init__(self, length=216, n_rr=0, hidden=(128, 64), act="relu", dropout=0.2, bn=False):
        super().__init__()
        layers, d = [], length + n_rr
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers += [act_layer(act), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.n_rr = n_rr

    def forward(self, x, rr=None):
        z = x.flatten(1)
        if self.n_rr:
            z = torch.cat([z, rr], dim=1)
        return self.net(z).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, length=216, n_rr=0, channels=(32, 64, 128), kernel=5, bn=True, act="relu",
                 pool="max", dropout=0.2, head="gap", rr_hidden=16):
        super().__init__()
        self.head_type = head
        self.features = _conv_stack(tuple(channels), kernel, bn, act, pool, last_pool=(head == "flatten"))
        t = _probe_length(self.features, length)
        c = channels[-1]
        if head == "flatten":
            self.dense = nn.Sequential(nn.Flatten(), nn.Linear(c * t, 64), act_layer(act))
            d = 64
        else:
            self.dense = None
            d = c
        self.rr = nn.Sequential(nn.Linear(n_rr, rr_hidden), act_layer(act)) if n_rr else None
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d + (rr_hidden if n_rr else 0), 1)
        self.gap_dim = c

    def representation(self, x):
        h = self.features(x)
        return h.mean(dim=-1) if self.dense is None else self.dense(h)

    def forward(self, x, rr=None):
        z = self.representation(x)
        if self.rr is not None:
            z = torch.cat([z, self.rr(rr)], dim=1)
        return self.out(self.drop(z)).squeeze(1)

    @torch.no_grad()
    def cam(self, x):
        """Class activation map (Zhou et al., 2016) pour head='gap' : sum_k w_k A_k(t), ré-échantillonné à L."""
        assert self.dense is None, "CAM défini pour head='gap'"
        a = self.features(x)
        w = self.out.weight[0, :self.gap_dim]
        cam = (w[None, :, None] * a).sum(1, keepdim=True)
        return torch.nn.functional.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[:, 0]


class CNNLSTM(nn.Module):
    """CNN (extraction locale, séquence de T pas) -> LSTM (dynamique sur ces T pas) -> logit."""

    def __init__(self, length=216, n_rr=0, channels=(32, 64), kernel=5, bn=True, act="relu",
                 cnn_dropout=0.2, hidden=64, layers=1, bidirectional=False, rnn_dropout=0.0,
                 pooling="last", head_dropout=0.3, rr_hidden=16):
        super().__init__()
        self.features = _conv_stack(tuple(channels), kernel, bn, act, "max", last_pool=True)
        self.seq_len = _probe_length(self.features, length)
        assert self.seq_len > 1, "séquence trop courte pour le LSTM"
        self.cnn_drop = nn.Dropout(cnn_dropout)
        self.lstm = nn.LSTM(channels[-1], hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=rnn_dropout if layers > 1 else 0.0)
        d = hidden * (2 if bidirectional else 1)
        self.pooling, self.bidirectional = pooling, bidirectional
        self.att = nn.Linear(d, 1) if pooling == "attention" else None
        self.rr = nn.Sequential(nn.Linear(n_rr, rr_hidden), act_layer(act)) if n_rr else None
        self.drop = nn.Dropout(head_dropout)
        self.out = nn.Linear(d + (rr_hidden if n_rr else 0), 1)

    def representation(self, x):
        h = self.cnn_drop(self.features(x)).transpose(1, 2)      # [B, T, C]
        out, (hn, _) = self.lstm(h)
        if self.pooling == "mean":
            return out.mean(dim=1)
        if self.pooling == "attention":
            a = torch.softmax(self.att(out), dim=1)
            self.last_attention = a.detach()
            return (a * out).sum(dim=1)
        return torch.cat([hn[-2], hn[-1]], dim=1) if self.bidirectional else hn[-1]

    def forward(self, x, rr=None):
        z = self.representation(x)
        if self.rr is not None:
            z = torch.cat([z, self.rr(rr)], dim=1)
        return self.out(self.drop(z)).squeeze(1)


FAMILIES = {"mlp": MLP, "cnn": CNN1D, "cnn_lstm": CNNLSTM}

PRESETS = {
    "mlp_a": ("mlp", {"hidden": (128, 64)}),
    "mlp_b": ("mlp", {"hidden": (256, 128, 64)}),
    "mlp_c": ("mlp", {"hidden": (512, 256, 64)}),
    "cnn_s": ("cnn", {"channels": (16, 32, 64)}),
    "cnn_m": ("cnn", {"channels": (32, 64, 128)}),
    "fcn": ("cnn", {"channels": (64, 128, 64)}),
    "cnnlstm_s": ("cnn_lstm", {"channels": (32, 64), "hidden": 64}),
    "cnnlstm_m": ("cnn_lstm", {"channels": (32, 64, 128), "hidden": 64}),
    "cnn_bilstm": ("cnn_lstm", {"channels": (32, 64), "hidden": 64, "bidirectional": True}),
}


def build_model(arch, params=None, length=216, n_rr=0):
    family, base = PRESETS[arch]
    kw = {**base, **(params or {})}
    return FAMILIES[family](length=length, n_rr=n_rr, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
