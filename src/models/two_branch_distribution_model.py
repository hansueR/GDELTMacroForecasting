import torch
import torch.nn as nn
import torch.nn.functional as F


class PriceLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, x_price):
        _, (h_n, _) = self.lstm(x_price)
        return h_n[-1]


class EventAttentionEncoder(nn.Module):
    """
    Event encoder with:
    1. learnable attention over event dimensions
    2. temporal attention over lookback steps
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.feature_logits = nn.Parameter(torch.zeros(input_dim))
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.time_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_event):
        feature_weight = torch.softmax(self.feature_logits, dim=0)
        x_weighted = x_event * feature_weight.view(1, 1, -1) * x_event.shape[-1]

        h = torch.tanh(self.proj(x_weighted))
        h = self.dropout(h)

        score = self.time_score(h).squeeze(-1)
        alpha = torch.softmax(score, dim=1)

        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        return context, alpha, feature_weight


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim=64, asset_emb_dim=0, dropout=0.1):
        super().__init__()
        gate_input_dim = hidden_dim * 2 + asset_emb_dim
        self.gate = nn.Linear(gate_input_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + asset_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, price_h, event_h, asset_emb=None):
        if asset_emb is None:
            gate_input = torch.cat([price_h, event_h], dim=-1)
        else:
            gate_input = torch.cat([price_h, event_h, asset_emb], dim=-1)

        g = torch.sigmoid(self.gate(gate_input))
        fused = g * price_h + (1.0 - g) * event_h

        if asset_emb is not None:
            fused = torch.cat([fused, asset_emb], dim=-1)

        return self.out(fused)


class TwoBranchDistributionModel(nn.Module):
    """
    Proposal-aligned main model:
    - price encoder: LSTM
    - event encoder: attention over event dimensions and time
    - fusion module: gating
    - multi-horizon head: outputs mu and sigma for 1d/3d/5d returns
    """
    def __init__(
        self,
        price_dim,
        event_dim,
        num_assets,
        hidden_dim=64,
        lstm_layers=1,
        dropout=0.1,
        horizons=(1, 3, 5),
        asset_emb_dim=8,
        sigma_floor=1e-6,
    ):
        super().__init__()
        self.horizons = tuple(horizons)
        self.sigma_floor = sigma_floor
        self.asset_emb_dim = asset_emb_dim

        self.price_encoder = PriceLSTMEncoder(
            input_dim=price_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.event_encoder = EventAttentionEncoder(
            input_dim=event_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        if asset_emb_dim > 0:
            self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        else:
            self.asset_embedding = None

        self.fusion = GatedFusion(
            hidden_dim=hidden_dim,
            asset_emb_dim=asset_emb_dim if asset_emb_dim > 0 else 0,
            dropout=dropout,
        )

        self.head = nn.Linear(hidden_dim, 2 * len(self.horizons))

    def forward(self, x_price, x_event, asset_id=None):
        price_h = self.price_encoder(x_price)
        event_h, time_alpha, feature_weight = self.event_encoder(x_event)

        if self.asset_embedding is not None:
            if asset_id is None:
                raise ValueError("asset_id is required when asset_emb_dim > 0")
            asset_emb = self.asset_embedding(asset_id)
        else:
            asset_emb = None

        fused = self.fusion(price_h, event_h, asset_emb)
        raw = self.head(fused)

        n_h = len(self.horizons)
        mu = raw[:, :n_h]
        raw_sigma = raw[:, n_h:]
        sigma = F.softplus(raw_sigma) + self.sigma_floor

        return {
            "mu": mu,
            "sigma": sigma,
            "time_attention": time_alpha,
            "feature_attention": feature_weight,
        }


def gaussian_nll_loss(y_true, mu, sigma):
    sigma = torch.clamp(sigma, min=1e-12)
    var = sigma ** 2
    nll = 0.5 * (torch.log(2.0 * torch.pi * var) + ((y_true - mu) ** 2) / var)
    return nll.mean()
