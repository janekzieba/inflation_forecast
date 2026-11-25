"""
Model LSTM do prognozowania CPI (PyTorch).

Warianty (sekcja 3.3 kamienia milowego 1):
  LSTM-L        cel: poziom CPI, strategia rekurencyjna
  LSTM-Δ        cel: przyrost Δy (rekonstrukcja poziomu po predykcji), rekurencyjnie
  LSTM-Δ+kal    jak wyżej + cyklicznie zakodowany miesiąc (sin, cos)
  LSTM-MO       Δy + kalendarz, bezpośrednia prognoza wektora 12 kroków (multi-output)

Skalowanie: StandardScaler dopasowany wyłącznie na części treningowej (bez 24 mies. walidacji).
Early stopping na chronologicznym zbiorze walidacyjnym (ostatnie 24 miesiące treningu).
Niepewność: MC-dropout (dropout aktywny przy inferencji, N przebiegów).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@dataclass
class LSTMConfig:
    target: str = "delta"          # "level" | "delta"
    window: int = 12
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    calendar: bool = False
    multi_output: bool = False
    H: int = 12
    loss: str = "huber"            # "mse" | "huber"
    epochs: int = 400
    patience: int = 20
    min_epochs: int = 20
    batch_size: int = 16
    val_months: int = 24
    weight_decay: float = 1e-4
    seed: int = 0

    @property
    def out_dim(self):
        return self.H if self.multi_output else 1

    @property
    def n_in(self):
        return 3 if self.calendar else 1


VARIANTS = {
    "LSTM-L": dict(target="level", calendar=False, multi_output=False),
    "LSTM-Δ": dict(target="delta", calendar=False, multi_output=False),
    "LSTM-Δ+kal": dict(target="delta", calendar=True, multi_output=False),
    "LSTM-MO": dict(target="delta", calendar=True, multi_output=True),
}


class LSTMNet(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(cfg.n_in, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                            dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, cfg.out_dim)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.drop(o[:, -1, :]))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMForecaster:
    """Opakowanie: przygotowanie okien, trening, prognozy z wielu punktów startowych, MC-dropout."""

    def __init__(self, cfg: LSTMConfig):
        self.cfg = cfg
        self.net = None
        self.mu = 0.0
        self.sd = 1.0
        self.history = {"train": [], "val": []}
        self.best_epoch = None
        self.train_time = None

    # ---------- dane ----------
    def _target_series(self, y: pd.Series) -> pd.Series:
        return y if self.cfg.target == "level" else y.diff()

    def _scaled(self, s: pd.Series) -> np.ndarray:
        return (s.values - self.mu) / self.sd

    @staticmethod
    def _calendar(dates: pd.DatetimeIndex) -> np.ndarray:
        m = dates.month.values
        return np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])

    def _inputs(self, z: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        """z: (T,) przeskalowany szereg -> (T, n_in)."""
        if self.cfg.calendar:
            return np.column_stack([z, self._calendar(dates)])
        return z[:, None]

    def _windows(self, feats, z, t_positions):
        w, o = self.cfg.window, self.cfg.out_dim
        X = np.stack([feats[t - w + 1:t + 1] for t in t_positions])
        Y = np.stack([z[t + 1:t + 1 + o] for t in t_positions])
        return X.astype(np.float32), Y.astype(np.float32)

    # ---------- trening ----------
    def fit(self, y: pd.Series, train_end_pos: int, verbose: bool = False):
        """Uczy na y[:train_end_pos]. Ostatnie val_months obserwacji -> walidacja (early stopping)."""
        cfg = self.cfg
        set_seed(cfg.seed)
        s = self._target_series(y)
        val_start = train_end_pos - cfg.val_months
        first = 1 if cfg.target == "delta" else 0
        fit_part = s.values[first:val_start]
        self.mu, self.sd = float(np.mean(fit_part)), float(np.std(fit_part) + 1e-8)
        z = np.nan_to_num(self._scaled(s))
        feats = self._inputs(z, y.index)
        w, o = cfg.window, cfg.out_dim
        t0_min = first + w - 1
        # próbki treningowe: wszystkie cele < val_start; walidacyjne: cele w [val_start, train_end_pos)
        tr_t = [t for t in range(t0_min, val_start - o)]
        va_t = [t for t in range(max(t0_min, val_start - 1), train_end_pos - o)]
        Xtr, Ytr = self._windows(feats, z, tr_t)
        Xva, Yva = self._windows(feats, z, va_t)
        Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))

        self.net = LSTMNet(cfg)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.HuberLoss(delta=1.0) if cfg.loss == "huber" else nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(cfg.seed)
        t_start = time.time()
        for ep in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            tl = 0.0
            for b in range(0, len(Xtr), cfg.batch_size):
                idx = perm[b:b + cfg.batch_size]
                opt.zero_grad()
                loss = crit(self.net(Xtr[idx]), Ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tl += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                vl = crit(self.net(Xva), Yva).item()
            self.history["train"].append(tl / len(Xtr))
            self.history["val"].append(vl)
            if vl < best - 1e-6 or ep < cfg.min_epochs // 2:
                best, best_state, bad, self.best_epoch = vl, copy.deepcopy(self.net.state_dict()), 0, ep
            else:
                bad += 1
                if bad >= cfg.patience and ep >= cfg.min_epochs:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.train_time = time.time() - t_start
        if verbose:
            print(f"  best epoch {self.best_epoch}, val loss {best:.4f}, {self.train_time:.1f}s")
        return self

    # ---------- prognoza ----------
    def _predict_scaled(self, X: torch.Tensor, mc: bool) -> torch.Tensor:
        if mc:
            self.net.train()   # włącza dropout (MC-dropout)
        else:
            self.net.eval()
        with torch.no_grad():
            out = self.net(X)
        self.net.eval()
        return out

    def forecast(self, y: pd.Series, origins: np.ndarray, H: int | None = None, mc_samples: int = 0):
        """Prognozy poziomu CPI z punktów startowych `origins` (pozycje w y; dane znane do y[i] włącznie).
        Zwraca dict: point (n,H) oraz – gdy mc_samples>0 – lo, hi (95%), samples (S,n,H)."""
        cfg = self.cfg
        H = H or cfg.H
        s = self._target_series(y)
        z = np.nan_to_num(self._scaled(s))
        feats = self._inputs(z, y.index)
        w = cfg.window
        X0 = np.stack([feats[i - w + 1:i + 1] for i in origins]).astype(np.float32)
        origin_dates = y.index[origins]

        def run(mc: bool, reps: int):
            X = torch.from_numpy(np.repeat(X0, reps, axis=0))
            dates = np.repeat(np.array(origin_dates), reps)
            if cfg.multi_output:
                zp = self._predict_scaled(X, mc).numpy()[:, :H]
            else:
                zp = np.zeros((len(X), H), dtype=np.float32)
                cur = X
                for k in range(H):
                    p = self._predict_scaled(cur, mc)[:, 0]
                    zp[:, k] = p.numpy()
                    if cfg.calendar:
                        nd = pd.DatetimeIndex(dates) + pd.DateOffset(months=k + 1)
                        cal = torch.from_numpy(self._calendar(nd).astype(np.float32))
                        step = torch.column_stack([p, cal])
                    else:
                        step = p[:, None]
                    cur = torch.cat([cur[:, 1:, :], step[:, None, :]], dim=1)
            sp = zp * self.sd + self.mu
            if cfg.target == "delta":
                base = np.repeat(y.values[origins], reps)[:, None]
                sp = base + np.cumsum(sp, axis=1)
            return sp.reshape(len(origins), reps, H) if reps > 1 else sp

        out = {"point": run(False, 1)}
        if mc_samples > 0:
            samp = run(True, mc_samples)          # (n, S, H)
            out["lo"] = np.percentile(samp, 2.5, axis=1)
            out["hi"] = np.percentile(samp, 97.5, axis=1)
            out["mc_mean"] = samp.mean(axis=1)
            out["samples"] = samp
        return out

    def input_tensor(self, y: pd.Series, origins: np.ndarray) -> torch.Tensor:
        """Okna wejściowe (już przeskalowane) – używane przez moduł XAI."""
        s = self._target_series(y)
        z = np.nan_to_num(self._scaled(s))
        feats = self._inputs(z, y.index)
        w = self.cfg.window
        return torch.from_numpy(np.stack([feats[i - w + 1:i + 1] for i in origins]).astype(np.float32))

    # ---------- zapis / odczyt ----------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "cfg": asdict(self.cfg), "mu": self.mu, "sd": self.sd,
                    "best_epoch": self.best_epoch, "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecaster":
        d = torch.load(path, map_location="cpu", weights_only=False)
        f = cls(LSTMConfig(**d["cfg"]))
        f.net = LSTMNet(f.cfg)
        f.net.load_state_dict(d["state"])
        f.net.eval()
        f.mu, f.sd, f.best_epoch, f.history = d["mu"], d["sd"], d["best_epoch"], d["history"]
        return f


def make_config(variant: str, base: dict | None = None, **kw) -> LSTMConfig:
    params = dict(base or {})
    params.update(VARIANTS[variant])
    params.update(kw)
    return LSTMConfig(**params)
