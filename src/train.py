import torch
import sys, os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
)
import numpy as np
import time

sys.path.append(os.path.abspath("../src"))
from model import ResNetEuler
from utils import to_numpy
from activations import *


def accuracy(model: ResNetEuler, X: torch.Tensor, y: torch.Tensor) -> float:
    """Calcula la precisión (accuracy) del modelo."""
    probs = model(X)
    preds = (probs >= 0.5).long().view(-1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def train_model_backtracking(
    model: ResNetEuler,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    epochs: int = 200,
    L_init: float = 1.0,
    rho: float = 0.5,
    rbar: float = 2.0,
    verbose_every: int = 20,
    random: bool = True,
    end_criterion: float = 1e-4,
    batch_size: int = 64,
):
    """
    Bucle principal de entrenamiento.
    """
    N = X_train.shape[0]
    L = float(L_init)

    print(f"Iniciando entrenamiento con {epochs} épocas...")

    for epoch in range(1, epochs + 1):
        if random:
            perm = torch.randperm(N, device=X_train.device)
            X_epoch = X_train[perm]
            y_epoch = y_train[perm]
        else:
            X_epoch = X_train
            y_epoch = y_train

        total_loss = 0.0

        # Iteración por mini-batches
        for i in range(0, N, batch_size):
            Xb = X_epoch[i : i + batch_size]
            yb = y_epoch[i : i + batch_size]

            # Paso de optimización
            phi, L, accepted = model.train_step_backtracking(
                y0=Xb,
                c=yb,
                average_loss=True,
                L=L,
                rho=rho,
                rbar=rbar,
            )

            total_loss += float(phi) * (Xb.size(0) / N)

            # Early stopping si la loss es muy baja
            if phi < end_criterion:
                print(f"Early stopping at epoch {epoch}, loss={phi:.6f}")
                return L

            # Seguridad numérico
            if L > 1e6:
                print(f"Stopping at epoch {epoch}, L too large: L={L:.6f}")
                return L

        # Logs
        if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
            acc_tr = accuracy(model, X_train, y_train)
            msg = f"[{epoch:04d}] loss={total_loss:.6f} | acc_tr={acc_tr * 100:.2f}% | L={L:.4g}"
            if X_val is not None and y_val is not None:
                acc_va = accuracy(model, X_val, y_val)
                msg += f" | acc_val={acc_va * 100:.2f}%"
            print(msg)

    return L


def train_eval_resnet_cv(
    X,
    y,
    model_name="ResNetEuler",
    n_splits=5,
    num_layers=50,
    epochs=1000,
    delta_t=0.1,
    device="cpu",
):
    X = X.to(device)
    y = y.to(device)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    accs, f1s, aucs, mses, times = [], [], [], [], []

    for train_idx, val_idx in cv.split(to_numpy(X), to_numpy(y)):
        X_tr_cv = X[train_idx]
        y_tr_cv = y[train_idx]
        X_va_cv = X[val_idx]
        y_va_cv = y[val_idx]

        # Crear modelo desde cero en cada fold
        model = ResNetEuler(
            dim=X.shape[1],
            num_layers=num_layers,
            activation=tanh,
            activation_derivative=tanh_derivative,
            hip_function=sigmoid,
            hip_function_derivative=sigmoid_derivative,
            delta_t=delta_t,
        ).to(device)

        # Entrenamiento y tiempo
        t0 = time.perf_counter()
        train_model_backtracking(
            model,
            X_tr_cv,
            y_tr_cv,
            X_val=X_va_cv,
            y_val=y_va_cv,
            epochs=epochs,
            L_init=1.0,
            rho=0.5,
            rbar=2.0,
            verbose_every=999999,  # para que no imprima
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

        # Predicciones
        with torch.no_grad():
            proba = model(X_va_cv).cpu().numpy().ravel()
        y_true = to_numpy(y_va_cv).astype(int).ravel()
        pred = (proba >= 0.5).astype(int)

        # Métricas
        accs.append(accuracy_score(y_true, pred))
        f1s.append(f1_score(y_true, pred, zero_division=0))

        try:
            aucs.append(roc_auc_score(y_true, proba))
        except:
            aucs.append(np.nan)

        mses.append(mean_squared_error(y_true, proba))

    # Formato mean ± std
    def fmt(arr, allow_nan=False):
        arr = np.asarray(arr, dtype=float)
        if allow_nan:
            m = np.nanmean(arr)
            s = np.nanstd(arr)
        else:
            m = arr.mean()
            s = arr.std()
        return f"{m:.4f} ± {s:.4f}"

    row = {
        "model": model_name,
        "accuracy": fmt(accs),
        "f1": fmt(f1s),
        "roc_auc": fmt(aucs, allow_nan=True),
        "mse": fmt(mses),
        "train_time_s": fmt(times),
    }

    return row
