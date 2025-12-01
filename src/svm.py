import sys, os
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
)
import time


sys.path.append(os.path.abspath("../src"))
from utils import to_numpy


def train_eval_svm_rbf_cv(X, y, C=3.0, gamma="scale", random_state=0, n_splits=5):
    X_np = to_numpy(X)
    y_np = to_numpy(y).astype(int).ravel()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    accs, f1s, aucs, mses, times = [], [], [], [], []

    for train_idx, val_idx in cv.split(X_np, y_np):
        X_tr_cv = X_np[train_idx]
        y_tr_cv = y_np[train_idx]
        X_va_cv = X_np[val_idx]
        y_va_cv = y_np[val_idx]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        t0 = time.perf_counter()
        model.fit(X_tr_cv, y_tr_cv)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        proba = model.predict_proba(X_va_cv)[:, 1]
        pred = (proba >= 0.5).astype(int)

        accs.append(accuracy_score(y_va_cv, pred))
        f1s.append(f1_score(y_va_cv, pred, zero_division=0))

        try:
            aucs.append(roc_auc_score(y_va_cv, proba))
        except Exception:
            aucs.append(np.nan)

        mses.append(mean_squared_error(y_va_cv, proba))

    def fmt_mean_std(arr, allow_nan=False):
        arr = np.asarray(arr, dtype=float)
        if allow_nan:
            m = np.nanmean(arr)
            s = np.nanstd(arr)
        else:
            m = arr.mean()
            s = arr.std()
        return f"{m:.4f} ± {s:.4f}"

    row = {
        "model": f"SVM_RBF(C={C}, gamma={gamma})",
        "accuracy": fmt_mean_std(accs),
        "f1": fmt_mean_std(f1s),
        "roc_auc": fmt_mean_std(aucs, allow_nan=True),
        "mse": fmt_mean_std(mses),
        "train_time_s": fmt_mean_std(times),
    }

    return row
