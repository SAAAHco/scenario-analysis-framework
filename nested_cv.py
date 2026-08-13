"""
Genuine nested cross-validation of the ensemble described in Section 3.5.

Outer 5-fold for unbiased performance estimation, inner 3-fold for
hyperparameter selection, members combined by inverse-error weighting.
Resumable: each (fold, scenario) chunk is appended to nested_cv_partial.csv,
so the script can be re-invoked until all 20 chunks are complete.

Tree counts are reduced relative to the production scripts to fit a single
CPU; the cross-validation structure itself is exactly as described in
Section 3.5. Per-scenario results are reported in Supplementary Table S1.

This is the standalone, resumable implementation. methods_verification.py
contains an equivalent routine run as part of the full verification suite;
the two are independent and either may be used.
"""
import sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 230.0
import os

# Paths are resolved relative to this file so the scripts run anywhere after a
# clone. Override with the SAF_DATA and SAF_OUT environment variables if your
# inputs or outputs live elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SAF_DATA", os.path.join(_HERE, "data"))
OUT_DIR = os.environ.get("SAF_OUT", os.path.join(_HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
PARTIAL = os.path.join(OUT_DIR, "nested_cv_partial.csv")

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

SCEN = ["Sino-Russian Partnership", "Russian Consolidation",
        "Fragmented Competition", "Western Coalition"]


def build_targets(df, rng):
    need = 80.0
    sr = (0.467 * df["capability_score"] * 1.2
          + 0.277 * df["eastern_momentum"] * 1.1
          + 0.160 * df["eastern_feasibility"]
          + 0.096 * df["partnership_synergy"] * 1.3) * (1 + df["eastern_alignment"] * 0.2)
    ru = (0.467 * (df["russian_icebreakers"] / 50)
          + 0.277 * (df["russian_investment"] / 20) * df["resilience_factor"]
          + 0.160 * df["governance_efficiency"]) * df["sanctions_resilience"]
    fr = (0.15 * df["coordination_friction"]
          + 0.10 * (1 - df["governance_efficiency"])
          + 0.05 * (df["num_decision_entities"] / 7)) * (2 - df["eastern_dominance_index"].clip(upper=1.9))
    we = (0.467 * (df["western_icebreakers"] / 50)
          + 0.277 * (df["western_committed"] / need)
          + 0.160 * df["western_coordination"]
          + 0.096 * 0.1) * (1 + df["tech_disruption_potential"])
    t = 10
    sr = sr * np.exp(-0.043 * t) * (1 + 0.20 * t)
    ru = ru * np.exp(-0.043 * t) * (1 + 0.15 * t)
    we = we * np.exp(-0.14 * t) * (1 + 0.05 * t)
    fr = fr * np.exp(-0.30 * t) * (1 + 0.02 * t)
    P = np.column_stack([sr.clip(lower=.01), ru.clip(lower=.01),
                         fr.clip(lower=.01), we.clip(lower=.01)])
    P = P / P.sum(axis=1, keepdims=True)
    P = P * 0.95 + rng.dirichlet([20, 20, 20, 20], size=len(df)) * 0.05
    return P / P.sum(axis=1, keepdims=True)


GRIDS = {
    "XGBoost":      (XGBRegressor(random_state=SEED, verbosity=0, n_jobs=1),
                     {"n_estimators": [150], "max_depth": [6], "learning_rate": [0.1]}),
    "LightGBM":     (LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1),
                     {"n_estimators": [150], "num_leaves": [15, 31]}),
    "RandomForest": (RandomForestRegressor(random_state=SEED, n_jobs=1),
                     {"n_estimators": [50], "max_depth": [None, 12]}),
    "ExtraTrees":   (ExtraTreesRegressor(random_state=SEED, n_jobs=1),
                     {"n_estimators": [50], "max_depth": [None, 12]}),
    "MLP":          (MLPRegressor(random_state=SEED, max_iter=300,
                                  early_stopping=True, n_iter_no_change=10),
                     {"hidden_layer_sizes": [(64, 32), (32, 16)]}),
}


def main():
    start = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "training_data.csv"))
    y = build_targets(df, np.random.default_rng(SEED))
    X = df.values

    done = set()
    if os.path.exists(PARTIAL):
        prev = pd.read_csv(PARTIAL)
        done = set(zip(prev.fold, prev.scenario))

    outer = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    todo = [(f, j) for f in range(1, 6) for j in range(4)
            if (f, SCEN[j]) not in done]
    print(f"chunks remaining: {len(todo)}/20")

    for fold, j in todo:
        if time.time() - start > BUDGET:
            print("budget reached, stopping cleanly")
            break
        tr, te = outer[fold - 1]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

        preds, wts = [], []
        for name, (est, grid) in GRIDS.items():
            gs = GridSearchCV(est, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=1)
            gs.fit(Xtr, y[tr, j])
            preds.append(gs.best_estimator_.predict(Xte))
            wts.append(1.0 / (-gs.best_score_ + 1e-6))
        w = np.array(wts) / np.sum(wts)
        ens = np.average(np.array(preds), axis=0, weights=w)

        row = pd.DataFrame([{
            "fold": fold, "scenario": SCEN[j],
            "r2": r2_score(y[te, j], ens),
            "mae": mean_absolute_error(y[te, j], ens),
        }])
        row.to_csv(PARTIAL, mode="a", header=not os.path.exists(PARTIAL), index=False)
        print(f"  fold {fold} / {SCEN[j]:<26} r2={row.r2[0]:.4f} "
              f"mae={row.mae[0]:.5f}  [{time.time()-start:.0f}s]")

    if os.path.exists(PARTIAL) and len(pd.read_csv(PARTIAL)) == 20:
        res = pd.read_csv(PARTIAL)
        s = res.groupby("scenario").agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                                        mae_mean=("mae", "mean"), mae_sd=("mae", "std")).reindex(SCEN)
        print("\nCOMPLETE. Nested CV (outer 5-fold, inner 3-fold), 5,000 samples\n")
        print(f"{'Scenario':<26}{'R2 mean +- SD':>22}{'MAE mean +- SD':>24}")
        for k, r in s.iterrows():
            print(f"{k:<26}{r.r2_mean:>13.4f} +- {r.r2_sd:.4f}{r.mae_mean:>15.5f} +- {r.mae_sd:.5f}")
        print(f"\nPooled R2 {res.r2.mean():.4f}   Pooled MAE {res.mae.mean():.5f}")
        s.to_csv(os.path.join(OUT_DIR, "nested_cv_summary.csv"))


if __name__ == "__main__":
    main()
