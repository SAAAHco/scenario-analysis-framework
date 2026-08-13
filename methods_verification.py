"""
Methods verification for the Arctic scenario manuscript.

Implements the three methodological claims in Sections 3.3.3, 3.5 and 3.6.2 that
are asserted in the text but not implemented in the analysis scripts, and reports
the values they actually produce.

  Part A. Feature count             (claim: 47 engineered features)
  Part B. MCDA weight robustness    (claim: rankings stable in 94% of iterations)
  Part C. Nested cross-validation   (claim: outer 5-fold, inner 3-fold)

Inputs : training_data.csv  (5,000 x 61 feature matrix as saved by ML.py)
Outputs: printed summary; nested_cv_results.csv; mcda_stability_results.csv
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)

import os

# Paths are resolved relative to this file so the scripts run anywhere after a
# clone. Override with the SAF_DATA and SAF_OUT environment variables if your
# inputs or outputs live elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SAF_DATA", os.path.join(_HERE, "data"))
OUT_DIR = os.environ.get("SAF_OUT", os.path.join(_HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
DATA = os.path.join(DATA_DIR, "training_data.csv")


# ----------------------------------------------------------------------
# Part A. Feature inventory
# ----------------------------------------------------------------------
def part_a():
    df = pd.read_csv(DATA)
    engineered = [
        "capability_score", "total_eastern_icebreakers", "icebreaker_ratio",
        "total_eastern_investment", "investment_gap", "investment_ratio",
        "experience_advantage", "port_capacity_ratio", "eastern_momentum",
        "eastern_feasibility", "partnership_synergy", "capability_decay",
        "infrastructure_lock_in", "experience_lock_in", "capability_gap",
        "investment_commitment_gap", "sanctions_resilience",
        "supply_chain_diversity", "tech_disruption_potential",
        "eastern_alignment", "western_coordination", "ice_melt_advantage",
        "resource_extraction_viability", "eastern_dominance_index",
    ]
    present = [c for c in engineered if c in df.columns]
    print("PART A. FEATURE INVENTORY")
    print(f"  Columns in training_data.csv : {df.shape[1]}")
    print(f"  Derived / engineered columns : {len(present)}")
    print(f"  Base input columns           : {df.shape[1] - len(present)}")
    print(f"  Manuscript Section 3.4 claim : 47 engineered features")
    return df


# ----------------------------------------------------------------------
# Part B. MCDA weight robustness
# ----------------------------------------------------------------------
# Central criterion scores, read from the perturbation block in Analysis.py.
# Fragmented Competition is specified there only as an aggregate index, so it
# carries no criterion decomposition and is weight-invariant by construction.
CRITERIA = {
    "Sino-Russian Partnership": dict(C=1.000, M=0.886, F=0.2235, S=0.775),
    "Russian Consolidation":    dict(C=0.931, M=0.665, F=0.1750, S=0.000),
    "Western Coalition":        dict(C=0.121, M=0.139, F=0.0270, S=0.054),
}
FRAGMENTED_INDEX = 0.253
W0 = dict(C=0.467, M=0.277, F=0.160, S=0.096)


def spi(weights):
    raw = {k: sum(weights[c] * v[c] for c in "CMFS") for k, v in CRITERIA.items()}
    raw["Fragmented Competition"] = FRAGMENTED_INDEX
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def part_b(n_iterations=10000):
    baseline_rank = sorted(spi(W0), key=lambda k: -spi(W0)[k])
    print("\nPART B. MCDA WEIGHT ROBUSTNESS")
    print("  Baseline ranking: " + " > ".join(baseline_rank))

    rows = []
    for rel_sd in (0.10, 0.20, 0.30):
        stable = 0
        for _ in range(n_iterations):
            w = {c: max(rng.normal(W0[c], rel_sd * W0[c]), 1e-6) for c in "CMFS"}
            s = sum(w.values())
            w = {c: v / s for c, v in w.items()}
            order = sorted(spi(w), key=lambda k: -spi(w)[k])
            stable += order == baseline_rank
        pct = 100 * stable / n_iterations
        rows.append({"relative_sd": rel_sd, "iterations": n_iterations,
                     "ranking_stability_pct": round(pct, 2)})
        print(f"  Weight SD = {rel_sd:>4.0%} of central value  ->  "
              f"ranking preserved in {pct:6.2f}% of {n_iterations} iterations")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "mcda_stability_results.csv"), index=False)
    print("  Manuscript Section 3.3.3 claim : 94%")
    return out


# ----------------------------------------------------------------------
# Part C. Nested cross-validation of the ensemble
# ----------------------------------------------------------------------
def build_targets(df):
    """Reproduces _generate_sophisticated_probabilities from ML.py."""
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

    P = np.column_stack([sr.clip(lower=0.01), ru.clip(lower=0.01),
                         fr.clip(lower=0.01), we.clip(lower=0.01)])
    P = P / P.sum(axis=1, keepdims=True)
    noise = rng.dirichlet([20, 20, 20, 20], size=len(df)) * 0.05
    P = P * 0.95 + noise
    return P / P.sum(axis=1, keepdims=True)


def part_c(df, n_outer=5, n_inner=3):
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    scenarios = ["Sino-Russian Partnership", "Russian Consolidation",
                 "Fragmented Competition", "Western Coalition"]
    X = df.values
    y = build_targets(df)

    grids = {
        "XGBoost":      (XGBRegressor(random_state=SEED, verbosity=0),
                         {"n_estimators": [200], "max_depth": [3, 6], "learning_rate": [0.1]}),
        "LightGBM":     (LGBMRegressor(random_state=SEED, verbose=-1),
                         {"n_estimators": [200], "num_leaves": [15, 31], "learning_rate": [0.1]}),
        "RandomForest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                         {"n_estimators": [200], "max_depth": [None, 12]}),
        "ExtraTrees":   (ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
                         {"n_estimators": [200], "max_depth": [None, 12]}),
        "MLP":          (MLPRegressor(random_state=SEED, max_iter=400),
                         {"hidden_layer_sizes": [(128, 64), (64, 32)]}),
    }

    print(f"\nPART C. NESTED CROSS-VALIDATION "
          f"(outer {n_outer}-fold, inner {n_inner}-fold)")
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    fold_rows = []

    for fold, (tr, te) in enumerate(outer.split(X), 1):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ens_pred = np.zeros((len(te), 4))

        for j, scen in enumerate(scenarios):
            preds, weights = [], []
            for name, (est, grid) in grids.items():
                gs = GridSearchCV(est, grid, cv=n_inner,
                                  scoring="neg_mean_squared_error", n_jobs=-1)
                gs.fit(Xtr, y[tr, j])
                preds.append(gs.best_estimator_.predict(Xte))
                weights.append(1.0 / (-gs.best_score_ + 1e-6))  # inverse-error weighting
            w = np.array(weights) / np.sum(weights)
            ens_pred[:, j] = np.average(np.array(preds), axis=0, weights=w)

        ens_pred = np.clip(ens_pred, 1e-9, None)
        ens_pred = ens_pred / ens_pred.sum(axis=1, keepdims=True)

        for j, scen in enumerate(scenarios):
            fold_rows.append({
                "fold": fold, "scenario": scen,
                "r2": r2_score(y[te, j], ens_pred[:, j]),
                "mae": mean_absolute_error(y[te, j], ens_pred[:, j]),
            })
        print(f"  outer fold {fold}/{n_outer} complete")

    res = pd.DataFrame(fold_rows)
    summary = res.groupby("scenario").agg(
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std")).reindex(scenarios)

    print(f"\n  {'Scenario':<26}{'R2 (mean +- SD)':>22}{'MAE (mean +- SD)':>24}")
    for scen, r in summary.iterrows():
        print(f"  {scen:<26}{r.r2_mean:>13.4f} +- {r.r2_sd:.4f}"
              f"{r.mae_mean:>15.5f} +- {r.mae_sd:.5f}")
    print(f"\n  Pooled R2 across scenarios : {res.r2.mean():.4f}")
    print(f"  Pooled MAE across scenarios: {res.mae.mean():.5f}")

    res.to_csv(os.path.join(OUT_DIR, "nested_cv_results.csv"), index=False)
    return summary


if __name__ == "__main__":
    print("=" * 72)
    d = part_a()
    part_b()
    part_c(d)
    print("=" * 72)
