# Scenario-Based Geopolitical Analysis Framework

A generalized machine learning and Monte Carlo simulation framework for multi-scenario geopolitical and strategic analysis.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides the machine learning and simulation code for
scenario-based geopolitical analysis:

- **Multi-Criteria Decision Analysis (MCDA)** with AHP-derived weights
- **Ensemble Machine Learning** (6 algorithms with inverse-error weighting)
- **Monte Carlo Simulation** for uncertainty quantification
- **Reproducible pipeline** with fixed seeds and documented iteration counts

The framework is designed to be domain-agnostic: customize the configuration classes with your own variables, scenarios, and logic.

---

## Features

- **Configurable Scenarios**: Define 2-6 scenarios with custom probability logic
- **MCDA Integration**: Four-criterion framework (Capability, Momentum, Feasibility, Synergy)
- **ML Ensemble**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Neural Network
- **Monte Carlo**: 10,000+ iterations with percentile intervals
- **Diagnostic plots**: Distribution plots, bar charts, feature importance
- **Export**: CSV, JSON, and optional Word reports

---

## Repository Structure

```
├── scenario_ml_framework.py    # Main ML ensemble framework
├── monte_carlo_framework.py    # Monte Carlo simulation engine
├── methods_verification.py     # MCDA weight stability and nested cross-validation
├── nested_cv.py                # Standalone resumable nested cross-validation
├── requirements.txt            # Python dependencies
├── CODEBOOK_TEMPLATE.md        # Variable documentation template
├── README.md                   # This file
│
├── .gitignore                  # Excludes generated outputs and local data
│
├── data/                       # Inputs (override with SAF_DATA); see data/README.md
└── outputs/                    # Generated outputs (override with SAF_OUT)
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
# The repository URL is withheld while the associated manuscript is under
# double anonymized review; it will be inserted here on acceptance.
git clone <REPOSITORY-URL-WITHHELD-DURING-ANONYMIZED-REVIEW>
cd scenario-analysis-framework

# Install dependencies
pip install -r requirements.txt
```

### 2. Customize Configuration

Edit the configuration classes in `scenario_ml_framework.py`:

```python
class ScenarioConfig:
    # Define your scenarios
    SCENARIOS = [
        'Partnership Scenario',
        'Dominance Scenario',
        'Competition Scenario',
        'Coalition Scenario'
    ]
    
    # Set MCDA weights (must sum to 1.0)
    MCDA_WEIGHTS = {
        'capability': 0.467,
        'momentum': 0.277,
        'feasibility': 0.160,
        'synergy': 0.096
    }
```

```python
class BaselineParameters:
    # Define your domain variables
    VARIABLES = {
        'actor_a_capability': {
            'baseline': 40,
            'low': 35, 'base': 45, 'high': 55,
            'distribution': 'normal',
            'params': {'mean': 40, 'std': 5}
        },
        # ... more variables
    }
```

### 3. Run Analysis

```bash
# Run ML ensemble analysis
python scenario_ml_framework.py

# Run Monte Carlo simulation
python monte_carlo_framework.py

# Verify MCDA weight stability and run nested cross-validation
python methods_verification.py

# Standalone resumable nested cross-validation
python nested_cv.py
```

### 4. Paths

Scripts resolve inputs and outputs relative to their own location, so they run
as-is after a clone. Inputs are read from `data/` and results written to
`outputs/`, which is created automatically. To point elsewhere, set:

```bash
export SAF_DATA=/path/to/inputs
export SAF_OUT=/path/to/results
```

---

## Customization Guide

### Defining Variables

Each variable requires:

| Field | Description | Example |
|-------|-------------|---------|
| `baseline` | Current/observed value | `40` |
| `low` | Conservative projection | `35` |
| `base` | Central projection | `45` |
| `high` | Optimistic projection | `55` |
| `distribution` | Statistical distribution | `'normal'`, `'uniform'`, `'beta'`, `'lognormal'` |
| `params` | Distribution parameters | `{'mean': 40, 'std': 5}` |

### Creating Engineered Features

Customize `create_engineered_features()` method:

```python
def create_engineered_features(self, df):
    df = df.copy()
    
    # Your domain-specific feature engineering
    df['capability_ratio'] = df['actor_a'] / (df['actor_b'] + 1)
    df['momentum_index'] = 0.4 * df['growth'] + 0.3 * df['investment']
    # ... more features
    
    return df
```

### Defining Scenario Logic

Customize `generate_scenario_probabilities()` method:

```python
def generate_scenario_probabilities(self, df):
    # Your scenario probability calculations
    scenario_a = (
        w['capability'] * row['cap_score'] +
        w['momentum'] * row['momentum_index'] +
        # ...
    )
    # Normalize and return probabilities
```

---

## Methodology

### MCDA Framework

The framework uses a four-criterion MCDA model:

```
L = α×C + β×M + γ×F + δ×S
```

Where:
- **C** = Capability criterion (infrastructure, assets)
- **M** = Momentum criterion (growth, investment)
- **F** = Feasibility criterion (governance, coordination)
- **S** = Synergy criterion (partnerships, alignment)
- **α, β, γ, δ** = AHP-derived weights

### ML Ensemble

Six algorithms with inverse-error weighting:

1. **XGBoost** - Regularized gradient boosting
2. **LightGBM** - Efficient gradient boosting
3. **CatBoost** - Categorical feature support
4. **Random Forest** - Bagged decision trees
5. **Extra Trees** - Extremely randomized trees
6. **Neural Network** - Multi-layer perceptron (256-128-64-32)

Members 1 to 5 run a fixed 200 boosting rounds or trees. Member 6 is the only
one that reserves an internal validation fraction (10% of the training
partition) and stops early.

### Monte Carlo Simulation

- **Output**: Mean, percentile intervals, convergence diagnostics

**Iteration counts differ by analysis, and the defaults reflect this:**

| Analysis | Iterations | Where set |
|----------|-----------|-----------|
| Scenario projection intervals (Table 2, Figure 3) | 1,000 | `ScenarioConfig.MC_ITERATIONS` |
| Threshold identification (SI Section S5) | 10,000 | `MCDAConfig.N_ITERATIONS` |
| MCDA weight stability (SI Section S2) | 10,000 | `methods_verification.py` |

Running a script with a different count will not reproduce the published
intervals.

### Model checking

Reported error statistics are internal diagnostics. Training samples are
generated by structural causal simulation rather than observed, so a low
held-out error shows that the ensemble has recovered the specified generating
relation without overfitting it. It is not evidence of predictive accuracy
against real-world outcomes, and no such claim is made. Checks provided are:

- Held-out partition and nested cross-validation (`nested_cv.py`,
  `methods_verification.py`)
- MCDA weight stability under resampling (`methods_verification.py`)
- One-at-a-time and two-dimensional parameter sweeps
- Convergence analysis of the Monte Carlo estimates

---

## Output Files

| File | Description |
|------|-------------|
| `training_data.csv` | Simulated training samples |
| `predictions_2035.csv` | Scenario probability predictions |
| `monte_carlo_results.csv` | Full Monte Carlo simulation output |
| `parameter_sweep_results.csv` | Two-dimensional investment x fleet sweep |
| `nested_cv_results.csv` | Per-fold, per-scenario cross-validation metrics |
| `nested_cv_summary.csv` | Aggregated cross-validation summary |
| `mcda_stability_results.csv` | Ranking stability under weight perturbation |
| `config.json` | Analysis configuration |

`scenario_ml_framework.py` and `monte_carlo_framework.py` also emit diagnostic
plots at runtime. The scripts that produced the figures in the associated
article are not included here; this repository covers the model and its
verification only.

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional (recommended)
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
python-docx>=0.8.11
```

---

## Citation

If you use this framework, please cite:

```bibtex
@article{AUTHOR2026scenario,
  title  = {Energy Geopolitics in the Arctic: A Scenario-Based Strategic Analysis},
  author = {[AUTHOR NAMES WITHHELD DURING ANONYMIZED REVIEW]},
  journal= {Energy Research & Social Science},
  year   = {2026},
  note   = {Under review}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## Contact

For questions or collaboration: contact details are withheld while the associated manuscript is under double anonymized review, and will be added on acceptance.

---

## Acknowledgments

- Analytic Hierarchy Process (Saaty, 1980)
- Ensemble Methods (Dietterich, 2000)
- Monte Carlo Methods (Metropolis & Ulam, 1949; Robert & Casella, 1999)
- Cross-validation (Kohavi, 1995)
- Permutation feature importance (Fisher, Rudin & Dominici, 2019)
