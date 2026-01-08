# Scenario-Based Geopolitical Analysis Framework

A generalized machine learning and Monte Carlo simulation framework for multi-scenario geopolitical and strategic analysis.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides a reusable framework for conducting scenario-based geopolitical analysis using:

- **Multi-Criteria Decision Analysis (MCDA)** with AHP-derived weights
- **Ensemble Machine Learning** (6 algorithms with inverse-error weighting)
- **Monte Carlo Simulation** for uncertainty quantification
- **Publication-ready visualizations**

The framework is designed to be domain-agnostic: customize the configuration classes with your own variables, scenarios, and logic.

---

## Features

- **Configurable Scenarios**: Define 2-6 scenarios with custom probability logic
- **MCDA Integration**: Four-criterion framework (Capability, Momentum, Feasibility, Synergy)
- **ML Ensemble**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Neural Network
- **Monte Carlo**: 10,000+ iterations with confidence intervals
- **Visualizations**: Distribution plots, bar charts, feature importance
- **Export**: CSV, JSON, and optional Word reports

---

## Repository Structure

```
├── scenario_ml_framework.py    # Main ML ensemble framework
├── monte_carlo_framework.py    # Monte Carlo simulation engine
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CODEBOOK.md                 # Variable documentation template
│
└── outputs/                    # Generated outputs (created at runtime)
    ├── figures/
    └── data/
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/scenario-analysis-framework.git
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

### Monte Carlo Simulation

- **Default**: 10,000 iterations
- **Output**: Mean, std, 95% CI (bias-corrected accelerated bootstrap)
- **Validation**: Convergence analysis, Sobol sensitivity indices

---

## Output Files

| File | Description |
|------|-------------|
| `training_data.csv` | Synthetic training samples |
| `predictions.csv` | Scenario probability predictions |
| `monte_carlo_results.csv` | Full MC simulation output |
| `config.json` | Analysis configuration |
| `figures/*.png` | Visualization plots |

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
@article{author2025scenario,
  title={Scenario-Based Geopolitical Analysis: A Machine Learning Framework},
  author={Ashkanani},
  journal={[Journal of Energy Policy]},
  year={2025},
  doi={[Under Review]}
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

For questions or collaboration: Ashkanani@tamu.edu

---

## Acknowledgments

- Analytic Hierarchy Process (Saaty, 1980)
- Ensemble Methods (Dietterich, 2000)
- Monte Carlo Methods (Robert & Casella, 1999)
