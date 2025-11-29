# CODEBOOK Template

## Variable Documentation for Scenario-Based Analysis

This template provides a structure for documenting variables in your scenario-based geopolitical analysis. Complete this codebook with your domain-specific variables.

---

## How to Use This Template

1. **Define each variable** following the structure below
2. **Specify distributions** based on your data and expert judgment
3. **Document sources** for transparency and reproducibility
4. **Note assumptions** that affect projections

---

## Variable Documentation Structure

For each variable, document:

| Field | Description |
|-------|-------------|
| **Variable Name** | Identifier used in code |
| **Definition** | Clear description of what the variable measures |
| **Baseline Value** | Current/observed value with date |
| **Unit** | Measurement unit |
| **Distribution Type** | `normal`, `uniform`, `beta`, `lognormal`, `triangular` |
| **Distribution Parameters** | mean/std, low/high, alpha/beta, etc. |
| **Projection (Low)** | Conservative future estimate |
| **Projection (Base)** | Central future estimate |
| **Projection (High)** | Optimistic future estimate |
| **Primary Sources** | Data sources with citations |
| **Confidence Level** | High / Medium / Low / Very Low |
| **Key Assumptions** | Underlying assumptions |
| **Scenario Dependency** | How variable changes across scenarios |

---

## Example Variable Entry

### actor_a_capability

| Field | Value |
|-------|-------|
| **Definition** | Aggregate capability index for Actor A |
| **Baseline Value** | 40 (as of 2025) |
| **Unit** | Index score |
| **Distribution** | Normal |
| **Parameters** | mean=40, std=5 |
| **Low Projection** | 35 |
| **Base Projection** | 45 |
| **High Projection** | 55 |
| **Sources** | [Source 1], [Source 2] |
| **Confidence** | High |
| **Assumptions** | Current trends continue; no major disruptions |
| **Scenario Dependency** | Higher in Scenario A (50-55); Lower in Scenario C (35-40) |

---

## Variable Categories

Organize your variables into logical categories:

### 1. Capability Variables
Variables measuring infrastructure, assets, and capacity.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

### 2. Investment Variables
Variables measuring financial commitment and resource allocation.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

### 3. Experience Variables
Variables measuring operational knowledge and learning.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

### 4. Governance Variables
Variables measuring institutional efficiency and coordination.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

### 5. External Factors
Variables measuring external pressures and constraints.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

### 6. Partnership Variables
Variables measuring cooperation and strategic alignment.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

### 7. Technology/Disruption Variables
Variables measuring potential technological changes.

| Variable | Baseline | Distribution | Confidence |
|----------|----------|--------------|------------|
| `variable_1` | | | |
| `variable_2` | | | |

---

## Engineered/Calculated Features

Document features derived from raw variables:

| Feature | Formula | Description |
|---------|---------|-------------|
| `feature_1` | `var_a / (var_b + 1)` | Ratio measuring... |
| `feature_2` | `0.4*x + 0.3*y + 0.3*z` | Composite index of... |

---

## MCDA Criteria Mapping

Map variables to MCDA criteria:

| Criterion | Weight | Contributing Variables |
|-----------|--------|----------------------|
| **Capability (α)** | 0.467 | var_1, var_2, var_3 |
| **Momentum (β)** | 0.277 | var_4, var_5 |
| **Feasibility (γ)** | 0.160 | var_6, var_7 |
| **Synergy (δ)** | 0.096 | var_8, var_9 |

---

## Key Formulas

Document your MCDA formulas:

### Composite Index
```
L = α×C + β×M + γ×F + δ×S
```

### Momentum Sub-Index
```
M = w1×investment + w2×growth + w3×resilience
```

### Feasibility Sub-Index
```
F = (timeline × governance) / (1 + ln(n) × friction)
```

### Synergy Sub-Index
```
S = complementarity × integration / conflict_index
```

---

## Temporal Dynamics

### Decay Rates
| Type | Rate (λ) | Description |
|------|----------|-------------|
| Active | 0.043 | Continuously utilized capabilities |
| Inactive | 0.14 | Reduced utilization |
| Unutilized | 0.30 | Dormant capabilities |

### Decay Formula
```
V(t) = V₀ × e^(-λt)
```

---

## Data Quality Assessment

### Source Triangulation
- Each value verified by ≥3 independent sources
- Discrepancies resolved using median values
- Uncertainty ranges documented

### Confidence Levels
| Level | Description | Uncertainty Multiplier |
|-------|-------------|----------------------|
| High | Multiple consistent sources | 0.5× |
| Medium | Some source variation | 1.0× |
| Low | Limited/conflicting sources | 1.5× |
| Very Low | Expert judgment only | 2.0× |

---

## Notes and Limitations

Document important caveats:

1. **Data limitations**: [Describe any data gaps]
2. **Assumption sensitivity**: [Note which assumptions most affect results]
3. **Temporal validity**: [When projections become less reliable]
4. **Scope boundaries**: [What the model does/doesn't capture]

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | YYYY-MM-DD | Initial release |

---

## References

List all data sources and citations used for variable estimation.

1. [Citation 1]
2. [Citation 2]
3. ...
