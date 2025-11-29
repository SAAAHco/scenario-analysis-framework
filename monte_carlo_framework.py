# -*- coding: utf-8 -*-
"""
Monte Carlo Analysis Framework for Multi-Criteria Decision Analysis
====================================================================

A generalized framework for uncertainty quantification in scenario-based
geopolitical or strategic analysis using Monte Carlo simulation.

This module provides:
- Configurable MCDA composite index calculation
- Parameter uncertainty propagation
- Statistical analysis and confidence intervals
- Publication-ready visualizations

Users should customize:
1. SCENARIOS: Define your scenario names
2. MCDA_WEIGHTS: Set criteria weights (via AHP methodology)
3. PARAMETER_CONFIG: Define your variables and distributions

Reference:
    [Your citation here]

Author: [Your name]
Version: 1.0
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# CONFIGURATION - CUSTOMIZE FOR YOUR ANALYSIS
# =============================================================================

class MCDAConfig:
    """
    Configuration for Multi-Criteria Decision Analysis.
    
    CUSTOMIZE THIS CLASS for your research domain.
    """
    
    # Define scenarios
    SCENARIOS = ['Scenario A', 'Scenario B', 'Scenario C', 'Scenario D']
    
    # MCDA criterion weights (derived via AHP - must sum to 1.0)
    WEIGHTS = {
        'alpha': 0.467,   # Capability criterion
        'beta': 0.277,    # Momentum criterion
        'gamma': 0.160,   # Feasibility criterion
        'delta': 0.096    # Synergy criterion
    }
    
    # Monte Carlo parameters
    N_ITERATIONS = 10000
    RANDOM_SEED = 42
    CONFIDENCE_LEVEL = 0.95


class ParameterConfig:
    """
    Parameter configuration with uncertainty ranges.
    
    CUSTOMIZE THIS CLASS with your domain-specific parameters.
    
    Each parameter should specify:
        - base: Central/expected value
        - uncertainty: Relative uncertainty range (as fraction)
        - distribution: 'normal', 'uniform', 'beta', 'triangular'
        - confidence: 'high', 'medium', 'low' (affects uncertainty spread)
    """
    
    # Example parameters - REPLACE with your domain variables
    PARAMETERS = {
        # Scenario A parameters
        'scenario_a': {
            'capability': {'base': 0.85, 'uncertainty': 0.05, 'distribution': 'normal', 'confidence': 'high'},
            'investment': {'base': 0.60, 'uncertainty': 0.10, 'distribution': 'normal', 'confidence': 'medium'},
            'growth': {'base': 0.75, 'uncertainty': 0.20, 'distribution': 'uniform', 'confidence': 'medium'},
            'resilience': {'base': 0.65, 'uncertainty': 0.15, 'distribution': 'uniform', 'confidence': 'low'},
            'timeline': {'base': 0.85, 'uncertainty': 0.15, 'distribution': 'triangular', 'confidence': 'medium'},
            'governance': {'base': 0.75, 'uncertainty': 0.10, 'distribution': 'normal', 'confidence': 'medium'},
            'friction': {'base': 0.10, 'uncertainty': 0.20, 'distribution': 'uniform', 'confidence': 'low'},
            'synergy': {'base': 0.80, 'uncertainty': 0.10, 'distribution': 'uniform', 'confidence': 'medium'},
        },
        
        # Scenario B parameters
        'scenario_b': {
            'capability': {'base': 1.00, 'uncertainty': 0.05, 'distribution': 'normal', 'confidence': 'high'},
            'momentum': {'base': 0.90, 'uncertainty': 0.10, 'distribution': 'normal', 'confidence': 'medium'},
            'timeline': {'base': 0.45, 'uncertainty': 0.10, 'distribution': 'triangular', 'confidence': 'medium'},
            'governance': {'base': 0.60, 'uncertainty': 0.10, 'distribution': 'normal', 'confidence': 'medium'},
            'friction': {'base': 0.30, 'uncertainty': 0.15, 'distribution': 'uniform', 'confidence': 'low'},
            'synergy': {'base': 0.80, 'uncertainty': 0.10, 'distribution': 'uniform', 'confidence': 'low'},
        },
        
        # Scenario C parameters
        'scenario_c': {
            'capability': {'base': 0.15, 'uncertainty': 0.15, 'distribution': 'normal', 'confidence': 'medium'},
            'momentum': {'base': 0.15, 'uncertainty': 0.20, 'distribution': 'uniform', 'confidence': 'low'},
            'feasibility': {'base': 0.05, 'uncertainty': 0.30, 'distribution': 'uniform', 'confidence': 'low'},
            'synergy': {'base': 0.10, 'uncertainty': 0.50, 'distribution': 'uniform', 'confidence': 'low'},
        },
        
        # Scenario D parameters
        'scenario_d': {
            'base_index': {'base': 0.25, 'uncertainty': 0.20, 'distribution': 'uniform', 'confidence': 'low'},
        },
    }
    
    @classmethod
    def get_uncertainty_multiplier(cls, confidence):
        """Get uncertainty multiplier based on confidence level."""
        multipliers = {
            'high': 0.5,      # Narrow uncertainty
            'medium': 1.0,    # Standard uncertainty
            'low': 1.5,       # Wide uncertainty
            'very_low': 2.0   # Very wide uncertainty
        }
        return multipliers.get(confidence, 1.0)


# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

class MonteCarloMCDA:
    """
    Monte Carlo simulation engine for MCDA-based scenario analysis.
    """
    
    def __init__(self, config=MCDAConfig, params=ParameterConfig):
        """
        Initialize Monte Carlo engine.
        
        Args:
            config: MCDA configuration class
            params: Parameter configuration class
        """
        self.config = config
        self.params = params
        self.scenarios = config.SCENARIOS
        self.weights = config.WEIGHTS
        self.results = None
        
        # Validate weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    def sample_parameter(self, param_config):
        """
        Sample a parameter value based on its configuration.
        
        Args:
            param_config: Dictionary with base, uncertainty, distribution, confidence
            
        Returns:
            Sampled value
        """
        base = param_config['base']
        uncertainty = param_config['uncertainty']
        dist = param_config['distribution']
        confidence = param_config.get('confidence', 'medium')
        
        # Adjust uncertainty based on confidence
        multiplier = self.params.get_uncertainty_multiplier(confidence)
        uncertainty = uncertainty * multiplier
        
        # Calculate bounds
        low = base * (1 - uncertainty)
        high = base * (1 + uncertainty)
        
        if dist == 'normal':
            # Normal distribution centered on base value
            std = (high - low) / 4  # 95% within range
            return np.random.normal(base, std)
        
        elif dist == 'uniform':
            return np.random.uniform(low, high)
        
        elif dist == 'beta':
            # Scale beta to desired range
            alpha = param_config.get('alpha', 2)
            beta_param = param_config.get('beta', 2)
            sample = np.random.beta(alpha, beta_param)
            return low + sample * (high - low)
        
        elif dist == 'triangular':
            return np.random.triangular(low, base, high)
        
        else:
            return np.random.uniform(low, high)
    
    def calculate_scenario_index(self, scenario_key, scenario_params):
        """
        Calculate composite MCDA index for a scenario.
        
        CUSTOMIZE THIS METHOD for your scenario logic.
        
        Args:
            scenario_key: Scenario identifier
            scenario_params: Sampled parameters for this iteration
            
        Returns:
            Composite index value
        """
        w = self.weights
        
        if scenario_key == 'scenario_a':
            # Scenario A: Partnership scenario
            C = scenario_params.get('capability', 0.85)
            
            # Momentum: weighted combination
            M_inv = scenario_params.get('investment', 0.60)
            M_growth = scenario_params.get('growth', 0.75)
            M_res = scenario_params.get('resilience', 0.65)
            M = 0.4 * M_inv + 0.35 * M_growth + 0.25 * M_res
            
            # Feasibility: governance adjusted by friction
            Ti = scenario_params.get('timeline', 0.85)
            Gi = scenario_params.get('governance', 0.75)
            sigma = scenario_params.get('friction', 0.10)
            n_entities = 2  # Number of coordinating entities
            F = (Ti * Gi) / (1 + np.log(n_entities) * sigma)
            
            # Synergy
            S = scenario_params.get('synergy', 0.80)
            
            L = w['alpha']*C + w['beta']*M + w['gamma']*F + w['delta']*S
            
        elif scenario_key == 'scenario_b':
            # Scenario B: Dominance scenario
            C = scenario_params.get('capability', 1.00)
            M = scenario_params.get('momentum', 0.90)
            
            Ti = scenario_params.get('timeline', 0.45)
            Gi = scenario_params.get('governance', 0.60)
            sigma = scenario_params.get('friction', 0.30)
            F = (Ti * Gi) / (1 + np.log(2) * sigma)
            
            S = scenario_params.get('synergy', 0.80)
            
            L = w['alpha']*C + w['beta']*M + w['gamma']*F + w['delta']*S
            
        elif scenario_key == 'scenario_c':
            # Scenario C: Challenge scenario
            C = scenario_params.get('capability', 0.15)
            M = scenario_params.get('momentum', 0.15)
            F = scenario_params.get('feasibility', 0.05)
            S = scenario_params.get('synergy', 0.10)
            
            L = w['alpha']*C + w['beta']*M + w['gamma']*F + w['delta']*S
            
        elif scenario_key == 'scenario_d':
            # Scenario D: Fragmentation
            L = scenario_params.get('base_index', 0.25)
            
        else:
            L = 0.1  # Default fallback
        
        return L
    
    def run_simulation(self, n_iterations=None):
        """
        Run Monte Carlo simulation.
        
        Args:
            n_iterations: Number of iterations (default from config)
            
        Returns:
            Dictionary with results for each scenario
        """
        if n_iterations is None:
            n_iterations = self.config.N_ITERATIONS
        
        np.random.seed(self.config.RANDOM_SEED)
        
        # Initialize results storage
        results = {scenario: [] for scenario in self.scenarios}
        
        # Map scenario names to parameter keys
        scenario_keys = ['scenario_a', 'scenario_b', 'scenario_c', 'scenario_d']
        
        for iteration in range(n_iterations):
            if iteration % (n_iterations // 10) == 0:
                print(f"  Progress: {iteration}/{n_iterations}", end='\r')
            
            # Sample all parameters
            sampled_indices = []
            
            for i, (scenario, param_key) in enumerate(zip(self.scenarios, scenario_keys)):
                scenario_params = {}
                
                if param_key in self.params.PARAMETERS:
                    for param_name, param_config in self.params.PARAMETERS[param_key].items():
                        scenario_params[param_name] = self.sample_parameter(param_config)
                
                # Calculate scenario index
                L = self.calculate_scenario_index(param_key, scenario_params)
                sampled_indices.append(max(L, 0.001))  # Ensure positive
            
            # Normalize to probabilities
            total = sum(sampled_indices)
            for i, scenario in enumerate(self.scenarios):
                results[scenario].append(sampled_indices[i] / total)
        
        print(f"\n✓ Completed {n_iterations} iterations")
        
        self.results = results
        return results
    
    def analyze_results(self):
        """
        Calculate comprehensive statistics from simulation results.
        
        Returns:
            Dictionary with statistics for each scenario
        """
        if self.results is None:
            raise ValueError("Run simulation first")
        
        stats_summary = {}
        
        for scenario, values in self.results.items():
            values = np.array(values) * 100  # Convert to percentages
            
            stats_summary[scenario] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'var': np.var(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'ci_90_lower': np.percentile(values, 5),
                'ci_90_upper': np.percentile(values, 95),
                'min': np.min(values),
                'max': np.max(values),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            }
        
        return stats_summary
    
    def statistical_tests(self):
        """
        Perform statistical hypothesis tests between scenarios.
        
        Returns:
            Dictionary with test results
        """
        if self.results is None:
            raise ValueError("Run simulation first")
        
        tests = {}
        scenarios = list(self.results.keys())
        
        # Pairwise comparisons
        for i in range(len(scenarios)):
            for j in range(i+1, len(scenarios)):
                s1, s2 = scenarios[i], scenarios[j]
                v1, v2 = np.array(self.results[s1]), np.array(self.results[s2])
                
                # T-test
                t_stat, t_pval = ttest_ind(v1, v2)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_pval = mannwhitneyu(v1, v2, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
                cohens_d = (v1.mean() - v2.mean()) / pooled_std
                
                tests[f"{s1} vs {s2}"] = {
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pval,
                    'cohens_d': cohens_d,
                    'significant_005': t_pval < 0.05,
                    'significant_001': t_pval < 0.01,
                }
        
        return tests


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_distributions(mc_engine, figsize=(14, 10), save_path=None):
    """
    Create comprehensive distribution visualization.
    
    Args:
        mc_engine: MonteCarloMCDA instance with results
        figsize: Figure size tuple
        save_path: Path to save figure (optional)
    """
    results = mc_engine.results
    stats_summary = mc_engine.analyze_results()
    
    n_scenarios = len(mc_engine.scenarios)
    n_cols = 2
    n_rows = (n_scenarios + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_scenarios))
    
    for i, (scenario, values) in enumerate(results.items()):
        ax = axes[i]
        values_pct = np.array(values) * 100
        stats_s = stats_summary[scenario]
        
        # Histogram with KDE
        ax.hist(values_pct, bins=50, density=True, alpha=0.7, 
                color=colors[i], edgecolor='black', linewidth=0.5)
        
        # KDE overlay
        kde = stats.gaussian_kde(values_pct)
        x_range = np.linspace(values_pct.min(), values_pct.max(), 200)
        ax.plot(x_range, kde(x_range), 'k-', linewidth=2)
        
        # Mean and CI lines
        ax.axvline(stats_s['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {stats_s['mean']:.1f}%")
        ax.axvline(stats_s['ci_lower'], color='darkred', linestyle=':', linewidth=1.5)
        ax.axvline(stats_s['ci_upper'], color='darkred', linestyle=':', linewidth=1.5)
        
        # Fill CI region
        ax.axvspan(stats_s['ci_lower'], stats_s['ci_upper'], 
                   alpha=0.2, color='red', label='95% CI')
        
        ax.set_title(f"{scenario}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability (%)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Monte Carlo Simulation Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_summary_bar(mc_engine, figsize=(10, 6), save_path=None):
    """
    Create summary bar chart with confidence intervals.
    """
    stats_summary = mc_engine.analyze_results()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scenarios = list(stats_summary.keys())
    means = [stats_summary[s]['mean'] for s in scenarios]
    ci_lower = [stats_summary[s]['ci_lower'] for s in scenarios]
    ci_upper = [stats_summary[s]['ci_upper'] for s in scenarios]
    
    # Calculate error bars
    errors_lower = [m - l for m, l in zip(means, ci_lower)]
    errors_upper = [u - m for m, u in zip(means, ci_upper)]
    
    x_pos = np.arange(len(scenarios))
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
    
    bars = ax.bar(x_pos, means, yerr=[errors_lower, errors_upper],
                  capsize=8, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors_upper)*0.1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Scenario Probabilities with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(ci_upper) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_boxplot_comparison(mc_engine, figsize=(10, 6), save_path=None):
    """
    Create box plot comparison of scenarios.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    data = [np.array(mc_engine.results[s]) * 100 for s in mc_engine.scenarios]
    
    bp = ax.boxplot(data, patch_artist=True, labels=mc_engine.scenarios)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(mc_engine.scenarios)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Scenario Probability Distributions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# RESULTS REPORTING
# =============================================================================

def print_results_table(mc_engine):
    """Print formatted results table."""
    stats_summary = mc_engine.analyze_results()
    
    print("\n" + "="*70)
    print(" MONTE CARLO RESULTS ")
    print("="*70)
    
    print(f"\n{'Scenario':<20} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
    print("-"*60)
    
    for scenario, stats_s in stats_summary.items():
        ci_str = f"[{stats_s['ci_lower']:.1f}%, {stats_s['ci_upper']:.1f}%]"
        print(f"{scenario:<20} {stats_s['mean']:>6.1f}%    {stats_s['std']:>5.1f}%    {ci_str}")
    
    print("\n" + "-"*60)
    print("Detailed Statistics:")
    print("-"*60)
    
    df = pd.DataFrame(stats_summary).T
    df = df.round(2)
    print(df[['mean', 'median', 'std', 'ci_lower', 'ci_upper', 'skewness', 'kurtosis']])


def export_results(mc_engine, output_path='mc_results'):
    """Export results to CSV files."""
    import os
    
    os.makedirs(output_path, exist_ok=True)
    
    # Raw results
    df_raw = pd.DataFrame(mc_engine.results)
    df_raw.to_csv(os.path.join(output_path, 'raw_results.csv'), index=False)
    
    # Summary statistics
    stats_summary = mc_engine.analyze_results()
    df_stats = pd.DataFrame(stats_summary).T
    df_stats.to_csv(os.path.join(output_path, 'statistics.csv'))
    
    # Statistical tests
    tests = mc_engine.statistical_tests()
    df_tests = pd.DataFrame(tests).T
    df_tests.to_csv(os.path.join(output_path, 'statistical_tests.csv'))
    
    print(f"✓ Results exported to: {output_path}/")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" MONTE CARLO MCDA ANALYSIS FRAMEWORK ")
    print("="*70)
    
    # Initialize engine
    mc_engine = MonteCarloMCDA()
    
    # Run simulation
    print("\nRunning Monte Carlo Simulation...")
    results = mc_engine.run_simulation()
    
    # Print results
    print_results_table(mc_engine)
    
    # Statistical tests
    print("\n" + "="*70)
    print(" STATISTICAL TESTS ")
    print("="*70)
    
    tests = mc_engine.statistical_tests()
    for comparison, test_results in tests.items():
        sig = "***" if test_results['significant_001'] else ("**" if test_results['significant_005'] else "")
        print(f"\n{comparison}: t={test_results['t_statistic']:.3f}, p={test_results['t_pvalue']:.4f} {sig}")
        print(f"  Cohen's d: {test_results['cohens_d']:.3f}")
    
    # Generate visualizations
    print("\n" + "-"*60)
    print(" GENERATING VISUALIZATIONS ")
    print("-"*60)
    
    plot_distributions(mc_engine, save_path='mc_distributions.png')
    plot_summary_bar(mc_engine, save_path='mc_summary.png')
    plot_boxplot_comparison(mc_engine, save_path='mc_boxplot.png')
    
    print("✓ Visualizations saved")
    
    # Export results
    export_results(mc_engine)
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE ")
    print("="*70)
    
    return mc_engine


if __name__ == "__main__":
    mc_engine = main()
