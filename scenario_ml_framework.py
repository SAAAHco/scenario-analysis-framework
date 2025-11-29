# -*- coding: utf-8 -*-
"""
Scenario-Based Geopolitical Analysis: Machine Learning Ensemble Framework
=========================================================================

A generalized framework for multi-scenario geopolitical prediction using
ensemble machine learning methods and Monte Carlo simulation.

This framework implements:
- Multi-Criteria Decision Analysis (MCDA) with AHP-derived weights
- Ensemble of 6 machine learning algorithms
- Monte Carlo uncertainty quantification
- Comprehensive visualization suite

Users should customize:
1. SCENARIOS: Define your scenario names
2. BASELINE_CONFIG: Set your domain-specific baseline parameters
3. MCDA_WEIGHTS: Adjust criteria weights via AHP methodology
4. DATA_GENERATION_CONFIG: Configure variable distributions

Reference:
    [Your citation here]

Author: [Your name]
Version: 1.0
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Optional imports - graceful degradation if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - using fallback algorithms")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - using fallback algorithms")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available - using fallback algorithms")

try:
    from docx import Document
    from docx.shared import Inches, Pt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("python-docx not available - Word reports disabled")

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION SECTION - CUSTOMIZE FOR YOUR ANALYSIS
# =============================================================================

class ScenarioConfig:
    """
    Configuration class for scenario-based analysis.
    
    CUSTOMIZE THIS CLASS for your specific domain/research question.
    """
    
    # Define your scenarios (2-6 scenarios recommended)
    SCENARIOS = [
        'Scenario A',
        'Scenario B',
        'Scenario C',
        'Scenario D'
    ]
    
    # Color scheme for visualizations
    COLORS = {
        'Scenario A': '#FF6B6B',
        'Scenario B': '#4ECDC4',
        'Scenario C': '#45B7D1',
        'Scenario D': '#96CEB4'
    }
    
    # MCDA Weights (must sum to 1.0)
    # Derive these using Analytic Hierarchy Process (AHP)
    MCDA_WEIGHTS = {
        'capability': 0.467,    # Infrastructure/capability criterion
        'momentum': 0.277,      # Growth/momentum criterion
        'feasibility': 0.160,   # Implementation feasibility criterion
        'synergy': 0.096        # Partnership/synergy criterion
    }
    
    # Temporal parameters
    TIME_HORIZON = 10  # Years for projection
    BASELINE_YEAR = 2025
    TARGET_YEAR = 2035
    
    # Decay rates for capability persistence (from organizational learning theory)
    DECAY_RATES = {
        'active': 0.043,      # Active utilization
        'inactive': 0.14,     # Reduced activity
        'unutilized': 0.30    # Dormant capabilities
    }
    
    # Monte Carlo parameters
    MC_ITERATIONS = 10000
    CONFIDENCE_LEVEL = 0.95
    
    # ML parameters
    N_SAMPLES = 5000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        weight_sum = sum(cls.MCDA_WEIGHTS.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"MCDA weights must sum to 1.0, got {weight_sum}")
        if len(cls.SCENARIOS) < 2:
            raise ValueError("At least 2 scenarios required")
        print("✓ Configuration validated")


class BaselineParameters:
    """
    Baseline parameters for your analysis domain.
    
    CUSTOMIZE THIS CLASS with your domain-specific variables.
    
    Structure each variable as:
        'variable_name': {
            'baseline': float,      # Current/baseline value
            'low': float,           # Conservative projection
            'base': float,          # Central projection  
            'high': float,          # Optimistic projection
            'distribution': str,    # 'normal', 'uniform', 'beta', 'lognormal'
            'params': dict          # Distribution parameters (mean, std, etc.)
        }
    """
    
    # Example structure - REPLACE with your domain variables
    VARIABLES = {
        # Actor A Capabilities
        'actor_a_capability': {
            'baseline': 40,
            'low': 35, 'base': 45, 'high': 55,
            'distribution': 'normal',
            'params': {'mean': 40, 'std': 5}
        },
        'actor_a_investment': {
            'baseline': 15.0,
            'low': 10, 'base': 20, 'high': 35,
            'distribution': 'lognormal',
            'params': {'mean': 15, 'std': 3}
        },
        'actor_a_experience': {
            'baseline': 30,
            'low': 25, 'base': 35, 'high': 45,
            'distribution': 'normal',
            'params': {'mean': 30, 'std': 5}
        },
        
        # Actor B Capabilities
        'actor_b_capability': {
            'baseline': 10,
            'low': 8, 'base': 15, 'high': 25,
            'distribution': 'normal',
            'params': {'mean': 10, 'std': 2}
        },
        'actor_b_investment': {
            'baseline': 20.0,
            'low': 15, 'base': 30, 'high': 50,
            'distribution': 'lognormal',
            'params': {'mean': 20, 'std': 4}
        },
        
        # Actor C Capabilities
        'actor_c_capability': {
            'baseline': 5,
            'low': 3, 'base': 10, 'high': 20,
            'distribution': 'normal',
            'params': {'mean': 5, 'std': 2}
        },
        'actor_c_investment': {
            'baseline': 5.0,
            'low': 0, 'base': 20, 'high': 50,
            'distribution': 'uniform',
            'params': {'low': 0, 'high': 30}
        },
        
        # Governance/Institutional factors
        'governance_efficiency': {
            'baseline': 0.7,
            'low': 0.4, 'base': 0.7, 'high': 0.9,
            'distribution': 'uniform',
            'params': {'low': 0.3, 'high': 0.9}
        },
        'coordination_friction': {
            'baseline': 1.0,
            'low': 0.5, 'base': 1.0, 'high': 1.5,
            'distribution': 'uniform',
            'params': {'low': 0.2, 'high': 1.8}
        },
        
        # External factors
        'external_pressure': {
            'baseline': 0.25,
            'low': 0.1, 'base': 0.25, 'high': 0.5,
            'distribution': 'beta',
            'params': {'alpha': 2, 'beta': 5}
        },
        'growth_rate': {
            'baseline': 0.15,
            'low': 0.08, 'base': 0.15, 'high': 0.25,
            'distribution': 'normal',
            'params': {'mean': 0.15, 'std': 0.03}
        },
        
        # Partnership factors
        'cooperation_level': {
            'baseline': 0.75,
            'low': 0.5, 'base': 0.75, 'high': 0.95,
            'distribution': 'beta',
            'params': {'alpha': 4, 'beta': 2}
        },
        'strategic_alignment': {
            'baseline': 0.80,
            'low': 0.6, 'base': 0.8, 'high': 0.95,
            'distribution': 'uniform',
            'params': {'low': 0.5, 'high': 0.95}
        },
        
        # Technology/disruption
        'tech_disruption_prob': {
            'baseline': 0.10,
            'low': 0.05, 'base': 0.12, 'high': 0.25,
            'distribution': 'uniform',
            'params': {'low': 0, 'high': 0.25}
        },
        'tech_impact_multiplier': {
            'baseline': 1.2,
            'low': 0.8, 'base': 1.2, 'high': 2.0,
            'distribution': 'lognormal',
            'params': {'mean': 1.2, 'std': 0.3}
        },
    }
    
    @classmethod
    def get_baseline_dict(cls):
        """Return simple baseline dictionary"""
        return {k: v['baseline'] for k, v in cls.VARIABLES.items()}


# =============================================================================
# CORE FRAMEWORK CLASSES
# =============================================================================

class ScenarioPredictor:
    """
    Multi-scenario prediction using ensemble machine learning.
    
    This class provides a general framework for:
    1. Generating synthetic training data based on theoretical parameter spaces
    2. Training an ensemble of ML algorithms
    3. Predicting scenario probabilities
    4. Quantifying uncertainty via Monte Carlo simulation
    """
    
    def __init__(self, config=ScenarioConfig, baseline=BaselineParameters, output_dir="Analysis_Output"):
        """
        Initialize the predictor.
        
        Args:
            config: Configuration class with scenario definitions
            baseline: Baseline parameters class
            output_dir: Directory for output files
        """
        self.config = config
        self.baseline = baseline
        self.scenarios = config.SCENARIOS
        self.colors = config.COLORS
        self.weights = config.MCDA_WEIGHTS
        
        # Validate configuration
        config.validate()
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
        # Model storage
        self.models = {}
        self.ensemble_weights = None
        self.feature_names = []
        
        # Setup output directory
        self.output_dir = output_dir
        self._setup_output_directory()
        
    def _setup_output_directory(self):
        """Create output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.output_dir}_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.data_dir = os.path.join(self.output_dir, "data")
        
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
    
    def generate_sample(self, n_samples=1):
        """
        Generate samples from configured distributions.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sampled values
        """
        data = {}
        
        for var_name, var_config in self.baseline.VARIABLES.items():
            dist = var_config['distribution']
            params = var_config['params']
            
            if dist == 'normal':
                data[var_name] = np.random.normal(params['mean'], params['std'], n_samples)
            elif dist == 'uniform':
                data[var_name] = np.random.uniform(params['low'], params['high'], n_samples)
            elif dist == 'beta':
                data[var_name] = np.random.beta(params['alpha'], params['beta'], n_samples)
            elif dist == 'lognormal':
                # Convert mean/std to lognormal parameters
                mean, std = params['mean'], params['std']
                sigma = np.sqrt(np.log(1 + (std/mean)**2))
                mu = np.log(mean) - sigma**2/2
                data[var_name] = np.random.lognormal(mu, sigma, n_samples)
            else:
                # Default to normal
                data[var_name] = np.random.normal(var_config['baseline'], 
                                                   var_config['baseline']*0.1, n_samples)
        
        return pd.DataFrame(data)
    
    def create_engineered_features(self, df):
        """
        Create engineered features from raw variables.
        
        CUSTOMIZE THIS METHOD for your domain-specific feature engineering.
        
        Args:
            df: DataFrame with raw variables
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        # Example engineered features - CUSTOMIZE for your domain
        
        # Capability ratios
        df['capability_ratio_ab'] = df['actor_a_capability'] / (df['actor_b_capability'] + 1)
        df['capability_ratio_ac'] = df['actor_a_capability'] / (df['actor_c_capability'] + 1)
        
        # Investment aggregates
        df['total_investment'] = (df['actor_a_investment'] + 
                                  df['actor_b_investment'] + 
                                  df['actor_c_investment'])
        df['investment_concentration'] = (df['actor_a_investment'] + df['actor_b_investment']) / df['total_investment']
        
        # Momentum composite
        df['momentum_index'] = (0.4 * df['growth_rate'] / df['growth_rate'].max() +
                                0.3 * df['actor_a_investment'] / df['actor_a_investment'].max() +
                                0.3 * df['cooperation_level'])
        
        # Feasibility composite
        df['feasibility_index'] = (df['governance_efficiency'] / 
                                   (1 + np.log(2) * df['coordination_friction']))
        
        # Synergy composite
        df['synergy_index'] = df['cooperation_level'] * df['strategic_alignment']
        
        # Technology disruption potential
        df['tech_disruption'] = df['tech_disruption_prob'] * df['tech_impact_multiplier']
        
        # Temporal decay factor
        df['decay_factor'] = np.exp(-self.config.DECAY_RATES['active'] * self.config.TIME_HORIZON)
        
        # External pressure resilience
        df['resilience'] = 1 - df['external_pressure']
        
        return df
    
    def generate_scenario_probabilities(self, df):
        """
        Generate scenario probabilities based on MCDA framework.
        
        CUSTOMIZE THIS METHOD for your scenario logic.
        
        Args:
            df: DataFrame with features
            
        Returns:
            numpy array of shape (n_samples, n_scenarios)
        """
        n_samples = len(df)
        n_scenarios = len(self.scenarios)
        probabilities = np.zeros((n_samples, n_scenarios))
        
        w = self.weights
        
        for i in range(n_samples):
            row = df.iloc[i]
            
            # Example scenario probability calculations - CUSTOMIZE for your domain
            
            # Scenario A: High Actor A + Actor B cooperation
            scenario_a = (
                w['capability'] * (row['actor_a_capability'] / 50) * 1.2 +
                w['momentum'] * row['momentum_index'] +
                w['feasibility'] * row['feasibility_index'] +
                w['synergy'] * row['synergy_index'] * 1.3
            ) * row['cooperation_level']
            
            # Scenario B: Actor A dominance
            scenario_b = (
                w['capability'] * (row['actor_a_capability'] / 50) +
                w['momentum'] * (row['actor_a_investment'] / 30) * row['resilience'] +
                w['feasibility'] * row['governance_efficiency'] +
                w['synergy'] * 0.1
            ) * row['resilience']
            
            # Scenario C: Fragmented competition
            scenario_c = (
                0.15 * row['coordination_friction'] +
                0.10 * (1 - row['governance_efficiency']) +
                0.05 * (1 - row['cooperation_level'])
            ) * 2
            
            # Scenario D: Actor C rise
            scenario_d = (
                w['capability'] * (row['actor_c_capability'] / 30) +
                w['momentum'] * (row['actor_c_investment'] / 50) +
                w['feasibility'] * 0.5 +
                w['synergy'] * 0.2
            ) * (1 + row['tech_disruption'])
            
            # Apply temporal evolution
            time_factor = self.config.TIME_HORIZON
            decay_active = self.config.DECAY_RATES['active']
            decay_inactive = self.config.DECAY_RATES['inactive']
            
            scenario_a *= np.exp(-decay_active * time_factor) * (1 + 0.20 * time_factor)
            scenario_b *= np.exp(-decay_active * time_factor) * (1 + 0.15 * time_factor)
            scenario_c *= np.exp(-decay_inactive * time_factor) * (1 + 0.02 * time_factor)
            scenario_d *= np.exp(-decay_inactive * time_factor) * (1 + 0.10 * time_factor)
            
            # Ensure positive values
            raw_probs = np.array([
                max(scenario_a, 0.01),
                max(scenario_b, 0.01),
                max(scenario_c, 0.01),
                max(scenario_d, 0.01)
            ])
            
            # Normalize to sum to 1
            probabilities[i] = raw_probs / raw_probs.sum()
        
        return probabilities
    
    def generate_training_data(self, n_samples=None):
        """
        Generate synthetic training data.
        
        Args:
            n_samples: Number of samples (default from config)
            
        Returns:
            Tuple of (features DataFrame, probabilities array)
        """
        if n_samples is None:
            n_samples = self.config.N_SAMPLES
            
        np.random.seed(self.config.RANDOM_STATE)
        
        # Generate raw samples
        df = self.generate_sample(n_samples)
        
        # Create engineered features
        df = self.create_engineered_features(df)
        
        # Generate target probabilities
        probabilities = self.generate_scenario_probabilities(df)
        
        self.feature_names = df.columns.tolist()
        
        print(f"✓ Generated {n_samples} training samples with {len(self.feature_names)} features")
        
        return df, probabilities
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Train ensemble of ML models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary of results and ensemble predictions
        """
        results = {}
        n_scenarios = len(self.scenarios)
        
        print("\n" + "="*60)
        print(" TRAINING ML ENSEMBLE ")
        print("="*60)
        
        # 1. XGBoost
        if XGBOOST_AVAILABLE:
            print("\n1. XGBoost...")
            xgb_models = []
            for i in range(n_scenarios):
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.config.RANDOM_STATE, verbosity=0
                )
                model.fit(X_train, y_train[:, i])
                xgb_models.append(model)
            
            xgb_pred = np.column_stack([m.predict(X_test) for m in xgb_models])
            xgb_pred = np.abs(xgb_pred) / np.sum(np.abs(xgb_pred), axis=1, keepdims=True)
            results['XGBoost'] = {'models': xgb_models, 'predictions': xgb_pred}
        
        # 2. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("2. LightGBM...")
            lgb_models = []
            for i in range(n_scenarios):
                model = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.config.RANDOM_STATE, verbosity=-1
                )
                model.fit(X_train, y_train[:, i])
                lgb_models.append(model)
            
            lgb_pred = np.column_stack([m.predict(X_test) for m in lgb_models])
            lgb_pred = np.abs(lgb_pred) / np.sum(np.abs(lgb_pred), axis=1, keepdims=True)
            results['LightGBM'] = {'models': lgb_models, 'predictions': lgb_pred}
        
        # 3. CatBoost
        if CATBOOST_AVAILABLE:
            print("3. CatBoost...")
            cat_models = []
            for i in range(n_scenarios):
                model = cb.CatBoostRegressor(
                    iterations=200, depth=6, learning_rate=0.1,
                    random_seed=self.config.RANDOM_STATE, verbose=False
                )
                model.fit(X_train, y_train[:, i])
                cat_models.append(model)
            
            cat_pred = np.column_stack([m.predict(X_test) for m in cat_models])
            cat_pred = np.abs(cat_pred) / np.sum(np.abs(cat_pred), axis=1, keepdims=True)
            results['CatBoost'] = {'models': cat_models, 'predictions': cat_pred}
        
        # 4. Random Forest
        print("4. Random Forest...")
        rf_models = []
        for i in range(n_scenarios):
            model = RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_split=10,
                random_state=self.config.RANDOM_STATE, n_jobs=-1
            )
            model.fit(X_train, y_train[:, i])
            rf_models.append(model)
        
        rf_pred = np.column_stack([m.predict(X_test) for m in rf_models])
        rf_pred = np.abs(rf_pred) / np.sum(np.abs(rf_pred), axis=1, keepdims=True)
        results['RandomForest'] = {'models': rf_models, 'predictions': rf_pred}
        
        # 5. Extra Trees
        print("5. Extra Trees...")
        et_models = []
        for i in range(n_scenarios):
            model = ExtraTreesRegressor(
                n_estimators=300, max_depth=12,
                random_state=self.config.RANDOM_STATE, n_jobs=-1
            )
            model.fit(X_train, y_train[:, i])
            et_models.append(model)
        
        et_pred = np.column_stack([m.predict(X_test) for m in et_models])
        et_pred = np.abs(et_pred) / np.sum(np.abs(et_pred), axis=1, keepdims=True)
        results['ExtraTrees'] = {'models': et_models, 'predictions': et_pred}
        
        # 6. Neural Network
        print("6. Neural Network...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu', solver='adam',
            learning_rate='adaptive', max_iter=2000,
            early_stopping=True, validation_fraction=0.1,
            random_state=self.config.RANDOM_STATE
        )
        nn_model.fit(X_train, y_train)
        nn_pred = nn_model.predict(X_test)
        nn_pred = np.abs(nn_pred) / np.sum(np.abs(nn_pred), axis=1, keepdims=True)
        results['NeuralNetwork'] = {'model': nn_model, 'predictions': nn_pred}
        
        # Create weighted ensemble
        print("\n7. Creating Weighted Ensemble...")
        all_preds = []
        weights = []
        
        for name, result in results.items():
            pred = result['predictions']
            mse = mean_squared_error(y_test, pred)
            weight = 1 / (mse + 1e-6)
            weights.append(weight)
            all_preds.append(pred)
            print(f"   {name}: MSE = {mse:.4f}, Weight = {weight:.2f}")
        
        weights = np.array(weights) / np.sum(weights)
        
        ensemble_pred = np.zeros_like(y_test)
        for i, pred in enumerate(all_preds):
            ensemble_pred += pred * weights[i]
        ensemble_pred = ensemble_pred / ensemble_pred.sum(axis=1, keepdims=True)
        
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        print(f"\n✓ Ensemble MSE: {ensemble_mse:.4f}")
        
        self.models = results
        self.ensemble_weights = weights
        
        return results, ensemble_pred
    
    def predict_scenarios(self, features_df=None):
        """
        Predict scenario probabilities.
        
        Args:
            features_df: Features for prediction (uses baseline if None)
            
        Returns:
            Dictionary of scenario probabilities
        """
        if features_df is None:
            # Use baseline parameters
            baseline_dict = self.baseline.get_baseline_dict()
            features_df = pd.DataFrame([baseline_dict])
            features_df = self.create_engineered_features(features_df)
        
        X = self.scaler.transform(features_df)
        
        predictions = []
        for name, model_info in self.models.items():
            if name == 'NeuralNetwork':
                pred = model_info['model'].predict(X)[0]
            elif 'models' in model_info:
                pred = np.array([m.predict(X)[0] for m in model_info['models']])
            else:
                continue
            
            pred = np.abs(pred) / np.sum(np.abs(pred))
            predictions.append(pred)
        
        # Weighted ensemble prediction
        final_pred = np.zeros(len(self.scenarios))
        for i, pred in enumerate(predictions):
            if i < len(self.ensemble_weights):
                final_pred += pred * self.ensemble_weights[i]
        
        final_pred = final_pred / final_pred.sum()
        
        return {scenario: prob for scenario, prob in zip(self.scenarios, final_pred)}
    
    def run_monte_carlo(self, n_iterations=None):
        """
        Run Monte Carlo simulation for uncertainty quantification.
        
        Args:
            n_iterations: Number of iterations (default from config)
            
        Returns:
            numpy array of shape (n_iterations, n_scenarios)
        """
        if n_iterations is None:
            n_iterations = self.config.MC_ITERATIONS
            
        print(f"\n" + "="*60)
        print(f" MONTE CARLO SIMULATION ({n_iterations} iterations) ")
        print("="*60)
        
        mc_results = []
        
        for iteration in range(n_iterations):
            if iteration % (n_iterations // 10) == 0:
                print(f"  Progress: {iteration}/{n_iterations}", end='\r')
            
            # Generate single sample
            sample_df = self.generate_sample(n_samples=1)
            sample_df = self.create_engineered_features(sample_df)
            X_sample = self.scaler.transform(sample_df)
            
            # Get predictions from all models
            preds = []
            for name, model_info in self.models.items():
                try:
                    if name == 'NeuralNetwork':
                        pred = model_info['model'].predict(X_sample)[0]
                    elif 'models' in model_info:
                        pred = np.array([m.predict(X_sample)[0] for m in model_info['models']])
                    else:
                        continue
                    pred = np.abs(pred) / np.sum(np.abs(pred))
                    preds.append(pred)
                except:
                    continue
            
            if preds:
                mc_results.append(np.mean(preds, axis=0))
        
        print(f"\n✓ Completed {n_iterations} iterations")
        
        mc_results = np.array(mc_results)
        
        # Calculate statistics
        means = mc_results.mean(axis=0) * 100
        stds = mc_results.std(axis=0) * 100
        ci_lower = np.percentile(mc_results, 2.5, axis=0) * 100
        ci_upper = np.percentile(mc_results, 97.5, axis=0) * 100
        
        print(f"\n{'Scenario':<25} {'Mean':<10} {'Std':<10} {'95% CI'}")
        print("-"*60)
        for i, scenario in enumerate(self.scenarios):
            print(f"{scenario:<25} {means[i]:>6.1f}%    {stds[i]:>5.1f}%    [{ci_lower[i]:>5.1f}%, {ci_upper[i]:>5.1f}%]")
        
        return mc_results
    
    def feature_importance_analysis(self, X_train, feature_names):
        """
        Analyze feature importance across models.
        
        Returns:
            Tuple of (importance array, feature names)
        """
        importance_list = []
        
        for name, model_info in self.models.items():
            if 'models' in model_info:
                for model in model_info['models']:
                    if hasattr(model, 'feature_importances_'):
                        importance_list.append(model.feature_importances_)
        
        if importance_list:
            avg_importance = np.mean(importance_list, axis=0)
            sorted_idx = np.argsort(avg_importance)[::-1]
            
            return avg_importance[sorted_idx], np.array(feature_names)[sorted_idx]
        
        return None, None
    
    def save_results(self, df, probabilities, predictions, mc_results):
        """Save all results to files."""
        # Save training data
        df.to_csv(os.path.join(self.data_dir, 'training_data.csv'), index=False)
        
        # Save predictions
        pd.DataFrame([predictions]).to_csv(
            os.path.join(self.data_dir, 'predictions.csv'), index=False)
        
        # Save Monte Carlo results
        pd.DataFrame(mc_results, columns=self.scenarios).to_csv(
            os.path.join(self.data_dir, 'monte_carlo_results.csv'), index=False)
        
        # Save configuration
        config_dict = {
            'scenarios': self.scenarios,
            'mcda_weights': self.weights,
            'time_horizon': self.config.TIME_HORIZON,
            'n_samples': self.config.N_SAMPLES,
            'mc_iterations': self.config.MC_ITERATIONS,
            'random_state': self.config.RANDOM_STATE
        }
        with open(os.path.join(self.data_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: {self.data_dir}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_scenario_comparison(predictor, predictions, save_path=None):
    """Plot scenario probability comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = list(predictions.keys())
    probs = [predictions[s] * 100 for s in scenarios]
    colors = [predictor.colors.get(s, '#888888') for s in scenarios]
    
    bars = ax.bar(scenarios, probs, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title(f'Scenario Probabilities ({predictor.config.TARGET_YEAR} Projection)', fontsize=14)
    ax.set_ylim(0, max(probs) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_monte_carlo_distributions(predictor, mc_results, save_path=None):
    """Plot Monte Carlo distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, scenario in enumerate(predictor.scenarios):
        ax = axes[i]
        data = mc_results[:, i] * 100
        
        color = predictor.colors.get(scenario, '#888888')
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black')
        
        mean = np.mean(data)
        ci_lower = np.percentile(data, 2.5)
        ci_upper = np.percentile(data, 97.5)
        
        ax.axvline(mean, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}%')
        ax.axvline(ci_lower, color='red', linestyle=':', linewidth=1.5)
        ax.axvline(ci_upper, color='red', linestyle=':', linewidth=1.5)
        
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.suptitle('Monte Carlo Simulation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(importance, names, top_n=15, save_path=None):
    """Plot feature importance."""
    if importance is None:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = min(top_n, len(importance))
    y_pos = np.arange(top_n)
    
    ax.barh(y_pos, importance[:top_n], color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[:top_n])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" SCENARIO-BASED GEOPOLITICAL ANALYSIS FRAMEWORK ")
    print(" Machine Learning Ensemble with Monte Carlo Uncertainty ")
    print("="*70)
    
    # Initialize predictor
    predictor = ScenarioPredictor()
    
    # Generate training data
    print("\n" + "-"*60)
    print(" GENERATING TRAINING DATA ")
    print("-"*60)
    df, probabilities = predictor.generate_training_data()
    
    # Prepare data for ML
    X = predictor.scaler.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, probabilities, 
        test_size=predictor.config.TEST_SIZE,
        random_state=predictor.config.RANDOM_STATE
    )
    
    # Train ensemble
    results, ensemble_pred = predictor.train_ensemble(X_train, y_train, X_test, y_test)
    
    # Make predictions
    print("\n" + "-"*60)
    print(f" {predictor.config.TARGET_YEAR} SCENARIO PREDICTIONS ")
    print("-"*60)
    predictions = predictor.predict_scenarios()
    
    for scenario, prob in predictions.items():
        print(f"  {scenario:<25} {prob*100:>6.2f}%")
    
    # Run Monte Carlo
    mc_results = predictor.run_monte_carlo()
    
    # Feature importance
    importance, names = predictor.feature_importance_analysis(X_train, predictor.feature_names)
    
    # Generate visualizations
    print("\n" + "-"*60)
    print(" GENERATING VISUALIZATIONS ")
    print("-"*60)
    
    plot_scenario_comparison(
        predictor, predictions,
        os.path.join(predictor.figures_dir, '01_scenario_comparison.png')
    )
    
    plot_monte_carlo_distributions(
        predictor, mc_results,
        os.path.join(predictor.figures_dir, '02_monte_carlo_distributions.png')
    )
    
    if importance is not None:
        plot_feature_importance(
            importance, names,
            save_path=os.path.join(predictor.figures_dir, '03_feature_importance.png')
        )
    
    # Save results
    predictor.save_results(df, probabilities, predictions, mc_results)
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE ")
    print("="*70)
    print(f"\n✓ All outputs saved to: {predictor.output_dir}")
    
    return predictor, predictions, mc_results


if __name__ == "__main__":
    predictor, predictions, mc_results = main()
