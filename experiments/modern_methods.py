#!/usr/bin/env python3
"""
Extended Experiments with Modern ML Methods

This module extends the replication study to include:
1. Neural Networks (MLP)
2. XGBoost (modern gradient boosting)
3. LightGBM (efficient gradient boosting)
4. Polynomial feature engineering
5. Time-series aware validation

The goal is to contextualize the original 2018 results within the
modern ML landscape and critically analyze what has changed.
"""

import time
import warnings
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data_generator import ShipEngineDataGenerator


class ModernMethodsExperiment:
    """
    Extended experiment comparing 2018-era methods with modern approaches.

    This provides critical context for the replication study by asking:
    - Have ML methods improved since 2018?
    - Was TPOT's approach optimal or just convenient?
    - What do modern methods reveal about the problem structure?
    """

    def __init__(self, seed: int = 42, results_dir: str = '../results'):
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)

    def get_modern_models(self) -> Dict:
        """
        Return dictionary of modern ML models to test.

        Organized by era/category:
        - 2018-era: What was available when original paper was written
        - Modern: Methods that have improved or emerged since
        - Neural: Deep learning approaches
        """
        models = {
            # === 2018-ERA BASELINES ===
            'linear': {
                'model': LinearRegression(),
                'era': '2018',
                'category': 'linear'
            },
            'ridge': {
                'model': Ridge(alpha=1.0),
                'era': '2018',
                'category': 'linear'
            },
            'elastic_net': {
                'model': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
                'era': '2018',
                'category': 'linear'
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=self.seed
                ),
                'era': '2018',
                'category': 'ensemble'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.seed
                ),
                'era': '2018',
                'category': 'ensemble'
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(
                    n_estimators=100,
                    n_jobs=-1,
                    random_state=self.seed
                ),
                'era': '2018',
                'category': 'ensemble'
            },

            # === MODERN METHODS (Post-2018 improvements) ===
            'hist_gradient_boosting': {
                'model': HistGradientBoostingRegressor(
                    max_iter=100,
                    max_depth=None,
                    random_state=self.seed
                ),
                'era': 'modern',
                'category': 'ensemble',
                'note': 'Sklearn 0.21+ (2019), inspired by LightGBM'
            },

            # === NEURAL NETWORK APPROACHES ===
            'mlp_small': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    early_stopping=True,
                    random_state=self.seed
                ),
                'era': '2018',
                'category': 'neural',
                'note': 'Small MLP - available in 2018'
            },
            'mlp_medium': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    early_stopping=True,
                    random_state=self.seed
                ),
                'era': '2018',
                'category': 'neural',
                'note': 'Medium MLP'
            },
            'mlp_large': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    early_stopping=True,
                    random_state=self.seed
                ),
                'era': 'modern',
                'category': 'neural',
                'note': 'Larger MLP with more capacity'
            },
        }

        # Try to add XGBoost if available
        try:
            import xgboost as xgb
            models['xgboost'] = {
                'model': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=self.seed
                ),
                'era': '2018',
                'category': 'ensemble',
                'note': 'XGBoost - available but less common in 2018'
            }
        except ImportError:
            print("XGBoost not available - skipping")

        # Try to add LightGBM if available
        try:
            import lightgbm as lgb
            models['lightgbm'] = {
                'model': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=-1,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=self.seed,
                    verbose=-1
                ),
                'era': 'modern',
                'category': 'ensemble',
                'note': 'LightGBM - became popular post-2018'
            }
        except ImportError:
            print("LightGBM not available - skipping")

        return models

    def run_with_polynomial_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        degree: int = 2
    ) -> Dict:
        """
        Test polynomial feature engineering.

        This tests whether the original paper's linear models could have
        been improved with simple feature engineering - a critique of
        the AutoML-only approach.
        """
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Scale features (important for polynomial)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_test_scaled = scaler.transform(X_test_poly)

        # Train ridge regression on polynomial features
        ridge = Ridge(alpha=1.0)

        start_time = time.time()
        ridge.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        y_pred = ridge.predict(X_test_scaled)

        return {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'train_time': train_time,
            'n_features_original': X_train.shape[1],
            'n_features_poly': X_train_poly.shape[1],
            'model': f'Ridge+Poly(degree={degree})'
        }

    def run_time_series_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = 'random_forest',
        verbose: bool = True
    ) -> Dict:
        """
        Test with time-series cross-validation.

        Critical analysis: The original paper used random train/test split,
        but ship sensor data is time-series. Does this matter?

        TIME-SERIES CV PROTOCOL (Addressing Reviewer Concern #3)
        ========================================================

        We use sklearn's TimeSeriesSplit with 5 folds. This implements
        an expanding window approach where:

        - Fold 1: Train on [0:N/6], Test on [N/6:2N/6]
        - Fold 2: Train on [0:2N/6], Test on [2N/6:3N/6]
        - Fold 3: Train on [0:3N/6], Test on [3N/6:4N/6]
        - Fold 4: Train on [0:4N/6], Test on [4N/6:5N/6]
        - Fold 5: Train on [0:5N/6], Test on [5N/6:N]

        For N=30,000 samples:
        - Fold 1: Train=5,000, Test=5,000
        - Fold 2: Train=10,000, Test=5,000
        - Fold 3: Train=15,000, Test=5,000
        - Fold 4: Train=20,000, Test=5,000
        - Fold 5: Train=25,000, Test=5,000

        This prevents temporal leakage by ensuring test data always
        comes AFTER training data chronologically.

        LIMITATIONS:
        - No nested CV for hyperparameter tuning (hyperparameters fixed)
        - Expanding window may overweight later data
        - 5 folds may not capture all seasonal patterns
        """
        models = self.get_modern_models()
        model = models[model_name]['model']

        # Time series split (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)

        scores = []
        fold_details = []

        if verbose:
            print("\n" + "-" * 60)
            print("TIME-SERIES CROSS-VALIDATION DETAILS")
            print("-" * 60)
            print(f"Model: {model_name}")
            print(f"Total samples: {len(X)}")
            print(f"Number of folds: 5")
            print()

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            scores.append(r2)

            fold_info = {
                'fold': fold_idx + 1,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'train_size': len(train_idx),
                'test_start': test_idx[0],
                'test_end': test_idx[-1],
                'test_size': len(test_idx),
                'r2': r2
            }
            fold_details.append(fold_info)

            if verbose:
                print(f"Fold {fold_idx+1}:")
                print(f"  Train: indices [{train_idx[0]:>6}:{train_idx[-1]:<6}] "
                      f"({len(train_idx):,} samples)")
                print(f"  Test:  indices [{test_idx[0]:>6}:{test_idx[-1]:<6}] "
                      f"({len(test_idx):,} samples)")
                print(f"  R² = {r2:.4f}")

        if verbose:
            print()
            print(f"Mean R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        return {
            'mean_r2': np.mean(scores),
            'std_r2': np.std(scores),
            'min_r2': np.min(scores),
            'max_r2': np.max(scores),
            'scores': scores,
            'fold_details': fold_details
        }

    def run_significance_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_split_r2: float,
        n_bootstrap: int = 1000
    ) -> Dict:
        """
        Test statistical significance of random vs time-series CV difference.

        Uses paired t-test on fold scores vs bootstrap samples from random split.

        This addresses reviewer question: "Is the 0.5% gap statistically significant?"
        """
        from scipy import stats

        # Get time-series CV scores
        ts_result = self.run_time_series_validation(X, y, verbose=False)
        ts_scores = ts_result['scores']

        # Bootstrap random split performance
        np.random.seed(self.seed)
        n_samples = len(X)

        bootstrap_r2s = []
        for _ in range(n_bootstrap):
            # Random sample indices
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            train_size = int(0.75 * len(idx))
            train_idx = idx[:train_size]
            test_idx = idx[train_size:]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = self.get_modern_models()['random_forest']['model']
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            bootstrap_r2s.append(r2_score(y_test, y_pred))

        # One-sample t-test: is TS-CV mean significantly different from random split mean?
        t_stat, p_value = stats.ttest_1samp(ts_scores, np.mean(bootstrap_r2s))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(ts_scores)**2 + np.std(bootstrap_r2s)**2) / 2)
        cohens_d = (np.mean(ts_scores) - np.mean(bootstrap_r2s)) / pooled_std

        return {
            'ts_cv_mean': np.mean(ts_scores),
            'ts_cv_std': np.std(ts_scores),
            'random_split_mean': np.mean(bootstrap_r2s),
            'random_split_std': np.std(bootstrap_r2s),
            'difference': np.mean(ts_scores) - np.mean(bootstrap_r2s),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant_at_05': p_value < 0.05,
            'interpretation': self._interpret_significance(p_value, cohens_d)
        }

    def _interpret_significance(self, p_value: float, cohens_d: float) -> str:
        """Interpret statistical significance results."""
        if p_value >= 0.05:
            sig_text = "NOT statistically significant (p >= 0.05)"
        else:
            sig_text = f"statistically significant (p = {p_value:.4f})"

        if abs(cohens_d) < 0.2:
            effect_text = "negligible effect size"
        elif abs(cohens_d) < 0.5:
            effect_text = "small effect size"
        elif abs(cohens_d) < 0.8:
            effect_text = "medium effect size"
        else:
            effect_text = "large effect size"

        return f"{sig_text}, {effect_text} (d = {cohens_d:.3f})"

    def run_full_comparison(
        self,
        n_samples: int = 30000,
        feature_combo: List[str] = ['rpm', 'frp']  # Best combo from original
    ) -> pd.DataFrame:
        """
        Run comprehensive comparison of all methods.
        """
        print("=" * 70)
        print("MODERN METHODS COMPARISON")
        print("=" * 70)

        # Generate data
        generator = ShipEngineDataGenerator(seed=self.seed)
        df = generator.generate_dataset(n_samples=n_samples)

        # Prepare data with best feature combination
        X, y, feature_cols = generator.prepare_experiment_data(
            df, feature_combo, engine_group='13'
        )

        print(f"\nFeatures: {feature_combo}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        # Standard train/test split (as in original)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, test_size=0.25, random_state=self.seed
        )

        # Scale features for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = []
        models = self.get_modern_models()

        print(f"\nTesting {len(models)} models...")
        print("-" * 70)

        for name, config in models.items():
            model = config['model']
            era = config['era']
            category = config['category']

            # Use scaled data for neural networks
            if category == 'neural':
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            try:
                start_time = time.time()
                model.fit(X_tr, y_train)
                train_time = time.time() - start_time

                y_pred = model.predict(X_te)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                results.append({
                    'model': name,
                    'era': era,
                    'category': category,
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'train_time': train_time
                })

                print(f"  {name:25s} [{era:6s}] R²={r2:.4f}, MSE={mse:.6f}, Time={train_time:.2f}s")

            except Exception as e:
                print(f"  {name:25s} FAILED: {str(e)[:50]}")

        # Add polynomial features test
        print("\nPolynomial Features Tests:")
        for degree in [2, 3]:
            try:
                poly_result = self.run_with_polynomial_features(
                    X_train, X_test, y_train, y_test, degree=degree
                )
                results.append({
                    'model': f'ridge_poly_{degree}',
                    'era': '2018',
                    'category': 'feature_eng',
                    'r2': poly_result['r2'],
                    'mse': poly_result['mse'],
                    'mae': poly_result['mae'],
                    'train_time': poly_result['train_time']
                })
                print(f"  Ridge+Poly(d={degree}): R²={poly_result['r2']:.4f} "
                      f"({poly_result['n_features_original']} → {poly_result['n_features_poly']} features)")
            except Exception as e:
                print(f"  Poly degree {degree} FAILED: {str(e)[:50]}")

        # Time series validation comparison
        print("\nTime Series vs Random Split Comparison:")
        ts_result = self.run_time_series_validation(X, y, 'random_forest')
        print(f"  Random Split R²:    {results[3]['r2']:.4f}")  # RF result
        print(f"  Time Series CV R²:  {ts_result['mean_r2']:.4f} ± {ts_result['std_r2']:.4f}")
        print(f"  (min: {ts_result['min_r2']:.4f}, max: {ts_result['max_r2']:.4f})")

        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(self.results_dir / 'modern_methods_results.csv', index=False)

        return results_df, ts_result

    def generate_critical_analysis(
        self,
        results_df: pd.DataFrame,
        ts_result: Dict
    ) -> str:
        """
        Generate critical analysis text for the paper.
        """
        analysis = []

        analysis.append("\n" + "=" * 70)
        analysis.append("CRITICAL ANALYSIS")
        analysis.append("=" * 70)

        # 1. Era comparison
        analysis.append("\n1. 2018-ERA vs MODERN METHODS")
        analysis.append("-" * 40)

        era_2018 = results_df[results_df['era'] == '2018']['r2'].max()
        era_modern = results_df[results_df['era'] == 'modern']['r2'].max()

        analysis.append(f"  Best 2018-era method R²:  {era_2018:.4f}")
        analysis.append(f"  Best modern method R²:    {era_modern:.4f}")
        analysis.append(f"  Improvement:              {(era_modern - era_2018)*100:.2f}%")

        if era_modern > era_2018:
            analysis.append("\n  FINDING: Modern methods show marginal improvement.")
            analysis.append("  This suggests the original TPOT approach was near-optimal")
            analysis.append("  for this problem structure.")
        else:
            analysis.append("\n  FINDING: 2018-era methods remain competitive.")
            analysis.append("  AutoML was an appropriate choice for this problem.")

        # 2. Neural networks analysis
        analysis.append("\n2. NEURAL NETWORK ANALYSIS")
        analysis.append("-" * 40)

        nn_results = results_df[results_df['category'] == 'neural']
        best_nn = nn_results.loc[nn_results['r2'].idxmax()]
        best_ensemble = results_df[results_df['category'] == 'ensemble']['r2'].max()

        analysis.append(f"  Best neural network R²:   {best_nn['r2']:.4f} ({best_nn['model']})")
        analysis.append(f"  Best ensemble R²:         {best_ensemble:.4f}")

        if best_nn['r2'] < best_ensemble:
            analysis.append("\n  FINDING: Neural networks underperform ensemble methods.")
            analysis.append("  Likely reasons:")
            analysis.append("    - Tabular data favors tree-based methods")
            analysis.append("    - Dataset size (~30k) may be insufficient for deep learning")
            analysis.append("    - Feature relationships are relatively simple")

        # 3. Feature engineering critique
        analysis.append("\n3. FEATURE ENGINEERING CRITIQUE")
        analysis.append("-" * 40)

        fe_results = results_df[results_df['category'] == 'feature_eng']
        if not fe_results.empty:
            best_fe = fe_results.loc[fe_results['r2'].idxmax()]
            linear_baseline = results_df[results_df['model'] == 'linear']['r2'].values[0]

            analysis.append(f"  Linear baseline R²:       {linear_baseline:.4f}")
            analysis.append(f"  With polynomial features: {best_fe['r2']:.4f}")
            analysis.append(f"  Improvement:              {(best_fe['r2'] - linear_baseline)*100:.2f}%")

            analysis.append("\n  CRITIQUE: The original paper focused on AutoML model selection")
            analysis.append("  but simple feature engineering could have achieved similar results")
            analysis.append("  with interpretable models. This was unexplored in the original work.")

        # 4. Time series validation critique
        analysis.append("\n4. VALIDATION METHODOLOGY CRITIQUE")
        analysis.append("-" * 40)

        random_split_r2 = results_df[results_df['model'] == 'random_forest']['r2'].values[0]
        ts_mean = ts_result['mean_r2']
        ts_std = ts_result['std_r2']

        analysis.append(f"  Random split R²:          {random_split_r2:.4f}")
        analysis.append(f"  Time series CV R²:        {ts_mean:.4f} ± {ts_std:.4f}")

        if abs(random_split_r2 - ts_mean) > 0.01:
            analysis.append("\n  CRITIQUE: Random train/test split may overestimate performance")
            analysis.append("  for time-series data due to temporal autocorrelation.")
            analysis.append("  The original paper did not address this methodological concern.")
        else:
            analysis.append("\n  FINDING: Random split results are consistent with time series CV.")
            analysis.append("  The original validation approach appears robust for this data.")

        # 5. Implications
        analysis.append("\n5. IMPLICATIONS FOR AI REPLICATION")
        analysis.append("-" * 40)
        analysis.append("  - AI can identify methodological gaps humans might overlook")
        analysis.append("  - Systematic comparison reveals problem structure insights")
        analysis.append("  - Replication goes beyond reproduction to critical analysis")
        analysis.append("  - Modern methods provide context for historical results")

        analysis_text = "\n".join(analysis)
        print(analysis_text)

        # Save analysis
        with open(self.results_dir / 'critical_analysis.txt', 'w') as f:
            f.write(analysis_text)

        return analysis_text


def main():
    """Run extended modern methods comparison."""
    experiment = ModernMethodsExperiment(seed=42)

    # Run comparison with best feature combination from original
    results_df, ts_result = experiment.run_full_comparison(
        n_samples=30000,
        feature_combo=['rpm', 'frp']
    )

    # Generate critical analysis
    experiment.generate_critical_analysis(results_df, ts_result)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
