#!/usr/bin/env python3
"""
Uncertainty Analysis for AI Replication Study

This module addresses reviewer concern #4: "No uncertainty quantification -
single-run results not credible."

Methodology:
1. Run all experiments with 10 different random seeds
2. Report mean ± std for all R² values
3. Compute bootstrap confidence intervals (1000 resamples)
4. Test robustness of model rankings across seeds

Seeds chosen: 42, 123, 456, 789, 1024, 2048, 3141, 4269, 5555, 6789
(First is original, others are arbitrary but fixed for reproducibility)
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

from data_generator import ShipEngineDataGenerator


# Fixed seeds for reproducibility
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4269, 5555, 6789]


class UncertaintyAnalysis:
    """
    Multi-seed uncertainty quantification for all experiments.

    This class runs all model experiments across multiple random seeds
    to provide robust uncertainty estimates for reported results.
    """

    def __init__(self, seeds: List[int] = None, results_dir: str = '../results'):
        self.seeds = seeds or RANDOM_SEEDS
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_bootstrap = 1000

    def get_models(self, seed: int) -> Dict:
        """Return dictionary of models to test with specific seed."""
        models = {
            'linear': {
                'model': LinearRegression(),
                'era': '2018',
                'category': 'linear',
                'needs_scaling': False
            },
            'ridge': {
                'model': Ridge(alpha=1.0),
                'era': '2018',
                'category': 'linear',
                'needs_scaling': False
            },
            'elastic_net': {
                'model': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
                'era': '2018',
                'category': 'linear',
                'needs_scaling': False
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100, max_depth=None, n_jobs=-1, random_state=seed
                ),
                'era': '2018',
                'category': 'ensemble',
                'needs_scaling': False
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, random_state=seed
                ),
                'era': '2018',
                'category': 'ensemble',
                'needs_scaling': False
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(
                    n_estimators=100, n_jobs=-1, random_state=seed
                ),
                'era': '2018',
                'category': 'ensemble',
                'needs_scaling': False
            },
            'hist_gradient_boosting': {
                'model': HistGradientBoostingRegressor(
                    max_iter=100, max_depth=None, random_state=seed
                ),
                'era': 'modern',
                'category': 'ensemble',
                'needs_scaling': False
            },
            'mlp_small': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                    max_iter=500, early_stopping=True, random_state=seed
                ),
                'era': '2018',
                'category': 'neural',
                'needs_scaling': True
            },
            'mlp_medium': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                    max_iter=500, early_stopping=True, random_state=seed
                ),
                'era': '2018',
                'category': 'neural',
                'needs_scaling': True
            },
            'mlp_large': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32), activation='relu', solver='adam',
                    max_iter=1000, early_stopping=True, random_state=seed
                ),
                'era': 'modern',
                'category': 'neural',
                'needs_scaling': True
            },
        }

        # Try to add XGBoost
        try:
            import xgboost as xgb
            models['xgboost'] = {
                'model': xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    n_jobs=-1, random_state=seed
                ),
                'era': '2018',
                'category': 'ensemble',
                'needs_scaling': False
            }
        except ImportError:
            pass

        # Try to add LightGBM
        try:
            import lightgbm as lgb
            models['lightgbm'] = {
                'model': lgb.LGBMRegressor(
                    n_estimators=100, max_depth=-1, learning_rate=0.1,
                    n_jobs=-1, random_state=seed, verbose=-1
                ),
                'era': 'modern',
                'category': 'ensemble',
                'needs_scaling': False
            }
        except ImportError:
            pass

        return models

    def run_single_experiment(
        self,
        seed: int,
        n_samples: int = 30000,
        feature_combo: List[str] = ['rpm', 'frp']
    ) -> pd.DataFrame:
        """Run all models with a single seed."""
        np.random.seed(seed)

        # Generate data with this seed
        generator = ShipEngineDataGenerator(seed=seed)
        df = generator.generate_dataset(n_samples=n_samples)
        X, y, _ = generator.prepare_experiment_data(df, feature_combo, engine_group='13')

        # Split with this seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, test_size=0.25, random_state=seed
        )

        # Scale for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = []
        models = self.get_models(seed)

        for name, config in models.items():
            model = config['model']
            needs_scaling = config['needs_scaling']

            X_tr = X_train_scaled if needs_scaling else X_train
            X_te = X_test_scaled if needs_scaling else X_test

            try:
                start_time = time.time()
                model.fit(X_tr, y_train)
                train_time = time.time() - start_time

                y_pred = model.predict(X_te)

                results.append({
                    'seed': seed,
                    'model': name,
                    'era': config['era'],
                    'category': config['category'],
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"    {name} failed with seed {seed}: {str(e)[:50]}")

        # Add polynomial features
        for degree in [2, 3]:
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)

                poly_scaler = StandardScaler()
                X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
                X_test_poly_scaled = poly_scaler.transform(X_test_poly)

                ridge = Ridge(alpha=1.0)

                start_time = time.time()
                ridge.fit(X_train_poly_scaled, y_train)
                train_time = time.time() - start_time

                y_pred = ridge.predict(X_test_poly_scaled)

                results.append({
                    'seed': seed,
                    'model': f'ridge_poly_{degree}',
                    'era': '2018',
                    'category': 'feature_eng',
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"    ridge_poly_{degree} failed with seed {seed}: {str(e)[:50]}")

        return pd.DataFrame(results)

    def compute_bootstrap_ci(
        self,
        values: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(values)
        bootstrap_means = []

        for _ in range(self.n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_means, alpha * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return ci_lower, ci_upper

    def run_full_analysis(
        self,
        n_samples: int = 30000,
        feature_combo: List[str] = ['rpm', 'frp']
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all experiments across all seeds and compute statistics.

        Returns:
            all_results: Raw results for each seed
            summary: Aggregated statistics with uncertainty bounds
        """
        print("=" * 70)
        print("UNCERTAINTY ANALYSIS")
        print(f"Running {len(self.seeds)} seeds: {self.seeds}")
        print("=" * 70)

        all_results = []

        for i, seed in enumerate(self.seeds):
            print(f"\n[{i+1}/{len(self.seeds)}] Running with seed={seed}...")
            seed_results = self.run_single_experiment(
                seed=seed, n_samples=n_samples, feature_combo=feature_combo
            )
            all_results.append(seed_results)

        all_results_df = pd.concat(all_results, ignore_index=True)

        # Compute summary statistics
        summary_rows = []

        for model in all_results_df['model'].unique():
            model_data = all_results_df[all_results_df['model'] == model]

            r2_values = model_data['r2'].values
            mse_values = model_data['mse'].values
            mae_values = model_data['mae'].values
            time_values = model_data['train_time'].values

            # Bootstrap CI for R²
            r2_ci_lower, r2_ci_upper = self.compute_bootstrap_ci(r2_values)

            summary_rows.append({
                'model': model,
                'era': model_data['era'].iloc[0],
                'category': model_data['category'].iloc[0],
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values),
                'r2_ci_lower': r2_ci_lower,
                'r2_ci_upper': r2_ci_upper,
                'r2_min': np.min(r2_values),
                'r2_max': np.max(r2_values),
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'train_time_mean': np.mean(time_values),
                'train_time_std': np.std(time_values),
                'n_seeds': len(r2_values)
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('r2_mean', ascending=False)

        # Save results
        all_results_df.to_csv(
            self.results_dir / 'uncertainty_all_seeds.csv', index=False
        )
        summary_df.to_csv(
            self.results_dir / 'uncertainty_summary.csv', index=False
        )

        return all_results_df, summary_df

    def analyze_ranking_stability(self, all_results: pd.DataFrame) -> Dict:
        """
        Analyze whether model rankings are stable across seeds.

        Returns statistics on ranking consistency.
        """
        rankings_per_seed = []

        for seed in self.seeds:
            seed_data = all_results[all_results['seed'] == seed]
            seed_data = seed_data.sort_values('r2', ascending=False)
            rankings_per_seed.append(seed_data['model'].tolist())

        # Check if top-3 models are consistent
        top3_per_seed = [r[:3] for r in rankings_per_seed]

        # Count how often each model appears in top-3
        from collections import Counter
        top3_counts = Counter()
        for top3 in top3_per_seed:
            top3_counts.update(top3)

        # Find the most common top-1 model
        top1_per_seed = [r[0] for r in rankings_per_seed]
        top1_counts = Counter(top1_per_seed)

        return {
            'top1_stability': top1_counts.most_common(1)[0][1] / len(self.seeds),
            'top1_most_common': top1_counts.most_common(1)[0][0],
            'top3_appearances': dict(top3_counts),
            'rankings_per_seed': rankings_per_seed
        }

    def print_summary(
        self,
        summary: pd.DataFrame,
        ranking_analysis: Dict
    ) -> str:
        """Print and return formatted summary."""
        lines = []

        lines.append("\n" + "=" * 70)
        lines.append("UNCERTAINTY ANALYSIS SUMMARY")
        lines.append("=" * 70)

        lines.append(f"\nNumber of seeds: {len(self.seeds)}")
        lines.append(f"Bootstrap samples: {self.n_bootstrap}")

        lines.append("\n" + "-" * 70)
        lines.append("MODEL PERFORMANCE (sorted by mean R²)")
        lines.append("-" * 70)
        lines.append(f"{'Model':<25} {'Mean R²':>10} {'Std':>8} {'95% CI':>18}")
        lines.append("-" * 70)

        for _, row in summary.iterrows():
            ci_str = f"[{row['r2_ci_lower']:.4f}, {row['r2_ci_upper']:.4f}]"
            lines.append(
                f"{row['model']:<25} {row['r2_mean']:>10.4f} {row['r2_std']:>8.4f} {ci_str:>18}"
            )

        lines.append("\n" + "-" * 70)
        lines.append("RANKING STABILITY")
        lines.append("-" * 70)
        lines.append(
            f"Top-1 model consistency: {ranking_analysis['top1_stability']*100:.0f}% "
            f"({ranking_analysis['top1_most_common']})"
        )
        lines.append("\nTop-3 appearances across seeds:")
        for model, count in sorted(
            ranking_analysis['top3_appearances'].items(),
            key=lambda x: -x[1]
        )[:5]:
            lines.append(f"  {model}: {count}/{len(self.seeds)} seeds")

        summary_text = "\n".join(lines)
        print(summary_text)

        # Save summary
        with open(self.results_dir / 'uncertainty_report.txt', 'w') as f:
            f.write(summary_text)

        return summary_text


def main():
    """Run full uncertainty analysis."""
    print("Starting uncertainty analysis...")
    print(f"This will run experiments across {len(RANDOM_SEEDS)} seeds.")
    print("Estimated time: 2-3 hours\n")

    start_time = time.time()

    analysis = UncertaintyAnalysis(seeds=RANDOM_SEEDS)

    all_results, summary = analysis.run_full_analysis(
        n_samples=30000,
        feature_combo=['rpm', 'frp']
    )

    ranking_analysis = analysis.analyze_ranking_stability(all_results)
    analysis.print_summary(summary, ranking_analysis)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {analysis.results_dir}")


if __name__ == '__main__':
    main()
