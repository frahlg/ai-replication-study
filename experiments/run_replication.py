#!/usr/bin/env python3
"""
AI-Driven Scientific Replication Study

This script replicates the methodology from:
Ahlgren, F., Thern, M. (2018). "Auto Machine Learning for predicting Ship Fuel Consumption"
ECOS 2018 Conference.

Meta-Study Objective:
Document and analyze an AI system (Claude Code) performing scientific replication,
following the scientific method with hypothesis, experiments, and analysis.

Replication Methodology:
1. Generate synthetic data matching original statistical properties
2. Apply same experimental design (feature combinations, train/test split)
3. Compare TPOT (original) with modern AutoML alternatives
4. Statistical comparison with reported results
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import local data generator
from data_generator import ShipEngineDataGenerator


class ReplicationExperiment:
    """
    Conduct replication experiment following original methodology.

    This class implements the experimental design from Ahlgren & Thern (2018)
    with extensions for meta-analysis of AI-driven replication.
    """

    def __init__(
        self,
        seed: int = 42,
        results_dir: str = '../results',
        figures_dir: str = '../figures'
    ):
        """Initialize experiment with reproducibility settings."""
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Track all experimental results
        self.results = []
        self.meta_log = []

        # Original paper baseline results (R² scores from paper)
        self.original_results = {
            'linear_baseline': 0.957,  # Linear regression baseline
            'tpot_best': 0.992,        # Best TPOT result
            'tpot_mean': 0.97,         # Mean TPOT across tests
        }

        self._log_meta("Experiment initialized", {
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "original_paper_baseline_r2": self.original_results['tpot_best']
        })

    def _log_meta(self, event: str, data: Dict[str, Any]):
        """Log meta-information about the replication process."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        }
        self.meta_log.append(entry)

    def run_baseline_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_desc: str
    ) -> Dict[str, Dict]:
        """
        Run baseline models (matching original paper's methodology).

        Original paper used:
        - Linear Regression as baseline
        - TPOT AutoML for optimization

        We add:
        - Ridge Regression
        - Random Forest
        - Gradient Boosting
        """
        results = {}

        # 1. Linear Regression (original baseline)
        start_time = time.time()
        lr = LinearRegression(n_jobs=-1)
        lr.fit(X_train, y_train)
        lr_time = time.time() - start_time

        y_pred_lr = lr.predict(X_test)
        results['linear'] = {
            'r2': r2_score(y_test, y_pred_lr),
            'mse': mean_squared_error(y_test, y_pred_lr),
            'mae': mean_absolute_error(y_test, y_pred_lr),
            'train_time': lr_time,
            'model': 'LinearRegression'
        }

        # 2. Ridge Regression (regularized baseline)
        start_time = time.time()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_time = time.time() - start_time

        y_pred_ridge = ridge.predict(X_test)
        results['ridge'] = {
            'r2': r2_score(y_test, y_pred_ridge),
            'mse': mean_squared_error(y_test, y_pred_ridge),
            'mae': mean_absolute_error(y_test, y_pred_ridge),
            'train_time': ridge_time,
            'model': 'Ridge'
        }

        # 3. Random Forest (ensemble baseline)
        start_time = time.time()
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=self.seed)
        rf.fit(X_train, y_train)
        rf_time = time.time() - start_time

        y_pred_rf = rf.predict(X_test)
        results['random_forest'] = {
            'r2': r2_score(y_test, y_pred_rf),
            'mse': mean_squared_error(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'train_time': rf_time,
            'model': 'RandomForest(n=100)'
        }

        # 4. Gradient Boosting
        start_time = time.time()
        gb = GradientBoostingRegressor(n_estimators=100, random_state=self.seed)
        gb.fit(X_train, y_train)
        gb_time = time.time() - start_time

        y_pred_gb = gb.predict(X_test)
        results['gradient_boosting'] = {
            'r2': r2_score(y_test, y_pred_gb),
            'mse': mean_squared_error(y_test, y_pred_gb),
            'mae': mean_absolute_error(y_test, y_pred_gb),
            'train_time': gb_time,
            'model': 'GradientBoosting(n=100)'
        }

        # 5. Extra Trees (often found in TPOT pipelines)
        start_time = time.time()
        et = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=self.seed)
        et.fit(X_train, y_train)
        et_time = time.time() - start_time

        y_pred_et = et.predict(X_test)
        results['extra_trees'] = {
            'r2': r2_score(y_test, y_pred_et),
            'mse': mean_squared_error(y_test, y_pred_et),
            'mae': mean_absolute_error(y_test, y_pred_et),
            'train_time': et_time,
            'model': 'ExtraTrees(n=100)'
        }

        # 6. Stacking Ensemble (simulates TPOT's typical best pipelines)
        start_time = time.time()
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=self.seed)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=self.seed)),
            ('et', ExtraTreesRegressor(n_estimators=50, random_state=self.seed)),
        ]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)
        stacking_time = time.time() - start_time

        y_pred_stack = stacking.predict(X_test)
        results['stacking_ensemble'] = {
            'r2': r2_score(y_test, y_pred_stack),
            'mse': mean_squared_error(y_test, y_pred_stack),
            'mae': mean_absolute_error(y_test, y_pred_stack),
            'train_time': stacking_time,
            'model': 'StackingEnsemble(RF+GB+ET+Ridge)'
        }

        self._log_meta("Baseline models completed", {
            "feature_combination": feature_desc,
            "best_r2": max(r['r2'] for r in results.values()),
            "models_tested": list(results.keys())
        })

        return results

    def run_tpot_experiment(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_desc: str,
        generations: int = 5,
        population_size: int = 20
    ) -> Dict:
        """
        Run TPOT AutoML (original paper's method).

        TPOT uses genetic programming to optimize ML pipelines.
        Original paper used generations=10, population_size=50.
        We use reduced parameters for faster execution in replication.
        """
        try:
            from tpot import TPOTRegressor
        except (ImportError, Exception) as e:
            self._log_meta("TPOT not available", {"error": str(e)})
            return {
                'r2': None,
                'mse': None,
                'mae': None,
                'train_time': None,
                'model': 'TPOT (not available)',
                'error': f'TPOT error: {str(e)[:100]}'
            }

        start_time = time.time()

        tpot = TPOTRegressor(
            generations=generations,
            population_size=population_size,
            verbosity=1,
            n_jobs=-1,
            random_state=self.seed,
            max_time_mins=5,  # Limit time for replication
            early_stop=3
        )

        tpot.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = tpot.predict(X_test)

        result = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'train_time': train_time,
            'model': f'TPOT(gen={generations}, pop={population_size})',
            'best_pipeline': str(tpot.fitted_pipeline_)
        }

        self._log_meta("TPOT experiment completed", {
            "feature_combination": feature_desc,
            "r2": result['r2'],
            "best_pipeline": result['best_pipeline'][:100] + "..."
        })

        return result

    def run_autogluon_experiment(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_desc: str,
        time_limit: int = 60
    ) -> Dict:
        """
        Run AutoGluon (modern AutoML comparison).

        AutoGluon is a state-of-the-art AutoML library that wasn't
        available when the original paper was published (2018).
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            self._log_meta("AutoGluon not available", {"error": "AutoGluon not installed"})
            return {
                'r2': None,
                'mse': None,
                'mae': None,
                'train_time': None,
                'model': 'AutoGluon (not installed)',
                'error': 'AutoGluon not installed - pip install autogluon'
            }

        # Prepare data in AutoGluon format
        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train

        test_data = pd.DataFrame(X_test)

        start_time = time.time()

        predictor = TabularPredictor(
            label='target',
            problem_type='regression',
            eval_metric='r2'
        ).fit(
            train_data,
            time_limit=time_limit,
            presets='medium_quality',
            verbosity=1
        )

        train_time = time.time() - start_time

        y_pred = predictor.predict(test_data)

        result = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'train_time': train_time,
            'model': f'AutoGluon(time={time_limit}s)',
            'best_model': predictor.get_model_best()
        }

        self._log_meta("AutoGluon experiment completed", {
            "feature_combination": feature_desc,
            "r2": result['r2'],
            "best_model": result['best_model']
        })

        return result

    def run_full_experiment(
        self,
        n_samples: int = 30000,
        run_tpot: bool = True,
        run_autogluon: bool = False  # Optional, can be slow
    ) -> pd.DataFrame:
        """
        Run the complete replication experiment.

        Following original methodology:
        1. Generate data
        2. Test all 15 feature combinations
        3. Use 75/25 train/test split
        4. Compare multiple models
        """
        print("=" * 70)
        print("AI-DRIVEN SCIENTIFIC REPLICATION STUDY")
        print("Replicating: Ahlgren & Thern (2018) - Ship Fuel Consumption ML")
        print("=" * 70)

        # Generate synthetic data
        generator = ShipEngineDataGenerator(seed=self.seed)
        df = generator.generate_dataset(n_samples=n_samples)

        # Get feature combinations (15 combinations as in original)
        combinations = generator.create_feature_combinations()

        print(f"\nTesting {len(combinations)} feature combinations...")
        print("-" * 70)

        all_results = []

        for i, (feature_types, desc) in enumerate(combinations):
            print(f"\n[{i+1}/{len(combinations)}] Features: {desc}")

            # Prepare data (using engine group 13, as in paper)
            X, y, feature_cols = generator.prepare_experiment_data(
                df, feature_types, engine_group='13'
            )

            # Train/test split (75/25 as in original)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.75, test_size=0.25, random_state=self.seed
            )

            # Run baseline models
            baseline_results = self.run_baseline_models(
                X_train, X_test, y_train, y_test, desc
            )

            for model_name, metrics in baseline_results.items():
                result_entry = {
                    'combination_id': i + 1,
                    'features': desc,
                    'n_features': len(feature_cols),
                    'method': model_name,
                    **metrics
                }
                all_results.append(result_entry)
                print(f"  {model_name}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.6f}")

            # Run TPOT (original method)
            if run_tpot:
                tpot_result = self.run_tpot_experiment(
                    X_train, X_test, y_train, y_test, desc,
                    generations=5, population_size=20
                )
                if tpot_result['r2'] is not None:
                    result_entry = {
                        'combination_id': i + 1,
                        'features': desc,
                        'n_features': len(feature_cols),
                        'method': 'tpot',
                        **tpot_result
                    }
                    all_results.append(result_entry)
                    print(f"  TPOT: R²={tpot_result['r2']:.4f}, MSE={tpot_result['mse']:.6f}")

            # Run AutoGluon (modern comparison)
            if run_autogluon:
                ag_result = self.run_autogluon_experiment(
                    X_train, X_test, y_train, y_test, desc,
                    time_limit=60
                )
                if ag_result['r2'] is not None:
                    result_entry = {
                        'combination_id': i + 1,
                        'features': desc,
                        'n_features': len(feature_cols),
                        'method': 'autogluon',
                        **ag_result
                    }
                    all_results.append(result_entry)
                    print(f"  AutoGluon: R²={ag_result['r2']:.4f}")

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Save results
        results_path = self.results_dir / 'replication_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Save meta log
        meta_path = self.results_dir / 'meta_log.json'
        with open(meta_path, 'w') as f:
            json.dump(self.meta_log, f, indent=2, default=str)
        print(f"Meta log saved to: {meta_path}")

        return results_df

    def generate_comparison_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate comparison report between replication and original results.
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("REPLICATION COMPARISON REPORT")
        report.append("=" * 70)

        # Overall statistics
        report.append("\n1. OVERALL PERFORMANCE SUMMARY")
        report.append("-" * 40)

        for method in results_df['method'].unique():
            method_results = results_df[results_df['method'] == method]
            mean_r2 = method_results['r2'].mean()
            max_r2 = method_results['r2'].max()
            report.append(f"  {method:20s}: Mean R²={mean_r2:.4f}, Best R²={max_r2:.4f}")

        # Comparison with original
        report.append("\n2. COMPARISON WITH ORIGINAL PAPER")
        report.append("-" * 40)
        report.append(f"  Original TPOT Best R²:  {self.original_results['tpot_best']:.4f}")
        report.append(f"  Original Linear R²:     {self.original_results['linear_baseline']:.4f}")

        best_replication = results_df['r2'].max()
        report.append(f"  Replication Best R²:    {best_replication:.4f}")

        # Feature importance analysis
        report.append("\n3. FEATURE COMBINATION ANALYSIS")
        report.append("-" * 40)

        best_by_combo = results_df.groupby('features')['r2'].max().sort_values(ascending=False)
        report.append("  Top 5 Feature Combinations:")
        for features, r2 in best_by_combo.head().items():
            report.append(f"    {features:30s}: R²={r2:.4f}")

        # Replication success assessment
        report.append("\n4. REPLICATION ASSESSMENT")
        report.append("-" * 40)

        threshold = 0.95  # Consider successful if within 5% of original
        success = best_replication >= (self.original_results['tpot_best'] * threshold)

        if success:
            report.append("  STATUS: REPLICATION SUCCESSFUL")
            report.append(f"  Achieved {best_replication/self.original_results['tpot_best']*100:.1f}% of original performance")
        else:
            report.append("  STATUS: PARTIAL REPLICATION")
            gap = self.original_results['tpot_best'] - best_replication
            report.append(f"  Performance gap: {gap:.4f} R² points")

        report_text = "\n".join(report)
        print(report_text)

        # Save report
        report_path = self.results_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text

    def create_visualizations(self, results_df: pd.DataFrame):
        """Create publication-quality visualizations."""

        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. R² scores by method
        fig, ax = plt.subplots(figsize=(10, 6))
        methods_order = results_df.groupby('method')['r2'].mean().sort_values(ascending=False).index
        sns.boxplot(data=results_df, x='method', y='r2', order=methods_order, ax=ax)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Model Performance Comparison (Replication Study)', fontsize=14)
        ax.axhline(y=self.original_results['tpot_best'], color='red',
                   linestyle='--', label=f'Original TPOT Best ({self.original_results["tpot_best"]})')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'method_comparison.png', dpi=300)
        plt.close()

        # 2. Performance by number of features
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = results_df.pivot_table(values='r2', index='n_features',
                                        columns='method', aggfunc='mean')
        pivot.plot(ax=ax, marker='o')
        ax.set_xlabel('Number of Feature Types', fontsize=12)
        ax.set_ylabel('Mean R² Score', fontsize=12)
        ax.set_title('Performance vs Feature Complexity', fontsize=14)
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'feature_complexity.png', dpi=300)
        plt.close()

        # 3. Heatmap of results
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_heatmap = results_df.pivot_table(values='r2', index='features',
                                                columns='method', aggfunc='mean')
        sns.heatmap(pivot_heatmap, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=ax, vmin=0.8, vmax=1.0)
        ax.set_title('R² Scores by Feature Combination and Method', fontsize=14)
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'results_heatmap.png', dpi=300)
        plt.close()

        print(f"\nVisualizations saved to: {self.figures_dir}")


def main():
    """Main entry point for replication experiment."""
    print("\n" + "=" * 70)
    print("STARTING AI-DRIVEN SCIENTIFIC REPLICATION")
    print("=" * 70)
    print(f"\nExperiment Start: {datetime.now().isoformat()}")
    print("Original Paper: Ahlgren & Thern (2018) - ECOS 2018")
    print("Replication Agent: Claude Code (Anthropic)")
    print("=" * 70)

    # Initialize experiment
    experiment = ReplicationExperiment(
        seed=42,
        results_dir='../results',
        figures_dir='../figures'
    )

    # Run full experiment
    # Note: Set run_tpot=True only if TPOT is installed and working
    # Set run_autogluon=True only if AutoGluon is installed
    results_df = experiment.run_full_experiment(
        n_samples=30000,
        run_tpot=False,  # Disabled due to xgboost/libomp dependency issues
        run_autogluon=False  # Optional, can be slow
    )

    # Generate comparison report
    experiment.generate_comparison_report(results_df)

    # Create visualizations
    experiment.create_visualizations(results_df)

    print("\n" + "=" * 70)
    print("REPLICATION EXPERIMENT COMPLETE")
    print(f"Experiment End: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
