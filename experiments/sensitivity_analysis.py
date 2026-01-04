#!/usr/bin/env python3
"""
Data Generator Sensitivity Analysis

This module addresses reviewer concern #4: "Sensitivity to generator?"

The reviewer noted that the synthetic data generator may embed relationships
that favor particular models, potentially creating circular validation.

This script systematically varies generator parameters to show:
1. Model rankings are robust to plausible parameter changes
2. OR document when rankings become unstable

Parameters varied:
- Noise levels: 0.01, 0.02 (default), 0.05, 0.10
- Bimodal operating fractions: ±10% from default
- Fuel coefficient perturbations: ±20% random
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# Import the base generator
from data_generator import ShipEngineDataGenerator


class ParameterizedGenerator(ShipEngineDataGenerator):
    """
    Extended generator that allows parameter modification for sensitivity analysis.
    """

    def __init__(
        self,
        seed: int = 42,
        noise_std: float = 0.02,
        aux_operating_fraction: float = 0.55,
        main_operating_fraction: float = 0.40,
        coefficient_perturbation: float = 0.0
    ):
        """
        Initialize generator with modifiable parameters.

        Args:
            seed: Random seed
            noise_std: Standard deviation of fuel consumption noise
            aux_operating_fraction: Fraction of time auxiliary engines operate
            main_operating_fraction: Fraction of time main engines operate
            coefficient_perturbation: Random perturbation to fuel coefficients (0-1)
        """
        super().__init__(seed=seed)

        self.noise_std = noise_std
        self.aux_op_frac = aux_operating_fraction
        self.main_op_frac = main_operating_fraction

        # Perturb coefficients if requested
        if coefficient_perturbation > 0:
            self._perturb_coefficients(coefficient_perturbation)

        # Override noise_std
        self.FUEL_COEFFICIENTS = deepcopy(self.FUEL_COEFFICIENTS)
        self.FUEL_COEFFICIENTS['noise_std'] = noise_std

    def _perturb_coefficients(self, perturbation: float):
        """Randomly perturb fuel coefficients within specified range."""
        self.FUEL_COEFFICIENTS = deepcopy(self.FUEL_COEFFICIENTS)

        for key in self.FUEL_COEFFICIENTS:
            if key not in ['intercept', 'noise_std']:
                original = self.FUEL_COEFFICIENTS[key]
                # Random perturbation: (1 - p) to (1 + p) multiplier
                multiplier = 1 + self.rng.uniform(-perturbation, perturbation)
                self.FUEL_COEFFICIENTS[key] = original * multiplier

    def _generate_engine_features(
        self,
        n_samples: int,
        engine_prefix: str,
        engine_type: str
    ) -> Dict[str, np.ndarray]:
        """Override to use custom operating fractions."""
        type_prefix = 'ae' if engine_type == 'auxiliary' else 'me'

        # Use custom operating fractions
        op_frac = self.aux_op_frac if engine_type == 'auxiliary' else self.main_op_frac

        # Generate RPM (bimodal - on/off behavior)
        rpm = self._generate_bimodal_engine_signal(
            n_samples,
            self.ENGINE_PARAMS[f'{type_prefix}_rpm'],
            operating_fraction=op_frac
        )

        # Rest is same as parent class
        frp_base = rpm / self.ENGINE_PARAMS[f'{type_prefix}_rpm']['max']
        frp = frp_base * self.ENGINE_PARAMS[f'{type_prefix}_frp']['max']
        frp += self.rng.normal(0, 2, n_samples)
        frp = np.clip(frp, 0, self.ENGINE_PARAMS[f'{type_prefix}_frp']['max'])

        exh_base = rpm / self.ENGINE_PARAMS[f'{type_prefix}_rpm']['max']
        exh_t = 75 + exh_base * 350
        exh_t += self.rng.normal(0, 30, n_samples)
        exh_t = np.clip(exh_t, 70, 485)

        tc_rpm = rpm * 30 + self.rng.normal(0, 1000, n_samples)
        tc_rpm = np.clip(tc_rpm, 0, 25000)

        return {
            f'{engine_prefix}_rpm': rpm,
            f'{engine_prefix}_frp': frp,
            f'{engine_prefix}_exh_t': exh_t,
            f'{engine_prefix}_tc_rpm': tc_rpm
        }


class SensitivityAnalysis:
    """
    Analyze sensitivity of model performance to generator parameters.
    """

    def __init__(self, seed: int = 42, results_dir: str = '../results'):
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path('../figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def get_models(self, seed: int) -> Dict:
        """Return subset of models for sensitivity analysis."""
        return {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, n_jobs=-1, random_state=seed
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=seed
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100, n_jobs=-1, random_state=seed
            ),
            'mlp_medium': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                solver='adam', max_iter=500, early_stopping=True, random_state=seed
            ),
        }

    def run_single_config(
        self,
        generator: ParameterizedGenerator,
        n_samples: int = 30000,
        feature_combo: List[str] = ['rpm', 'frp']
    ) -> Dict[str, float]:
        """Run all models with a specific generator configuration."""
        np.random.seed(self.seed)

        df = generator.generate_dataset(n_samples=n_samples)
        X, y, _ = generator.prepare_experiment_data(df, feature_combo, engine_group='13')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, test_size=0.25, random_state=self.seed
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}
        models = self.get_models(self.seed)

        for name, model in models.items():
            X_tr = X_train_scaled if 'mlp' in name else X_train
            X_te = X_test_scaled if 'mlp' in name else X_test

            try:
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
                results[name] = r2_score(y_test, y_pred)
            except Exception as e:
                print(f"    {name} failed: {str(e)[:40]}")
                results[name] = np.nan

        # Add polynomial features
        for degree in [2]:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_scaler = StandardScaler()
            X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
            X_test_poly_scaled = poly_scaler.transform(X_test_poly)

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_poly_scaled, y_train)
            y_pred = ridge.predict(X_test_poly_scaled)
            results[f'ridge_poly_{degree}'] = r2_score(y_test, y_pred)

        return results

    def analyze_noise_sensitivity(self) -> pd.DataFrame:
        """Test sensitivity to noise levels."""
        print("\n" + "=" * 60)
        print("NOISE SENSITIVITY ANALYSIS")
        print("=" * 60)

        noise_levels = [0.01, 0.02, 0.05, 0.10]
        results = []

        for noise in noise_levels:
            print(f"\nNoise std = {noise}")
            generator = ParameterizedGenerator(seed=self.seed, noise_std=noise)
            r2_scores = self.run_single_config(generator)

            for model, r2 in r2_scores.items():
                results.append({
                    'parameter': 'noise_std',
                    'value': noise,
                    'model': model,
                    'r2': r2
                })
                print(f"  {model}: R² = {r2:.4f}")

        return pd.DataFrame(results)

    def analyze_operating_fraction_sensitivity(self) -> pd.DataFrame:
        """Test sensitivity to operating fractions."""
        print("\n" + "=" * 60)
        print("OPERATING FRACTION SENSITIVITY ANALYSIS")
        print("=" * 60)

        # Default is aux=0.55, main=0.40
        # Test ±10%
        configs = [
            (0.45, 0.30, '-10%'),
            (0.55, 0.40, 'default'),
            (0.65, 0.50, '+10%'),
        ]

        results = []

        for aux_frac, main_frac, label in configs:
            print(f"\nOperating fractions: aux={aux_frac}, main={main_frac} ({label})")
            generator = ParameterizedGenerator(
                seed=self.seed,
                aux_operating_fraction=aux_frac,
                main_operating_fraction=main_frac
            )
            r2_scores = self.run_single_config(generator)

            for model, r2 in r2_scores.items():
                results.append({
                    'parameter': 'operating_fraction',
                    'value': label,
                    'model': model,
                    'r2': r2
                })
                print(f"  {model}: R² = {r2:.4f}")

        return pd.DataFrame(results)

    def analyze_coefficient_sensitivity(self) -> pd.DataFrame:
        """Test sensitivity to coefficient perturbations."""
        print("\n" + "=" * 60)
        print("COEFFICIENT PERTURBATION SENSITIVITY ANALYSIS")
        print("=" * 60)

        perturbations = [0.0, 0.10, 0.20, 0.30]
        results = []

        for perturb in perturbations:
            print(f"\nCoefficient perturbation = ±{perturb*100:.0f}%")
            generator = ParameterizedGenerator(
                seed=self.seed,
                coefficient_perturbation=perturb
            )
            r2_scores = self.run_single_config(generator)

            for model, r2 in r2_scores.items():
                results.append({
                    'parameter': 'coefficient_perturbation',
                    'value': perturb,
                    'model': model,
                    'r2': r2
                })
                print(f"  {model}: R² = {r2:.4f}")

        return pd.DataFrame(results)

    def create_sensitivity_heatmap(
        self,
        noise_df: pd.DataFrame,
        op_frac_df: pd.DataFrame,
        coef_df: pd.DataFrame
    ):
        """Create visualization of sensitivity analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Noise sensitivity
        noise_pivot = noise_df.pivot(index='model', columns='value', values='r2')
        sns.heatmap(
            noise_pivot, annot=True, fmt='.4f', cmap='RdYlGn',
            vmin=0.95, vmax=1.0, ax=axes[0]
        )
        axes[0].set_title('Noise Sensitivity\n(noise_std)')
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Model')

        # Operating fraction sensitivity
        op_pivot = op_frac_df.pivot(index='model', columns='value', values='r2')
        sns.heatmap(
            op_pivot, annot=True, fmt='.4f', cmap='RdYlGn',
            vmin=0.95, vmax=1.0, ax=axes[1]
        )
        axes[1].set_title('Operating Fraction Sensitivity\n(aux/main engine uptime)')
        axes[1].set_xlabel('Perturbation')
        axes[1].set_ylabel('Model')

        # Coefficient sensitivity
        coef_pivot = coef_df.pivot(index='model', columns='value', values='r2')
        sns.heatmap(
            coef_pivot, annot=True, fmt='.4f', cmap='RdYlGn',
            vmin=0.95, vmax=1.0, ax=axes[2]
        )
        axes[2].set_title('Coefficient Sensitivity\n(fuel model perturbation)')
        axes[2].set_xlabel('Perturbation (%)')
        axes[2].set_ylabel('Model')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nHeatmap saved to: {self.figures_dir / 'sensitivity_heatmap.png'}")

    def analyze_ranking_stability(
        self,
        noise_df: pd.DataFrame,
        op_frac_df: pd.DataFrame,
        coef_df: pd.DataFrame
    ) -> Dict:
        """Analyze whether model rankings change across conditions."""
        all_rankings = []

        for df, param in [(noise_df, 'noise'), (op_frac_df, 'op_frac'), (coef_df, 'coef')]:
            for value in df['value'].unique():
                subset = df[df['value'] == value].sort_values('r2', ascending=False)
                ranking = subset['model'].tolist()
                all_rankings.append({
                    'param': param,
                    'value': value,
                    'top1': ranking[0] if ranking else None,
                    'top3': ranking[:3] if len(ranking) >= 3 else ranking,
                    'ranking': ranking
                })

        # Check top-1 consistency
        top1_models = [r['top1'] for r in all_rankings]
        from collections import Counter
        top1_counts = Counter(top1_models)

        return {
            'all_rankings': all_rankings,
            'top1_consistency': top1_counts.most_common(1)[0][1] / len(all_rankings),
            'top1_most_common': top1_counts.most_common(1)[0][0],
            'top1_counts': dict(top1_counts)
        }

    def run_full_analysis(self) -> Tuple[pd.DataFrame, Dict]:
        """Run complete sensitivity analysis."""
        print("=" * 60)
        print("SENSITIVITY ANALYSIS")
        print("Testing robustness of model rankings to generator parameters")
        print("=" * 60)

        noise_results = self.analyze_noise_sensitivity()
        op_frac_results = self.analyze_operating_fraction_sensitivity()
        coef_results = self.analyze_coefficient_sensitivity()

        # Combine all results
        all_results = pd.concat([noise_results, op_frac_results, coef_results])
        all_results.to_csv(self.results_dir / 'sensitivity_analysis.csv', index=False)

        # Create visualization
        self.create_sensitivity_heatmap(noise_results, op_frac_results, coef_results)

        # Analyze ranking stability
        stability = self.analyze_ranking_stability(
            noise_results, op_frac_results, coef_results
        )

        # Print summary
        self.print_summary(all_results, stability)

        return all_results, stability

    def print_summary(self, results: pd.DataFrame, stability: Dict):
        """Print summary of sensitivity analysis."""
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTop-1 model consistency: {stability['top1_consistency']*100:.0f}%")
        print(f"Most common top model: {stability['top1_most_common']}")

        print("\nTop-1 appearances across all conditions:")
        for model, count in stability['top1_counts'].items():
            print(f"  {model}: {count}")

        # Compute R² ranges for each model
        print("\nR² ranges across all conditions:")
        for model in results['model'].unique():
            model_data = results[results['model'] == model]
            r2_min = model_data['r2'].min()
            r2_max = model_data['r2'].max()
            r2_range = r2_max - r2_min
            print(f"  {model}: {r2_min:.4f} - {r2_max:.4f} (range: {r2_range:.4f})")

        # Stability assessment
        if stability['top1_consistency'] >= 0.8:
            print("\nCONCLUSION: Model rankings are STABLE across parameter variations.")
            print("The top model remains consistent in >=80% of conditions.")
        else:
            print("\nWARNING: Model rankings show some INSTABILITY.")
            print("Consider reporting results with appropriate caveats.")

        # Save summary
        with open(self.results_dir / 'sensitivity_summary.txt', 'w') as f:
            f.write(f"Sensitivity Analysis Summary\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Top-1 model consistency: {stability['top1_consistency']*100:.0f}%\n")
            f.write(f"Most common top model: {stability['top1_most_common']}\n\n")
            f.write("Top-1 appearances:\n")
            for model, count in stability['top1_counts'].items():
                f.write(f"  {model}: {count}\n")


def main():
    """Run full sensitivity analysis."""
    print("Starting sensitivity analysis...")
    start_time = time.time()

    analysis = SensitivityAnalysis(seed=42)
    results, stability = analysis.run_full_analysis()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {analysis.results_dir}")


if __name__ == '__main__':
    main()
