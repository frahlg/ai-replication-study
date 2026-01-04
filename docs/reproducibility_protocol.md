# Reproducibility Protocol

This document addresses Reviewer Question #9: "What hardware and software
environment were used (CPU/GPU, library versions), and were multiple seeds
evaluated to quantify variance?"

## Environment Specification

### Hardware
- **Platform**: macOS (Darwin)
- **Architecture**: ARM64 (Apple Silicon)
- **CPU**: Apple M-series chip
- **Memory**: 16+ GB RAM
- **GPU**: Not used (CPU-only training)

### Software

#### Python Environment
```
Python 3.10+
Package Manager: uv (recommended) or pip
```

#### Core Dependencies
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

#### Optional Dependencies
```
xgboost>=1.7.0      # For XGBoost comparison
lightgbm>=4.0.0     # For LightGBM comparison
tpot>=0.12.0        # For TPOT replication (may have install issues)
autogluon>=1.0.0    # For AutoGluon comparison (optional)
```

### Installation

```bash
# Clone repository
git clone https://github.com/[repo]/ai-replication-study.git
cd ai-replication-study

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

## Random Seed Strategy

### Primary Seed
All experiments use `seed=42` as the primary random state for reproducibility.

### Multi-Seed Analysis
For uncertainty quantification, we use 10 seeds:
```python
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4269, 5555, 6789]
```

Seeds are applied at three levels:
1. **Data generation**: `ShipEngineDataGenerator(seed=X)`
2. **Train/test split**: `train_test_split(..., random_state=X)`
3. **Model training**: `RandomForestRegressor(..., random_state=X)`

## Execution Order

Run experiments in this order for full reproducibility:

### Step 1: Generate Data and Run Main Replication
```bash
cd experiments
python run_replication.py
```
Expected output:
- `results/replication_results.csv` (90 rows)
- `results/comparison_report.txt`
- `figures/method_comparison.png`
- `figures/feature_complexity.png`
- `figures/results_heatmap.png`

### Step 2: Run Modern Methods Comparison
```bash
python modern_methods.py
```
Expected output:
- `results/modern_methods_results.csv` (14-16 rows)
- `results/critical_analysis.txt`

### Step 3: Run Uncertainty Analysis
```bash
python run_uncertainty_analysis.py
```
Expected output:
- `results/uncertainty_all_seeds.csv`
- `results/uncertainty_summary.csv`
- `results/uncertainty_report.txt`

Estimated time: 2-3 hours (runs all models × 10 seeds)

### Step 4: Run Sensitivity Analysis
```bash
python sensitivity_analysis.py
```
Expected output:
- `results/sensitivity_analysis.csv`
- `results/sensitivity_summary.txt`
- `figures/sensitivity_heatmap.png`

Estimated time: ~30 minutes

## Expected Results

### Main Replication (seed=42)

| Metric | Expected Value |
|--------|---------------|
| Best R² (AutoML-style) | 0.992 ± 0.001 |
| Best R² (All methods) | 0.996 ± 0.001 |
| Best feature combo | rpm + frp |
| Best model | ridge_poly_2 |

### Uncertainty Analysis

| Model | Expected R² (mean ± std) |
|-------|-------------------------|
| ridge_poly_2 | 0.996 ± 0.001 |
| mlp_large | 0.996 ± 0.002 |
| extra_trees | 0.992 ± 0.001 |
| random_forest | 0.988 ± 0.002 |

### Time-Series Validation

| Validation Method | Expected R² |
|-------------------|-------------|
| Random split | 0.988 ± 0.002 |
| Time-series CV | 0.984 ± 0.006 |
| Difference | ~0.004 |

## Verification Checksums

To verify reproducibility, compare file checksums:

```bash
md5 results/replication_results.csv
# Expected: [will vary by exact numpy version]

wc -l results/replication_results.csv
# Expected: 91 (90 data rows + header)

wc -l results/uncertainty_all_seeds.csv
# Expected: ~1201 (120 experiments × 10 seeds + header)
```

## Troubleshooting

### TPOT Installation Fails
TPOT may have dependency conflicts. Workaround:
```bash
pip install tpot --no-deps
pip install deap update-checker stopit
```
If still fails, skip TPOT and use baseline comparisons.

### LightGBM Installation Issues on macOS
```bash
brew install libomp
pip install lightgbm
```

### Memory Issues
If running out of memory:
- Reduce `n_samples` from 30000 to 10000
- Reduce `n_estimators` in tree models
- Run seeds sequentially instead of storing all results

## File Structure

```
ai-replication-study/
├── experiments/
│   ├── data_generator.py          # Data synthesis
│   ├── run_replication.py         # Main experiment
│   ├── modern_methods.py          # Extended comparison
│   ├── run_uncertainty_analysis.py # Multi-seed analysis
│   └── sensitivity_analysis.py    # Parameter sensitivity
├── results/
│   ├── replication_results.csv
│   ├── modern_methods_results.csv
│   ├── uncertainty_summary.csv
│   ├── sensitivity_analysis.csv
│   └── meta_log.json
├── figures/
│   ├── method_comparison.png
│   ├── feature_complexity.png
│   ├── results_heatmap.png
│   └── sensitivity_heatmap.png
├── paper/
│   └── ai_replication_study.tex
├── docs/
│   ├── agent_architecture.md
│   └── reproducibility_protocol.md
└── requirements.txt
```

## Contact

For questions about reproducibility:
- Open an issue on the repository
- Original study author: Fredrik Ahlgren

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2024-01 | 1.0 | Initial release |
| 2025-01 | 1.1 | Added uncertainty analysis, sensitivity analysis |
