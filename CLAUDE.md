# AI Replication Study: Claude Code Replicating Ship Fuel Consumption ML Research

## Project Overview

This is a **meta-study** documenting an AI system (Claude Code) replicating scientific research:
- **Original Paper:** "Auto Machine Learning for predicting Ship Fuel Consumption" (Ahlgren & Thern, ECOS 2018)
- **Meta-Objective:** Document and analyze AI-driven scientific replication

## Architecture

```
ai-replication-study/
├── data/               # Synthetic and processed data
├── experiments/        # Experiment scripts and notebooks
├── results/           # Experimental results and metrics
├── paper/             # IEEE-formatted paper
├── blog/              # Blog post about the meta-perspective
└── figures/           # Generated visualizations
```

## Methodology

1. **Data Synthesis:** Generate synthetic ship engine data matching original statistics
2. **AutoML Comparison:** TPOT (original), AutoGluon, scikit-learn baselines
3. **Reproducibility Analysis:** Compare results with original findings
4. **Meta-Analysis:** Document AI's approach to scientific replication

## Key Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn
- TPOT (original AutoML)
- autogluon (modern comparison)
- matplotlib, seaborn (visualization)
- LaTeX (IEEE paper generation)

## Running Experiments

```bash
cd experiments
python run_replication.py
```

## Original Paper Reference

Ahlgren, F., Thern, M. (2018). Auto Machine Learning for predicting Ship Fuel Consumption.
ECOS 2018 - 31st International Conference on Efficiency, Cost, Optimization,
Simulation and Environmental Impact of Energy Systems.
