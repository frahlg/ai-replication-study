# When AI Replicates Science

**A meta-study on AI-driven scientific replication: Claude Code autonomously reproducing machine learning research**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue)](paper/ai_replication_study.pdf)
[![Original](https://img.shields.io/badge/Original-ECOS%202018-green)](paper/original_ahlgren_thern_2018.pdf)
[![Blog Post](https://img.shields.io/badge/Blog-Read%20More-orange)](blog/meta-study-blog-post.md)

---

## What Is This?

I asked Claude Code to replicate one of my old academic papers. This repository documents what happened.

**Original Paper:** [Auto Machine Learning for predicting Ship Fuel Consumption](paper/original_ahlgren_thern_2018.pdf) (Ahlgren & Thern, ECOS 2018)

**Result:** The AI successfully replicated the results, then found that a 1970s technique (polynomial features + Ridge regression) beats the 2018 AutoML approach.

This isn't just about ship fuel. It's about what happens to engineering research when AI can systematically verify and critique published work.

---

## Key Findings

| Discovery | Original (2018) | AI Replication (2026) |
|-----------|-----------------|----------------------|
| Best R² (AutoML-style) | 0.992 | 0.9924 |
| Best R² (All methods) | — | **0.9966** |
| Best method | TPOT (AutoML) | Ridge + Polynomial features |
| Time to replicate | — | 21 minutes |

**The uncomfortable findings:**
- Simple polynomial features (1970s technique) beat complex AutoML
- Neural networks worked but were never tried in the original
- Modern gradient boosting (LightGBM, etc.) provides zero improvement over 2018 methods
- Random train/test splits inflated results by ~0.5% vs proper time-series CV

---

## Repository Structure

```
ai-replication-study/
├── paper/
│   ├── original_ahlgren_thern_2018.pdf   # Original ECOS 2018 paper
│   ├── ai_replication_study.tex          # Meta-study (IEEE format)
│   └── ai_replication_study.pdf          # Compiled paper (5 pages)
├── blog/
│   └── meta-study-blog-post.md           # Narrative blog post
├── experiments/
│   ├── data_generator.py                 # Physics-informed synthetic data
│   ├── run_replication.py                # Original methodology replication
│   └── modern_methods.py                 # Extended comparison (14 methods)
├── results/
│   ├── replication_results.csv           # 90 experiments (15 combos × 6 models)
│   ├── modern_methods_results.csv        # Modern methods comparison
│   ├── comparison_report.txt             # Summary vs original
│   └── critical_analysis.txt             # AI-generated critique
├── figures/
│   ├── method_comparison.png             # Model performance boxplot
│   ├── feature_complexity.png            # R² vs feature count
│   └── results_heatmap.png               # Full results matrix
└── pyproject.toml                        # Python dependencies (uv)
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/frahlg/ai-replication-study.git
cd ai-replication-study

# Install dependencies (requires uv)
uv sync

# Run the original replication
cd experiments
uv run python run_replication.py

# Run modern methods comparison
uv run python modern_methods.py
```

---

## The Meta-Question

> **What happens to engineering research when AI can autonomously replicate studies?**

This project explores:

1. **Accelerated verification** — 21 minutes vs days/weeks for human replication
2. **Documentation standards** — AI exposes gaps humans miss
3. **Living methodology** — Historical results get continuous modern context
4. **Critique as feature** — AI naturally tests alternatives and finds weaknesses

---

## Methods Tested

### 2018-Era (Available when original was written)
- Linear Regression, Ridge, ElasticNet
- Random Forest, Gradient Boosting, Extra Trees
- XGBoost
- MLP Neural Networks
- Polynomial Feature Engineering

### Modern (Post-2018)
- LightGBM
- HistGradientBoosting
- Larger neural architectures

**Finding:** Modern methods provide negligible improvement. For tabular data, the field has matured.

---

## Original Paper

**Citation:**
```
Ahlgren, F., Thern, M. (2018). Auto Machine Learning for predicting Ship Fuel Consumption.
ECOS 2018 - 31st International Conference on Efficiency, Cost, Optimization,
Simulation and Environmental Impact of Energy Systems. Guimarães, Portugal.
```

**Abstract:** The paper applied TPOT (a genetic algorithm-based AutoML tool) to predict fuel oil consumption on a Baltic Sea cruise ship using engine sensor data (RPM, fuel rack position, exhaust temperature, turbocharger RPM). Achieved R² = 0.992.

**Original code:** [github.com/frahlg/ML-dyn](https://github.com/frahlg/ML-dyn)

---

## Meta-Study Paper

The [full IEEE-formatted paper](paper/ai_replication_study.pdf) (5 pages) documents:

- Methodology for AI-driven replication
- Comprehensive results across 14 methods
- Critical analysis of original and replication
- Discussion of implications for engineering research
- 12 academic references

---

## Blog Post

For a narrative version, read the [blog post](blog/meta-study-blog-post.md):

> "I asked an AI to replicate one of my old papers. What happened next made me rethink how engineering research works."

---

## The Self-Critique

Watching an AI dissect my own paper forced uncomfortable reflection:

- **We were seduced by novelty.** TPOT was exciting in 2018. We framed the problem as "model selection" because that's what the tool did.
- **We didn't try obvious alternatives.** Neural networks and polynomial features existed. We didn't test them.
- **We validated incorrectly.** Random splits for time-series data is a known anti-pattern.

This is how research actually works. We make choices—some good, some expedient, some wrong. Normally, nobody checks.

AI replication checks.

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management
- LaTeX (tectonic) for paper compilation

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Authors

- **Claude Code** (Anthropic) — Autonomous experiments and analysis
- **Fredrik Ahlgren** — Human supervision and original research

*The original 2018 research was conducted with Marcus Thern at Lund University.*

---

## Citation

If you use this work:

```bibtex
@misc{ai_replication_2026,
  title={When AI Replicates Science: A Meta-Study on LLM Agents Reproducing ML Research},
  author={Claude Code and Ahlgren, Fredrik},
  year={2026},
  howpublished={\url{https://github.com/frahlg/ai-replication-study}},
  note={AI-driven replication study conducted by Claude Code (Anthropic)}
}
```
