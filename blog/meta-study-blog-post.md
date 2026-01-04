# When AI Replicates Science: What I Learned Watching Claude Reproduce My Research

*January 4, 2026*

I asked an AI to replicate one of my old papers. What happened next made me rethink how engineering research works.

---

## The Setup

In 2018, I published a paper at ECOS on using AutoML to predict ship fuel consumption. The paper used TPOT—a genetic algorithm that evolves machine learning pipelines—and achieved R² = 0.992 on sensor data from a Baltic Sea cruise ship. Standard academic fare.

Seven years later, I gave Claude Code a simple prompt: *"Replicate this paper."*

The original data wasn't available. The computational environment from 2018 is gone. All that remained was the paper itself and some Jupyter notebooks in a dusty GitHub repo.

What followed wasn't just a successful replication. It was a mirror held up to how we do engineering research—and a glimpse of what's coming.

---

## What the AI Actually Did

### Phase 1: Understanding (5 minutes)

Claude didn't just read the paper. It reverse-engineered it:

- Fetched the abstract from DIVA portal
- Cloned the GitHub repository
- Analyzed output cells in notebooks to extract statistical properties
- Inferred physical relationships from variable naming conventions

This is the kind of forensic reconstruction that takes a human researcher an afternoon. The AI did it systematically, extracting means, standard deviations, and correlations from whatever documentation existed.

### Phase 2: Physics-Informed Data Generation (10 minutes)

Here's where it got interesting. Without the original ship sensor data, Claude built a synthetic data generator. But not randomly—it reasoned about ship engine physics:

> "Ship engines operate in bimodal states—either running at operational speed or turned off. Fuel rack position correlates with RPM because more fuel is injected under load. Exhaust temperature follows from combustion dynamics..."

This is domain knowledge I never explicitly provided. The AI inferred it from variable names, code patterns, and general knowledge of diesel engines. It then generated 30,000 samples matching the statistical properties documented in our notebooks.

**Question this raises:** If an AI can reconstruct plausible data from documentation, what does that say about reproducibility? Is "reproduce from description" a valid test?

### Phase 3: Systematic Experimentation (6 minutes)

Claude ran 90 experiments across 15 feature combinations, testing:
- Linear regression (our baseline)
- Ridge, ElasticNet (regularized variants)
- Random Forest, Gradient Boosting, Extra Trees (2018 ensembles)
- XGBoost, LightGBM, HistGradientBoosting (modern boosting)
- Neural networks (MLP with varying architectures)
- Polynomial feature engineering

It also ran time-series cross-validation to check whether our random train/test split was methodologically sound.

Total time from prompt to results: **21 minutes**.

---

## The Results That Made Me Uncomfortable

### Replication: Success

| Metric | Original (2018) | Replication (2026) |
|--------|-----------------|-------------------|
| Best R² (AutoML-style) | 0.992 | 0.9924 |
| Linear Baseline | 0.957 | 0.9844 |
| Best Feature Combo | rpm + frp | rpm + frp |

The AI matched our results. Replication successful. But that's not the interesting part.

### The Critique That Stung

Claude didn't stop at reproduction. It tested alternatives we never considered:

| Method | Era | R² | What It Means |
|--------|-----|-----|---------------|
| **Ridge + Polynomial(d=2)** | 1970s | **0.9966** | Beats everything |
| MLP (3 layers) | 2018 | 0.9962 | We never tried this |
| Extra Trees | 2018 | 0.9924 | What TPOT found |
| XGBoost | 2018 | 0.9912 | Available, unused |
| LightGBM | 2022 | 0.9912 | Modern = no better |

**Finding 1: A 1970s technique beat our 2018 AutoML.**

Polynomial feature engineering with Ridge regression—available since forever—achieved R² = 0.9966. That's better than our complex genetic-algorithm-optimized pipeline. The method is interpretable, fast (0.01 seconds vs minutes), and was fully available when we wrote the paper.

We focused on model selection. We should have tried feature engineering.

**Finding 2: Neural networks worked. We never tried them.**

MLPRegressor has been in scikit-learn since 2007. It achieves R² = 0.9962 on our problem. Eleven years before our paper, this method existed. We didn't consider it.

**Finding 3: Modern methods provide zero improvement.**

LightGBM (2017), HistGradientBoosting (2019)—the gradient boosting variants that emerged after our paper—achieve the same R² = 0.9912 as 2018 XGBoost. For tabular data prediction, the field has hit diminished returns.

**Finding 4: Our validation was slightly optimistic.**

| Validation Method | R² |
|-------------------|-----|
| Random 75/25 split | 0.9882 |
| Time-series CV | 0.9836 ± 0.006 |

Using random splits on time-series data inflated our results by about 0.5%. Not enough to invalidate the findings, but a methodological weakness we didn't acknowledge.

---

## The Meta-Question

The ship fuel consumption problem is solved. That's not what matters here.

What matters is: **What happens to engineering research when AI can do this?**

### 1. Accelerated Verification

Claude replicated, extended, and critically analyzed a published study in 21 minutes. Traditional human replication takes days to weeks. If AI can systematically verify published findings at this speed, the reproducibility crisis gets a scalable solution.

### 2. Documentation as Testable Artifact

The replication revealed specific gaps in our documentation:
- How did we handle negative sensor values? (Inferred from code: clipped to zero)
- What resampling did we apply? (Inferred: 15-minute intervals, mean aggregation)
- Why those specific feature combinations? (Never documented)

If an AI can't reproduce your results from your paper, neither can most human researchers. AI replication could establish minimum documentation standards.

### 3. Living Methodology

Our 2018 paper is now situated in the 2026 methodological landscape. The AI didn't just reproduce—it contextualized. This kind of continuous re-evaluation is impossible with static publications but natural for AI agents.

### 4. Critique as Feature

The most valuable output wasn't "yes, we reproduced your R² = 0.992." It was:
- "Polynomial features beat your AutoML"
- "Neural networks were viable but unexplored"
- "Your validation methodology was slightly flawed"

AI replication naturally generates critical analysis. This is precisely what replication studies should do.

---

## What I Got Wrong (A Self-Critique)

Watching an AI dissect my own paper forced uncomfortable reflection:

**We were seduced by novelty.** TPOT was new and exciting in 2018. AutoML was the hot topic. We framed the problem as "model selection" because that's what the tool did. A more rigorous approach would have asked: "What's the simplest method that works?"

**We didn't try obvious alternatives.** Neural networks existed. Polynomial features existed. We didn't test them. Not because they were inappropriate, but because they weren't the story we wanted to tell.

**We validated incorrectly.** Random train/test splits for time-series data is a known anti-pattern. We did it anyway. The impact was small, but the methodology was wrong.

This is how research actually works. We make choices. Some are good, some are expedient, some are wrong. Normally, nobody checks.

AI replication checks.

---

## The Limitations (Being Honest)

This replication has weaknesses:

**Synthetic data isn't real data.** Claude generated plausible data from documentation. This tests reproducibility from *description*, not from *data*. These are related but distinct concepts. Real ship engines have phenomena that no generator captures.

**AI interpretation isn't human interpretation.** Some choices Claude made (noise parameters, operating fractions) required inference from incomplete documentation. The AI's interpretation may differ from what we intended.

**This was an easy case.** We had code, notebooks with outputs, a clear methodology. Many papers provide only prose. That's a harder replication problem.

---

## Where This Goes

I see several implications:

**For peer review:** What if reviewers could run AI replication as part of evaluation? "Your paper claims X. Our AI achieved 0.8X. Please explain the gap."

**For documentation standards:** Journals could require that papers pass an "AI reproducibility test." If Claude can't replicate it from your materials, revise until it can.

**For literature review:** Imagine AI agents systematically replicating papers in a field, generating a reproducibility landscape. Which results hold up? Which don't? Which were never tested against obvious alternatives?

**For research itself:** If AI handles systematic verification, humans can focus on creativity, domain expertise, and judgment. The division of labor shifts.

---

## The Strange Loop

This blog post—like the paper it describes—was produced through human-AI collaboration.

Claude ran the experiments autonomously. I provided the original paper and said "replicate this." The AI identified documentation gaps I'd forgotten about, tested methods I hadn't considered, and critiqued my validation methodology.

Then I wrote about it. And Claude helped with that too.

This is the strange loop we're entering: AI that can do research, analyzed by AI-assisted writing, about AI doing research. The meta-level keeps rising.

I don't know where this ends. But I know that watching an AI find the flaws in my old paper—flaws I never saw—changed how I think about engineering research.

The reproducibility crisis is real. AI-driven replication might be part of the answer.

---

## Technical Details

**Original Paper:**
> Ahlgren, F., Thern, M. (2018). Auto Machine Learning for predicting Ship Fuel Consumption. ECOS 2018, Guimarães, Portugal.

**Replication Repository:**
```
ai-replication-study/
├── experiments/
│   ├── data_generator.py      # Physics-informed synthetic data
│   ├── run_replication.py     # Original methodology replication
│   └── modern_methods.py      # Extended comparison
├── results/
│   ├── replication_results.csv
│   └── modern_methods_results.csv
├── paper/
│   └── ai_replication_study.pdf  # Full IEEE paper
└── figures/
    └── *.png                     # Visualizations
```

**Key Numbers:**
- Samples: 30,000
- Features tested: 15 combinations
- Models tested: 14
- Time to replicate: 21 minutes
- Best R² achieved: 0.9966 (polynomial features + Ridge)

---

*Fredrik Ahlgren, January 2026*

*This post documents a meta-study conducted with Claude Code (Anthropic). The experiments were autonomous; the interpretation was collaborative. The original research was conducted with Marcus Thern at Lund University.*
