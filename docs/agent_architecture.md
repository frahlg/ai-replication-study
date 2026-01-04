# Agent Architecture Specification

This document addresses Reviewer Question #1: "What, precisely, was the agent
architecture (tools, prompts, planning algorithm, allowed web/code access),
and what interventions did the human supervisor perform?"

## System Identification

- **Agent System**: Claude Code (Anthropic's CLI for Claude)
- **Model**: Claude Opus 4.5
- **Model ID**: `claude-opus-4-5-20251101`
- **Knowledge Cutoff**: January 2025

## Tools Available

The agent had access to the following tools:

| Tool | Purpose | Usage in This Study |
|------|---------|---------------------|
| `Read` | Read files from filesystem | Read original paper, code, notebooks |
| `Write` | Create new files | Create experiment scripts, results |
| `Edit` | Modify existing files | Refine scripts iteratively |
| `Bash` | Execute shell commands | Run Python scripts, pip install |
| `Glob` | Find files by pattern | Locate relevant source files |
| `Grep` | Search file contents | Search for specific code patterns |
| `WebSearch` | Search the web | Find related papers, benchmarks |
| `WebFetch` | Fetch web content | Download documentation |

## Agent Workflow

### Phase 1: Understanding (estimated 10 minutes)
1. Read the original paper (Ahlgren & Thern 2018)
2. Search for the original code repository
3. Analyze the notebook structure and data format
4. Identify key statistical properties from documentation

### Phase 2: Data Synthesis (estimated 15 minutes)
1. Design physics-informed data generator
2. Implement `ShipEngineDataGenerator` class
3. Calibrate parameters from documented statistics
4. Validate generated data properties

### Phase 3: Experimentation (estimated 15 minutes)
1. Implement baseline models (sklearn)
2. Run 90 experiments (15 feature combos × 6 models)
3. Extend to modern methods (neural networks, gradient boosting)
4. Implement time-series validation

### Phase 4: Analysis (estimated 5 minutes)
1. Compare results to original paper
2. Generate critical analysis
3. Create visualizations
4. Document findings

**Total Autonomous Time**: ~45 minutes

## Human Supervisor Role

The human supervisor (Fredrik Ahlgren, original paper author) performed:

1. **Initial Prompt**: Provided the research question and context
   - "Replicate the 2018 ship fuel consumption AutoML study"
   - Pointed to the original paper and repository

2. **Guidance Checkpoints**: Approved direction at key decision points
   - Confirmed synthetic data approach when original data unavailable
   - Approved model selection strategy

3. **Final Review**: Reviewed outputs before committing
   - Verified code correctness
   - Approved final paper draft
   - ~15 minutes total review time

**Total Human Time**: ~15 minutes

## Decision Log

### Why These 14 Models?

**Decision**: Test models from Table I in the paper.

**Rationale**:
- 2018-era models: Available to original authors
  - Linear, Ridge, ElasticNet (baselines)
  - Random Forest, Gradient Boosting, Extra Trees (TPOT components)
- Modern comparisons: What has improved since 2018?
  - HistGradientBoosting (sklearn 0.21+, 2019)
  - LightGBM (became popular post-2018)
- Neural networks: Unexplored in original
  - MLP variants (available in sklearn since 2007, but not used)
- Feature engineering: Simple alternative to AutoML
  - Polynomial features (degree 2, 3)

### Why Polynomial Degrees 2 and 3?

**Decision**: Test quadratic and cubic polynomial features.

**Rationale**:
- Degree 2: Captures pairwise interactions (common in physics)
- Degree 3: Tests diminishing returns hypothesis
- Higher degrees: Risk overfitting, too many features

**Result**: Degree 2 optimal (R² = 0.9966), degree 3 marginal improvement.

### Why These Hyperparameters?

**Decision**: Use sklearn defaults with minor adjustments.

**Rationale**:
- Goal was methodological comparison, not hyperparameter tuning
- TPOT in original paper performed automated tuning
- Fair comparison uses reasonable defaults
- Notable adjustments:
  - `n_estimators=100` for tree models (balance speed/performance)
  - `max_iter=500-1000` for MLPs (ensure convergence)
  - `early_stopping=True` for MLPs (prevent overfitting)

### Why Synthetic Data Instead of Original?

**Decision**: Generate physics-informed synthetic data.

**Rationale**:
- Original data proprietary (ship operator confidentiality)
- No public archive available
- Opportunity to test: "Is documentation sufficient for reproduction?"
- Physics-informed approach preserves relationships

**Limitation**: Tests documentation sufficiency, not empirical reproducibility.

## Agent Limitations

What the agent could NOT do autonomously:

1. **Access proprietary data**: Required human confirmation that data unavailable
2. **Contact external parties**: Could not email original data owners
3. **Install problematic packages**: TPOT installation failed, reported to human
4. **Make publication decisions**: Human approved final paper content
5. **Validate domain assumptions**: Relied on documented physics relationships

## Reproducibility Considerations

To reproduce this agent workflow:

1. Use Claude Code with claude-opus-4-5 or later
2. Provide same initial prompt
3. Allow tool access to: Read, Write, Edit, Bash, Glob, Grep
4. Expect ~45 minutes autonomous execution
5. Budget ~15 minutes human review

Note: Results may vary due to:
- Model non-determinism in generation
- Package version differences
- Different random seeds (if not fixed)

## Audit Trail

Key artifacts generated by agent:

| Artifact | Purpose | Location |
|----------|---------|----------|
| `data_generator.py` | Synthetic data generation | `experiments/` |
| `run_replication.py` | Main experiment runner | `experiments/` |
| `modern_methods.py` | Extended comparisons | `experiments/` |
| `replication_results.csv` | Raw experiment results | `results/` |
| `meta_log.json` | Execution log | `results/` |
| `ai_replication_study.tex` | Paper draft | `paper/` |

## Comparison to Human Replication

| Aspect | AI Agent | Human Researcher |
|--------|----------|------------------|
| Time to replicate | ~1 hour | Days to weeks |
| Documentation reviewed | Complete | Often partial |
| Methods tested | 14 systematically | Subset, incrementally |
| Bias in selection | None (tested all) | Confirmation bias possible |
| Domain expertise | Limited to documented | Deep tacit knowledge |
| Novel insights | Pattern-based | Conceptual |

## Addressing Reviewer Concerns

**Q**: "Can you provide a reproducible protocol and logs?"
**A**: See `docs/reproducibility_protocol.md` and `results/meta_log.json`

**Q**: "What interventions did the human supervisor perform?"
**A**: Initial prompt, 2 checkpoint approvals, final review (~15 min total)

**Q**: "Why were specific choices made?"
**A**: See Decision Log above - each choice documented with rationale
