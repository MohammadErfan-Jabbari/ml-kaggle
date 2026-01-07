# Knowledge Management System

This directory stores accumulated knowledge, insights, and experimental findings for the AMR prediction competition.

## Structure

```
knowledge/
├── research/           # Domain knowledge and literature findings
│   ├── maldi_tof.md   # MALDI-TOF fundamentals and preprocessing
│   ├── amr_biology.md # AMR mechanisms by species
│   └── ml_sota.md     # State-of-the-art ML approaches
├── experiments/        # Experiment tracking and results
│   ├── experiment_log.md  # Chronological experiment log
│   └── results/       # Raw results, model outputs
├── insights/          # Key insights and learnings
│   └── data_insights.md   # EDA findings and data characteristics
└── hypotheses/        # Hypothesis tracking
    └── hypothesis_tracker.md  # All hypotheses, predictions, outcomes
```

## Hypothesis-Driven Development Workflow

1. **Formulate Hypothesis**: Based on research and data insights
2. **Design Experiment**: Minimal change to test hypothesis
3. **Implement & Run**: Track with MLflow
4. **Evaluate**: Compare against baseline
5. **Reflect**: Document learnings, update knowledge
6. **Iterate**: Next hypothesis

## Session Continuity

Each session should:
1. Read `hypotheses/hypothesis_tracker.md` for current state
2. Continue from last hypothesis or start next
3. Update tracker with new findings
4. Document any new insights discovered
