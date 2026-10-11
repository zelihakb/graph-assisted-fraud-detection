# PaySim — V2 Development Roadmap

## Project Status

**V1.0.0 — Controlled-Sample MVP: Complete**

The existing implementation uses 200,000 sampled CASH_OUT/TRANSFER transactions, Neo4j-based historical features, temporal evaluation, and Random Forest classification.

V1 remains preserved as a tagged milestone. Its reported metrics describe the controlled-sample evaluation, not natural-prevalence deployment performance.

**V2 Goal:** Transform the existing MVP into a more trustworthy, reproducible, and eventually dataset-generalizable fraud detection pipeline.

## Implementation Rule

Reuse the existing V1 implementation wherever appropriate. Do not repeat completed EDA, graph analysis, or model experiments unless a methodological change requires recomputation.

Reference notebooks:

| Notebook                             | V2 responsibility                           |
| ------------------------------------ | ------------------------------------------- |
| 01 — Data Preparation                | Full-data ingestion and train-only sampling |
| 02 — Graph Construction              | Full-history graph construction             |
| 03 — Graph EDA & Feature Engineering | Strict-past graph and temporal features     |
| 04 — Modeling & Ablation             | Model training and controlled comparisons   |
| 05 — Threshold & Evaluation          | Threshold tuning and final reporting        |

New reusable implementations belong in `src/`. Notebooks remain for diagnostics, experimentation, and visualization.

## V2.0 — Methodological Repair

### Completed

- [x] Preserve V1 as `v1.0.0`.
- [x] Create a dedicated V2 development branch.
- [x] Establish `docs/v2_methodology.md`.
- [x] Predefine chronological 70/15/15 splitting without inspecting fraud labels.
- [x] Freeze PaySim split boundaries:
  - Train: steps 1–322
  - Validation: steps 323–376
  - Test: steps 377–743
- [x] Define validation F1 maximization as the initial threshold-selection objective.

### Next: Data Diagnostics

- [ ] Report fraud counts and natural prevalence in each frozen partition.
- [ ] Inspect temporal fraud distribution without modifying split boundaries.
- [ ] Document PaySim hourly step semantics using the primary publication.

### Full-History Feature Engineering

- [ ] Reference V1 notebooks 02 and 03.
- [ ] Separate the complete historical transaction context from model training rows.
- [ ] Build reusable full-history graph/temporal feature functions.
- [ ] Enforce strict-past filtering and same-step exclusion.
- [ ] Preserve all scoped historical transactions, regardless of later training downsampling.
- [ ] Address full-data processing and memory constraints.

**Core principle: Sampling affects model fitting rows, never historical context.**

### Evaluation Protocol

- [ ] Apply downsampling only within training data, if required.
- [ ] Preserve natural prevalence in validation and test.
- [ ] Add automated leakage and temporal consistency checks.
- [ ] Reuse V1 modeling and threshold-tuning logic where valid.
- [ ] Compare Logistic Regression and Random Forest baselines under consistent conditions.
- [ ] Perform temporal robustness analysis on development data.
- [ ] Freeze model, features, and threshold before final test evaluation.
- [ ] Evaluate the frozen V2 test partition once for final reporting.

### Reporting

- [ ] PR-AUC, ROC-AUC, precision–recall curve.
- [ ] Confusion matrix and precision/recall/F1.
- [ ] Time-aware uncertainty estimates, with bootstrap assumptions documented.
- [ ] Permutation feature importance.
- [ ] Transparent V1 versus V2 comparison.

**Evaluation caveat:** The V2 test time range overlaps the previously evaluated V1 holdout. It is excluded from V2 model fitting and tuning, but should not be represented as completely unseen across the project's history.

## V2.1 — Engineering & Reproducibility

- [ ] Consolidate reusable code into `src/`.
- [ ] Unit and integration tests.
- [ ] Configuration and artifact management.
- [ ] Dockerized Neo4j setup.
- [ ] Reproducible data/graph initialization.
- [ ] GitHub Actions CI.
- [ ] Improved setup documentation.

Core modules required by V2.0 should be implemented during V2.0 rather than postponed to this phase.

## V2.2 — Dataset Generalization

- [ ] Define and validate the canonical transaction schema.
- [ ] Create a PaySim dataset adapter.
- [ ] Select a second real transaction dataset.
- [ ] Implement an adapter for the second dataset.
- [ ] Support real timestamps and configurable time windows.
- [ ] Define behavior for missing features and optional labels.
- [ ] Demonstrate reuse of the core pipeline without PaySim-specific changes.

## V2.3 — Extensions

- [ ] Additional recency and velocity features.
- [ ] Gradient boosting model benchmarks.
- [ ] Extended temporal validation.
- [ ] Calibration assessment if probability-based decisions are required.
- [ ] Cost-sensitive or investigation-budget threshold strategies.

## Optional Research

Graph embeddings, advanced network representations, and graph neural networks are explicitly outside the immediate scope.

## Development Workflow

For each task:

1. Identify the relevant V1 implementation.
2. Explain what can be reused and what must change.
3. Implement the smallest meaningful improvement.
4. Validate it against the frozen methodology.
5. Document results and limitations.
6. Commit and merge completed milestones.

The objective is not to maximize the number of experiments, but to build a reusable and defensible pipeline.
