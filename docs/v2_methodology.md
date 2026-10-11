# V2 Methodology Protocol

## Core principles

- Historical graph and temporal features are computed from the full,
  non-downsampled transaction history.

- Sampling affects model fitting rows only, never historical context.

- Train, validation, and test splits are chronological.

- Split rules are defined before inspecting fraud counts.

- Validation and test preserve the natural class prevalence.

- The final test set is not used for model selection, feature selection,
  threshold tuning, or split adjustment.

- Threshold-selection criteria are defined before evaluating thresholds.

- Historical features must use strictly past transactions only:

  historical_time < current_transaction_time

- Transactions occurring at the same time step must not see each other
  as historical context.

## Predefined evaluation protocol

- Scope: All CASH_OUT and TRANSFER transactions.
- Split: Chronological 70% train, 15% validation, 15% test.
- Boundaries: Approximate row proportions, respecting complete time_step groups.
- Split boundaries must be determined without inspecting fraud labels.
- Validation and test data must not be downsampled.
- Threshold selection: Maximize F1 on the validation set.
- The final holdout test is reserved for one final evaluation.

## Frozen PaySim temporal split

Scope: CASH_OUT and TRANSFER transactions.

Split boundaries were calculated using cumulative transaction
counts ordered by time_step, without inspecting fraud labels.

- Train: steps 1-322 (1,938,484 transactions)
- Validation: steps 323-376 (415,628 transactions)
- Test: steps 377-743 (416,297 transactions)

Target proportions: 70% / 15% / 15%.

The boundaries are frozen before examining fraud distribution.

Threshold selection objective: maximize validation F1.
