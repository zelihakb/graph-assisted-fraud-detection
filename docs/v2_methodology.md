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
