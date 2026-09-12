# Graph-Assisted Transaction Fraud Detection

A leakage-aware fraud detection pipeline combining **Neo4j graph features** with
traditional machine learning on the PaySim transaction dataset.

The project focuses not only on predictive performance, but also on identifying
dataset-specific shortcuts, preserving temporal order, and extracting historical
transaction behavior from a graph.

## Overview

The pipeline follows this workflow:

PaySim  
→ canonical transaction schema  
→ Neo4j transaction graph  
→ transaction-level features  
→ historical receiver graph features  
→ temporal graph features  
→ leakage-aware feature ablation  
→ temporal train / validation / test split  
→ Random Forest  
→ validation-based threshold tuning  
→ final holdout evaluation

The graph is represented as:

```text
(:Account)-[:SENT]->(:Transaction)-[:TO]->(:Account)

Transactions are modeled as nodes so that transaction attributes and historical
behavior can be queried directly from Neo4j.

Dataset

The project uses the synthetic PaySim mobile-money transaction dataset.

The original dataset contains approximately 6.36 million transactions.

Because fraud appears only in CASH_OUT and TRANSFER transactions, the
analysis focuses on those transaction types.

A controlled modeling dataset of 200,000 transactions was created:

8,213 fraudulent transactions
191,787 sampled non-fraudulent transactions
approximately 4.11% fraud prevalence

This sampling strategy is intended for experimentation and does not represent
real-world fraud prevalence.

Canonical Schema

Raw PaySim columns are converted to a reusable transaction schema including:

tx_id
time_step
tx_type
amount
src_id
src_balance_before
src_balance_after
dst_id
dst_balance_before
dst_balance_after
label_fraud
label_flagged

tx_id is an artificial identifier and is not used as a temporal variable.
Chronology is determined by time_step.

Graph Feature Engineering

Neo4j is used to derive historical receiver behavior using only transactions
that occurred before the current transaction.

Examples include:

receiver_prior_tx_count
receiver_prior_total_amount
receiver_prior_avg_amount
receiver_prior_7d_tx_count
receiver_prior_7d_total_amount

Historical queries enforce:

past_transaction.time_step < current_transaction.time_step

so the current transaction cannot leak into its own historical features.

Sender-side graph history was investigated but found to be extremely sparse in
the sampled dataset, so it was not retained in the final graph feature set.

Leakage and Synthetic-Data Artifacts

Exploratory analysis revealed unusually strong sender-balance patterns in PaySim.

Features such as:

src_balance_before
amount_to_src_balance_ratio
amount_exceeds_src_balance
src_balance_zero

produced strong predictive performance, but appeared closely tied to PaySim's
synthetic transaction-generation rules.

Post-transaction balances were also excluded from the primary model because they
may not be available at fraud-decision time.

Ablation experiments therefore compare models with and without the
sender-balance feature family.

The final model uses the cleaner feature set.

Temporal Evaluation

Random splitting is avoided.

Transactions are divided chronologically:

Training period: time_step <= 356
Final test:      time_step > 356

The training period is split again:

Train core:  time_step <= 302
Validation:  time_step > 302 and <= 356

Model development and threshold selection use only the train-core and validation
periods.

The final holdout test period is evaluated only after the threshold has been
selected.

Model Development

Logistic Regression is used to compare several feature groups:

transaction baseline
transaction + graph
clean transaction baseline
clean transaction + graph
clean transaction + temporal graph

A Random Forest is then trained on the clean temporal graph feature set to model
nonlinear interactions between transaction and historical receiver behavior.

Final Model

Final model:

RandomForestClassifier
n_estimators = 200
class_weight = balanced_subsample
random_state = 42

The fraud decision threshold is tuned on the validation period rather than using
the default probability threshold of 0.50.

The selected threshold is:

0.31
Final Holdout Results

Using the selected threshold on the reserved temporal test set:

Metric	Result
Precision	0.8138
Recall	0.4767
F1 Score	0.6012
Accuracy	0.9325

Final confusion matrix:

[[35029   463]
 [ 2221  2023]]

Compared with the default 0.50 threshold, the selected threshold detects more
fraud transactions while accepting a moderate increase in false positives.

Repository Structure
PAYSIM/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_graph_eda_and_feature_engineering.ipynb
│   ├── 04_modeling_and_ablation.ipynb
│   └── 05_threshold_tuning_and_evaluation.ipynb
│
├── data/
├── artifacts/
│   ├── final_metrics.csv
│   └── model_config.json
│
├── src/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md

Large datasets, generated prediction files, and serialized model files are
excluded from version control.

Installation
pip install -r requirements.txt

Core dependencies:

pandas
numpy
scikit-learn
neo4j
joblib

A local Neo4j instance is required for graph construction and graph feature
engineering.

Neo4j credentials are entered interactively and are not stored in the repository.

Limitations

This project is an experimental fraud-detection pipeline rather than a production
fraud system.

Important limitations include:

PaySim is synthetic.
The 200k modeling sample has a much higher fraud rate than real transaction systems.
Historical graph features are calculated from the sampled graph rather than the
full PaySim transaction history.
PaySim time_step values are interpreted as hourly intervals.
The current 7-day temporal window is therefore dataset-specific.
Future Work

The next stage is to make the pipeline dataset-independent by introducing:

configurable schema adapters
dataset validation
real timestamp-based temporal windows
configurable feature availability rules
graph history built from complete transaction history
reusable pipeline modules under src/
Purpose

The project demonstrates how graph-derived behavioral history can complement
traditional transaction features while maintaining temporal discipline and
actively testing for synthetic-data leakage.

