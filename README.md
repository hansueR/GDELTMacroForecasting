# GDELT Macro Forecasting

This project studies whether GDELT global event data can improve short-horizon macro-asset return forecasting.

We forecast future return distributions for three assets:

- Gold
- QQQ
- WTI Oil

The forecasting horizons are 1 day, 3 days, and 5 days.

## Method

The pipeline first aligns daily market prices with GDELT event features under a no-look-ahead rule. Then it builds price features, event features, and sequence inputs for forecasting.

The project compares several baselines:

| Name | Description |
|---|---|
| NV | Naive historical-mean forecast |
| EW | EWMA forecast |
| RP | Ridge regression with price features |
| RE | Ridge regression with event features |
| RPE | Ridge regression with price + event features |
| TB | Two-branch neural model |

The main model uses one branch for price sequences and one branch for event sequences. The two branches are combined to predict Gaussian return-distribution parameters.

## Metrics

- **NLL**: negative log-likelihood. Lower is better.
- **AUC**: directional AUC based on the predicted return mean. Higher is better.
- **MAE / RMSE**: used for auxiliary risk forecasting.

## Main Scripts

```text
src/6-walk_forward_risk_baseline.py
src/7-walk_forward_distribution_baseline.py
src/9-build_sequence_dataset.py
src/10-train_two_branch_distribution_model.py
src/13-summarize_ablation_results.py
src/14-build_report_summary_tables.py
````

## Run

Run the main scripts with the Slurm files in `src/scripts/`:

```bash
sbatch src/scripts/7-walk_forward_distribution_baseline.sh
sbatch src/scripts/9-build_sequence_dataset.sh
sbatch src/scripts/10-train_two_branch_distribution_model.sh
sbatch src/scripts/13-summarize_ablation_results.sh
sbatch src/scripts/14-build_report_summary_tables.sh
```

## Outputs

Results are saved under `results/`, including baseline results, two-branch model results, ablation tables, report summary tables, and figures.

## Goal

The goal is to test whether structured event information helps return-distribution forecasting, and to analyze when event features improve or hurt forecasting performance.
