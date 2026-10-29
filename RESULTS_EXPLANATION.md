# Experiment observations — user-provided `modified_data.csv`

## Data cleaning
4,600 rows, 18 original columns. Zero missing values and zero exact duplicates. Removed 49 records with `price == 0`, leaving 4,551 property records. Shared 80/20 split: 3,640 training rows, 911 test rows.

## Important leakage prevention
`price_per_sqft` equals `price / sqft_living` (rounded), so it directly encodes the target. It must not be a feature when `price` is the predicted quantity. `street` and `date` were excluded in this initial model. `city` and `statezip` are safely encoded using training-only preprocessing pipelines.

## Results (test set)
| Model | MAE (USD) | RMSE (USD) | R² | Median absolute error (USD) |
|---|---:|---:|---:|---:|
| Linear Regression | 139,638.89 | 232,409.07 | 0.637 | 92,667.40 |
| Decision Tree Regression | 148,648.65 | 279,707.32 | 0.474 | 88,523.18 |

Decision Tree tuned on training folds only: `max_depth=12`, `min_samples_leaf=10`.

**Interpretation:** On this particular 20% hold-out set, Linear Regression achieved better MAE, RMSE and R². The Decision Tree had a slightly smaller *median* absolute error, which measures a different aspect of performance. Neither model is validated for modern property valuation or another region. Extreme high-value properties may have a disproportionate influence on RMSE; a future sensitivity analysis can test the effects of such observations while avoiding ad hoc cherry-picking.

## Results locations
Use the CSV and image files in `results/` for tables and figures in your final microproject report. Do not state these numbers were observed before running the code.
