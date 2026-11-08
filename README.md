# House Price Prediction — Improved Microproject

This folder contains a reproducible experiment on the user-provided `modified_data.csv` (Washington State, May–July 2014). **It uses the two required algorithms only: Linear Regression and Decision Tree Regression.**

## How to run on Windows (PowerShell)

1. Extract the ZIP. Open the extracted `House_Price_Prediction_Improved` folder in VS Code.
2. Open Terminal > New Terminal.
3. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
explorer .\results
```

If you already have a Python environment, simply install requirements and run `python main.py`.

## What's new?

- The original Linear Regression and Decision Tree are retained as baselines.
- Numeric scaling is applied to the improved Linear Regression to reduce numerical conditioning issues.
- Log target transformation `log(1+price)` helps handle skewed prices. Predictions are transformed back using `expm1`.
- To limit extremely exaggerated predictions after reversing the log transformation, predictions can be capped at the **99.5th or 99.9th percentile of the training target prices**. Within cross-validation, the cap is calculated separately from each fold's training rows, never from the validation targets.
- The improved Decision Tree uses `criterion='absolute_error'`, making terminal leaves predict their median rather than mean, which is more robust to extreme prices. Depth and minimum leaf size are compared using five-fold CV on training rows.
- The script compares the variants using the mean five-fold **MAE**, chooses one Linear Regression and one Decision Tree configuration, then evaluates the chosen models and both originals on the same 20% test partition.

## Data hygiene and research integrity

- Invalid and zero prices (49 records) are excluded; 4,551 valid records remain.
- `price_per_sqft` is **never used** as input: it contains target-price leakage.
- Exact street address and date are not included in this baseline.
- No expensive properties are removed automatically. The `high_price_rows_to_verify.csv` report identifies suspicious extremes for checking with the original source.
- The dataset is from **2014 Washington State**, not current Indian real estate.
- This test split was already examined in the user's previous project run, so **the test figures are exploratory, not a new independent test**. Report CV scores as training-only model-selection evidence. Before making a strong generalization claim, obtain a genuinely untouched external or later-dated test dataset, or use nested cross-validation.
- A 99.5th-percentile prediction cap may strongly **underestimate genuine luxury properties**. Capping is a trade-off, not a generally valid way to improve house-price prediction. Inspect price-group errors and investigate high-value records before deploying.

## Output files

`results/training_cv_comparison.csv` — training-only five-fold scores for 7 candidate models.

`results/test_model_comparison_EXPLORATORY.csv` — original vs improved model performance on the previously examined test set.

`results/test_predictions.csv` — test predictions for each model.

`results/improved_actual_vs_predicted.png` — zoomed actual/predicted plots.

`results/model_improvement_comparison.png` — test MAE and R² comparison.

`results/training_cross_validation.png` — cross-validation MAE comparison.

`results/price_distribution.png` — typical house-price distribution.

`results/high_price_rows_to_verify.csv` — anomalous-price candidates to verify; not automatically removed.

`results/metadata.json` — configuration and integrity caveats.

## Complete Decision Tree output

When you run `python main.py`, the program also exports the **entire selected, fitted Decision Tree** to the `results` folder:

- `results/complete_decision_tree.svg` — complete vector diagram; zoom in without pixelation.
- `results/complete_decision_tree.pdf` — complete vector PDF.
- `results/open_complete_decision_tree.html` — browser viewer with zoom controls and scrollbars.
- `results/decision_tree_preview.png` — only the first four levels, for reports/slides.
- `results/complete_decision_tree_rules.txt` — all split rules in text form.

After the script finishes, open the complete visualization in Chrome or Edge with:

```powershell
start .\results\open_complete_decision_tree.html
```

The terminal will also print the total number of nodes, leaves, and depth. The full tree is too large to read at normal page scale; use the viewer's zoom slider and scrollbars, or open the SVG/PDF and zoom in.
