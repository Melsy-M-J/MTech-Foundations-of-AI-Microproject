# House Price Prediction — Microproject with Your CSV

## Goal
Compare **Linear Regression** and **Decision Tree Regression** using the actual uploaded `modified_data.csv`. The target is **individual property price in US dollars**, for housing data in the Seattle/Washington State region dated **May–July 2014**. This is historical data, **not current Indian property values**.

## Dataset information
- 4,600 rows and 18 original columns.
- Features used: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `city`, `statezip`.
- Target: `price` (US dollars).
- 49 rows with zero recorded sale price are omitted; **4,551** valid positive-price rows remain.
- No missing values or exact duplicate rows were found in this file.
- `price_per_sqft` is **not used** as a feature: it is literally calculated using the answer (`price / sqft_living`), and would leak the target into both models.
- `street` is excluded as a near-unique location identifier; `date` is excluded from the initial baseline. The final analysis should say these choices were made.
- Large prices (including a $26.59 million entry) remain, because they were not independently verified as errors. These can strongly influence error metrics; discuss this limitation rather than silently dropping the entries.

## Setup on Windows (PowerShell)
1. Extract this ZIP and open the `House_Price_Prediction_Custom` folder in VS Code.
2. In Terminal > New Terminal, run the following commands **one at a time**:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If `py` is unavailable, replace it with `python` on the first command.

The dataset is included under `data/modified_data.csv`, so an internet connection is **not required** to load it. To use a new compatible CSV, run `python main.py --csv path/to/new.csv` after activating the environment or use `.\.venv\Scripts\python.exe main.py --csv path\to\new.csv`.

## Methodology (what we actually do)
1. Load CSV and inspect shape, price distribution and missing values.
2. Remove exact duplicates and rows with zero/missing/negative `price`.
3. Select 12 numeric and 2 categorical input columns, excluding the leakage column.
4. Split into training (80%) and test (20%) records with `random_state=42`.
5. For each model, use a **scikit-learn pipeline**: impute numeric values with training medians and one-hot-encode `city` / `statezip` with unknown-category handling.
6. Fit ordinary Linear Regression as the baseline.
7. Tune the Decision Tree over `max_depth` = 4, 6, 8 or 12 and `min_samples_leaf` = 5, 10 or 20, using **5-fold cross-validation on training data only** and selecting best mean absolute error.
8. Test both on **the same untouched test set**. Compute MAE, RMSE, R² and median absolute error.
9. Save real results and charts. Note the actual-vs-predicted chart is *zoomed to improve readability*, whereas all evaluation metrics use **every test sample**.

## Files in results/
- `model_comparison.csv`: measured MAE, RMSE, R², Median AE for both models
- `test_predictions.csv`: actual and predicted prices for every test record
- `dataset_summary.csv`: dataset descriptive statistics
- `actual_vs_predicted.png`: scatter plots, appropriately labelled with zoom
- `error_comparison.png`: test MAE/RMSE bar chart
- `correlation_heatmap.png`: Pearson correlations on numerical fields
- `decision_tree_preview.png`: first four levels of trained tree
- `feature_importance.png`: tree top feature importances
- `tree_feature_importances.csv`: full encoded-feature importance table
- `run_metadata.json`: exact experiment settings, features and cleaning counts

## Important for your abstract / proposed methodology
Your original abstract said **dataset not selected and experiments not performed**. This is no longer accurate once you run this project. In the *final report*, replace it with this dataset's geographic scope and these measured results; **do not claim that your initial abstract was a results paper**.

## Findings from the included run (do not copy without understanding)
See `results/model_comparison.csv` for the exact numbers, reproduced on this file in the current environment. Scores can vary with library revisions, platform or future code changes. The model with the smallest MAE/RMSE is preferable on those metrics; performance on other datasets is not established.

## Viva questions to be able to answer
- What is the target? **Sale price** in US dollars.
- Why are there 4,551 rows after cleaning? **49 prices were zero**.
- Why exclude `price_per_sqft`? **Because it leaks the target into inputs.**
- Why use OneHotEncoder? To convert categorical `city` and `statezip` into model-compatible numeric indicators.
- How did we avoid leakage during preprocessing / tree tuning? Everything was fitted only using training folds; the final test set was untouched until evaluation.
- Why MAE/RMSE/R²? All are suitable for a numerical regression target; RMSE is especially sensitive to large errors.
- Why can't these predictions represent today's Kerala market? The dataset comes from historical Washington State transactions.
