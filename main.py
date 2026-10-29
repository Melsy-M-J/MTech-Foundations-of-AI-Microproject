"""House-price regression microproject using the user's 2014 Washington housing CSV.

Run: python main.py
Optional: python main.py --csv path/to/your_data.csv
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / 'data' / 'modified_data.csv'
RESULTS = ROOT / 'results'
TARGET = 'price'
NUMERIC = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated',
]
CATEGORICAL = ['city', 'statezip']
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = {
    'price_per_sqft': 'TARGET LEAKAGE: calculated from price / sqft_living',
    'street': 'specific address / near-unique ID, high-cardinality',
    'date': 'not used in this initial historical cross-sectional baseline',
}


def read_dataset(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f'Cannot find dataset: {file_path}')
    df = pd.read_csv(file_path)
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f'CSV lacks these required columns: {missing}')
    return df


def clean_dataset(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    invalid_target = int((df[TARGET].isna() | (df[TARGET] <= 0)).sum())
    df = df.loc[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in CATEGORICAL:
        df[col] = df[col].astype('string').fillna('UNKNOWN').astype(str)
    return df, duplicates_removed, invalid_target


def make_preprocessing():
    # Every imputer and encoder is fitted INSIDE the training/CV pipeline.
    return ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])


def model_metrics(y_true, y_pred, name):
    return {
        'Model': name,
        'MAE_USD': float(mean_absolute_error(y_true, y_pred)),
        'RMSE_USD': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'Median_AE_USD': float(median_absolute_error(y_true, y_pred)),
    }


def save_plots(y_test, predictions, metrics, tree_model, clean_df):
    labels = list(predictions)
    # Full-dataset metrics are always computed on all test samples.
    # A zoomed plot is used only to keep the dense majority of samples readable.
    cutoff = float(np.quantile(y_test, 0.99))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, labels):
        pred = predictions[label]
        mask = (y_test.to_numpy() <= cutoff) & (pred <= cutoff)
        ax.scatter(y_test.to_numpy()[mask] / 1e6, pred[mask] / 1e6, s=11, alpha=0.35)
        ax.plot([0, cutoff / 1e6], [0, cutoff / 1e6], '--', color='black', lw=1)
        ax.set(title=label + f'\nZoom: up to ${cutoff / 1e6:.2f}M (both axes)',
               xlabel='Actual price (million USD)', ylabel='Predicted price (million USD)',
               xlim=(0, cutoff / 1e6), ylim=(0, cutoff / 1e6))
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'actual_vs_predicted.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ['MAE_USD', 'RMSE_USD']):
        ax.bar(metrics['Model'], metrics[metric])
        ax.set(title=metric + ' (lower is better)', ylabel='Error in US dollars')
        ax.tick_params(axis='x', labelrotation=12)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'error_comparison.png', dpi=180)
    plt.close(fig)

    corr = clean_df[NUMERIC + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=65, ha='right')
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(im, ax=ax, label='Pearson correlation coefficient')
    fig.tight_layout()
    fig.savefig(RESULTS / 'correlation_heatmap.png', dpi=180)
    plt.close(fig)

    tree = tree_model.named_steps['model']
    encoded_names = tree_model.named_steps['preprocessing'].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(19, 8))
    plot_tree(tree, max_depth=3, feature_names=list(encoded_names), filled=True,
              rounded=True, fontsize=7, precision=0, ax=ax)
    ax.set_title('Decision tree: first four levels only (partial visualization)')
    fig.tight_layout()
    fig.savefig(RESULTS / 'decision_tree_preview.png', dpi=160)
    plt.close(fig)

    importance = pd.DataFrame({
        'Feature': encoded_names,
        'Importance': tree.feature_importances_,
    }).sort_values('Importance', ascending=False)
    importance.to_csv(RESULTS / 'tree_feature_importances.csv', index=False)
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top['Feature'], top['Importance'])
    ax.set(xlabel='Importance', title='Decision tree: top 12 encoded features')
    fig.tight_layout()
    fig.savefig(RESULTS / 'feature_importance.png', dpi=180)
    plt.close(fig)


def run(csv_file):
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = read_dataset(csv_file)
    clean, duplicates_removed, invalid_target = clean_dataset(raw)
    if len(clean) < 50:
        raise ValueError('Not enough positive-price records for a reliable split.')
    print('DATASET:', csv_file)
    print(f'Rows: original={len(raw):,}, retained={len(clean):,}; '
          f'duplicates removed={duplicates_removed}, invalid/zero prices removed={invalid_target}')
    print(f'Features: {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical')
    print('Leakage excluded:', EXCLUDED['price_per_sqft'])
    print('Missing values before cleaning:', int(raw.isna().sum().sum()))
    print('Target price summary (USD):')
    print(clean[TARGET].describe(percentiles=[.5, .95, .99]).to_string())

    clean[FEATURES + [TARGET]].describe(include='all').to_csv(RESULTS / 'dataset_summary.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=0.20, random_state=42
    )
    linear = Pipeline([
        ('preprocessing', make_preprocessing()),
        ('model', LinearRegression()),
    ])
    tree_pipeline = Pipeline([
        ('preprocessing', make_preprocessing()),
        ('model', DecisionTreeRegressor(random_state=42)),
    ])
    tree_search = GridSearchCV(
        estimator=tree_pipeline,
        param_grid={
            'model__max_depth': [4, 6, 8, 12],
            'model__min_samples_leaf': [5, 10, 20],
        },
        scoring='neg_mean_absolute_error', cv=5, n_jobs=-1,
    )
    print('\nTraining Linear Regression and tuning Decision Tree using training data only...')
    linear.fit(X_train, y_train)
    tree_search.fit(X_train, y_train)
    best_tree = tree_search.best_estimator_
    print('Best decision-tree settings:', tree_search.best_params_)

    predictions = {}
    rows = []
    for name, model in [('Linear Regression', linear), ('Decision Tree Regression', best_tree)]:
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append(model_metrics(y_test, predicted, name))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / 'model_comparison.csv', index=False)
    print('\nUNSEEN TEST SET METRICS (US dollars, for MAE/RMSE/MedAE):')
    print(metrics.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))
    pd.DataFrame({
        'Actual_USD': y_test.to_numpy(),
        'Linear_Regression_USD': predictions['Linear Regression'],
        'Decision_Tree_Regression_USD': predictions['Decision Tree Regression'],
    }).to_csv(RESULTS / 'test_predictions.csv', index=False)
    save_plots(y_test, predictions, metrics, best_tree, clean)

    metadata = {
        'source_file': csv_file.name,
        'data_description': 'User-supplied 2014 Washington State property listing dataset',
        'total_rows': len(raw), 'positive_price_rows': len(clean),
        'zero_or_missing_price_rows_removed': invalid_target,
        'duplicate_rows_removed': duplicates_removed,
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'test_size': 0.2, 'random_state': 42, 'cv_folds': 5,
        'numerical_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'excluded_fields': EXCLUDED, 'target': TARGET, 'target_units': 'USD',
        'tree_tuning_criterion': 'cross-validation mean absolute error',
        'best_tree_hyperparameters': tree_search.best_params_,
        'test_metric_note': 'Metrics calculated on ALL held-out test records; scatter plot is zoomed.',
        'unusually_high_price_max_retained': float(clean[TARGET].max()),
    }
    (RESULTS / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nOutput files saved under:', RESULTS)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Linear Regression and Decision Tree on housing CSV')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV, help='Path to housing CSV')
    opts = parser.parse_args()
    run(opts.csv)
