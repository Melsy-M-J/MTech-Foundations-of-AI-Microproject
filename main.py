"""Reproducible improvement experiment for the 2014 Washington housing dataset.

Usage (inside the folder):
    python main.py
    python main.py --csv path/to/modified_data.csv

This project intentionally compares ONLY Linear Regression and Decision Tree
Regression, using two training-set-only modifications:
  * standardizing numeric inputs and log1p transforming prices for Linear
    Regression, with a training-only quantile cap on predictions;
  * median-leaf regression trees (absolute_error splitting) tuned by CV.

IMPORTANT: The 20% holdout was seen in the user's prior experiments. Final
numbers from this experiment are exploratory, not an untouched external test.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

BASE = Path(__file__).resolve().parent
OUT = BASE/'results'
DEFAULT_CSV=BASE/'data'/'modified_data.csv'
TARGET='price'
NUMERIC=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
         'condition','sqft_above','sqft_basement','yr_built','yr_renovated']
CATEGORICAL=['city','statezip']
FEATURES=NUMERIC+CATEGORICAL

# These are intentional, pre-specified comparisons. Each CV fold fits all
# preprocessing and any target-dependent cap using only that fold's training rows.
CANDIDATES=[
 {'name':'Linear original','family':'linear','scale':False,'log':False,'cap':None},
 {'name':'Linear standardized','family':'linear','scale':True,'log':False,'cap':None},
 {'name':'Linear log + cap 99.5%','family':'linear','scale':True,'log':True,'cap':0.995},
 {'name':'Linear log + cap 99.9%','family':'linear','scale':True,'log':True,'cap':0.999},
 {'name':'Tree original','family':'tree','criterion':'squared_error','depth':12,'leaf':10},
 {'name':'Tree median depth 10 leaf 10','family':'tree','criterion':'absolute_error','depth':10,'leaf':10},
 {'name':'Tree median depth 12 leaf 5','family':'tree','criterion':'absolute_error','depth':12,'leaf':5},
]

def load_data(filename):
    df=pd.read_csv(filename)
    missing=set(FEATURES+[TARGET])-set(df.columns)
    if missing:raise ValueError(f'CSV is missing columns: {sorted(missing)}')
    total=len(df)
    df=df.drop_duplicates().copy()
    dup=total-len(df)
    df[TARGET]=pd.to_numeric(df[TARGET],errors='coerce')
    invalid=int((df[TARGET].isna() | (df[TARGET]<=0)).sum())
    df=df.loc[df[TARGET].gt(0)].copy()
    for col in NUMERIC:df[col]=pd.to_numeric(df[col],errors='coerce')
    for col in CATEGORICAL:df[col]=df[col].fillna('UNKNOWN').astype(str)
    return df, {'original_rows':total,'duplicates_removed':dup,'nonpositive_or_missing_target_removed':invalid,'usable_rows':len(df)}

def get_pipeline(config):
    if config['family']=='linear':
        # Scaling greatly reduces numerical conditioning problems when features
        # such as lot size, build year, and room count have very different ranges.
        numeric=Pipeline([('imputer',SimpleImputer(strategy='median')),
                          ('scaler',StandardScaler())]) if config['scale'] else SimpleImputer(strategy='median')
        estimator=LinearRegression()
    else:
        numeric=SimpleImputer(strategy='median')
        estimator=DecisionTreeRegressor(criterion=config['criterion'],max_depth=config['depth'],
                                        min_samples_leaf=config['leaf'],random_state=42)
    preprocessor=ColumnTransformer([('numeric',numeric,NUMERIC),
                                     ('categorical',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)])
    return Pipeline([('preprocessing',preprocessor),('model',estimator)])

def fit_and_predict(config,Xtrain,ytrain,Xvalid):
    pipeline=get_pipeline(config)
    train_target=np.log1p(ytrain) if config.get('log',False) else ytrain
    pipeline.fit(Xtrain,train_target)
    predictions=pipeline.predict(Xvalid)
    if config.get('log',False):predictions=np.expm1(np.clip(predictions,-10,25))
    upper=float(ytrain.quantile(config['cap'])) if config.get('cap') is not None else np.inf
    # No target value from Xvalid / test contributes to the cap.
    predictions=np.clip(predictions,0,upper)
    return predictions, pipeline, upper

def metrics(y,p):
    return dict(MAE_USD=float(mean_absolute_error(y,p)),
                RMSE_USD=float(np.sqrt(mean_squared_error(y,p))),
                R2=float(r2_score(y,p)),
                Median_AE_USD=float(median_absolute_error(y,p)))

def plot_results(ytest,predictions,report,cv_report,df):
    # Fit plots show only selected price range while metrics use ALL rows.
    zoom=float(np.quantile(ytest,0.99))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(title,pred) in zip(axes,predictions.items()):
        yv=np.asarray(ytest)
        keep=(yv<=zoom)&(pred<=zoom)
        ax.scatter(yv[keep]/1e6,pred[keep]/1e6,alpha=.35,s=12)
        ax.plot([0,zoom/1e6],[0,zoom/1e6],ls='--',c='black',lw=1)
        ax.set(title=title,xlabel='Actual price (million USD)',ylabel='Predicted price (million USD)',
               xlim=(0,zoom/1e6),ylim=(0,zoom/1e6))
        ax.grid(alpha=.15)
    fig.suptitle('Improved models: actual vs predicted (zoomed; metrics use full test set)')
    fig.tight_layout()
    fig.savefig(OUT/'improved_actual_vs_predicted.png',dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=report['Model'].tolist();pos=np.arange(len(labels))
    axes[0].bar(pos,report['MAE_USD'])
    axes[0].set_xticks(pos,labels,rotation=30,ha='right')
    axes[0].set_ylabel('MAE (USD), lower is better')
    axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(pos,report['R2'])
    axes[1].set_xticks(pos,labels,rotation=30,ha='right')
    axes[1].set_ylabel('R², higher is better')
    axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'model_improvement_comparison.png',dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,6))
    ranked=cv_report.sort_values('CV_MAE_USD')
    ax.barh(ranked['Model'][::-1],ranked['CV_MAE_USD'][::-1])
    ax.set(xlabel='Average five-fold training CV MAE (USD)',title='Training-only model comparison — lower is better')
    ax.grid(axis='x',alpha=.15)
    fig.tight_layout()
    fig.savefig(OUT/'training_cross_validation.png',dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    prices=np.asarray(df[TARGET])
    ax.hist(prices[prices<=np.quantile(prices,.995)]/1e6,bins=45)
    ax.set(title='House-price distribution (visualized through 99.5th percentile)',xlabel='Price (million USD)',ylabel='Number of properties')
    fig.tight_layout()
    fig.savefig(OUT/'price_distribution.png',dpi=160)
    plt.close(fig)



def save_complete_tree_visualizations(tree_pipeline):
    # Save the FULL selected Decision Tree as zoomable SVG/PDF and a preview PNG.
    preprocessor = tree_pipeline.named_steps['preprocessing']
    tree_model = tree_pipeline.named_steps['model']
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace('numeric__', '').replace('categorical__', '')
                     for name in feature_names]

    node_count = int(tree_model.tree_.node_count)
    leaf_count = int(tree_model.get_n_leaves())
    max_depth = int(tree_model.get_depth())

    # A wide canvas is needed because the complete tree has many terminal leaves.
    # SVG and PDF are vector formats, so labels remain sharp when zoomed in.
    fig_width = max(36, min(leaf_count * 0.78, 260))
    fig_height = max(20, max_depth * 1.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        precision=0,
        fontsize=4,
        ax=ax,
    )
    ax.set_title(
        f'Complete Decision Tree Regression | {node_count} nodes | '
        f'{leaf_count} leaves | depth {max_depth}',
        fontsize=14,
        pad=18,
    )
    ax.axis('off')
    fig.savefig(OUT / 'complete_decision_tree.svg', format='svg', bbox_inches='tight')
    fig.savefig(OUT / 'complete_decision_tree.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)

    # The first four levels are easier to read in a report or presentation.
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        max_depth=3,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        precision=0,
        fontsize=7,
        ax=ax,
    )
    ax.set_title('Decision Tree — First Four Levels (Preview)', fontsize=16, pad=16)
    ax.axis('off')
    fig.savefig(OUT / 'decision_tree_preview.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Export all split rules as searchable text.
    rules = export_text(tree_model, feature_names=feature_names, max_depth=max_depth,
                        decimals=0, show_weights=True)
    (OUT / 'complete_decision_tree_rules.txt').write_text(rules, encoding='utf-8')

    # HTML viewer with zoom controls and scrollbars for navigating the entire tree.
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complete Decision Tree</title>
<style>
body{font-family:Arial,sans-serif;margin:18px;color:#222} .toolbar{position:sticky;top:0;background:#fff;padding:10px;border:1px solid #ddd;z-index:2;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{padding:7px 12px;cursor:pointer} #viewport{height:78vh;overflow:auto;border:1px solid #bbb;margin-top:12px;background:#f7f7f7}
#tree{display:block;width:2800px;max-width:none;height:auto;background:white}a{margin-right:12px}
</style></head>
<body><h1>Complete Decision Tree Regression</h1>
<p>This is the complete fitted tree, including every split and leaf. Use the zoom controls and scrollbars to inspect branches.</p>
<p><a href="complete_decision_tree.svg">Open SVG</a> <a href="complete_decision_tree.pdf">Open PDF</a> <a href="decision_tree_preview.png">Open first-four-level preview</a> <a href="complete_decision_tree_rules.txt">Read all decision rules</a></p>
<div class="toolbar"><button id="minus">− Zoom out</button><button id="plus">+ Zoom in</button><button id="fit">Fit to window</button><label>Zoom <input id="zoom" type="range" min="1000" max="18000" step="250" value="2800"></label><span id="zoomLabel">2800 px</span></div>
<div id="viewport"><img id="tree" src="complete_decision_tree.svg" alt="Complete decision tree"></div>
<script>
const img=document.getElementById('tree'), slider=document.getElementById('zoom'), label=document.getElementById('zoomLabel');
function setWidth(w){w=Math.max(1000,Math.min(18000,w));img.style.width=w+'px';slider.value=w;label.textContent=w+' px';}
document.getElementById('minus').onclick=()=>setWidth(Number(slider.value)-750);
document.getElementById('plus').onclick=()=>setWidth(Number(slider.value)+750);
document.getElementById('fit').onclick=()=>setWidth(Math.max(1000,document.getElementById('viewport').clientWidth-20));
slider.oninput=()=>setWidth(Number(slider.value));
</script></body></html>'''
    (OUT / 'open_complete_decision_tree.html').write_text(html, encoding='utf-8')

    return node_count, leaf_count, max_depth

def run(filename):
    OUT.mkdir(exist_ok=True,parents=True)
    data,info=load_data(filename)
    Xtrain,Xtest,ytrain,ytest=train_test_split(data[FEATURES],data[TARGET],test_size=.2,random_state=42)
    cv=KFold(n_splits=5,shuffle=True,random_state=2026)
    cvrows=[]
    print('CSV:',filename,'\nRows:',info,'\nTraining:',len(Xtrain),'Test:',len(Xtest))
    print('Excluded from features: price_per_sqft (target leakage), street, date')
    print('Cross-validation: five folds on training rows only. Primary selection metric: MAE.\n')
    for config in CANDIDATES:
        fold_metrics=[]
        for train_idx,val_idx in cv.split(Xtrain):
            xf=Xtrain.iloc[train_idx];xv=Xtrain.iloc[val_idx]
            yf=ytrain.iloc[train_idx];yv=ytrain.iloc[val_idx]
            pred,_,_=fit_and_predict(config,xf,yf,xv)
            fold_metrics.append(metrics(yv,pred))
        avg={key:np.mean([fm[key] for fm in fold_metrics]) for key in fold_metrics[0]}
        cvrows.append({'Model':config['name'],'Family':config['family'],
                        **{'CV_'+key:val for key,val in avg.items()}})
        print(f"{config['name']:<32} CV MAE ${avg['MAE_USD']:>11,.0f}   CV R² {avg['R2']:.3f}")
    cv_report=pd.DataFrame(cvrows).sort_values('CV_MAE_USD').reset_index(drop=True)
    cv_report.to_csv(OUT/'training_cv_comparison.csv',index=False)

    best={}
    for family in ['linear','tree']:
        winning_name=cv_report.loc[cv_report['Family'].eq(family),'Model'].iloc[0]
        best[family]=next(c for c in CANDIDATES if c['name']==winning_name)
    print('\nSelected using training CV:',{k:v['name'] for k,v in best.items()})
    final_configs=[CANDIDATES[0],CANDIDATES[4],best['linear'],best['tree']]
    testrows=[];selected_preds={};selected_tree_pipeline=None
    for config in final_configs:
        pred,model,upper=fit_and_predict(config,Xtrain,ytrain,Xtest)
        testrows.append({'Model':config['name'],**metrics(ytest,pred),'Max_Prediction_Cap_USD':None if np.isinf(upper) else upper})
        selected_preds[config['name']]=pred
        if config['name']==best['tree']['name']:
            selected_tree_pipeline=model
    report=pd.DataFrame(testrows)
    report.to_csv(OUT/'test_model_comparison_EXPLORATORY.csv',index=False)
    print('\nTEST results (exploratory: user previously examined the test set):\n',report.to_string(index=False,float_format=lambda f:f'{f:,.3f}'))
    pd.DataFrame({'actual_price_usd':np.asarray(ytest),**{k+'_prediction_usd':v for k,v in selected_preds.items()}}).to_csv(OUT/'test_predictions.csv',index=False)
    plot_results(ytest,{best[k]['name']:selected_preds[best[k]['name']] for k in ['linear','tree']},report,cv_report,data)
    tree_stats=save_complete_tree_visualizations(selected_tree_pipeline)
    print('\nComplete Decision Tree visualization generated:')
    print(f'  Nodes: {tree_stats[0]} | Leaves: {tree_stats[1]} | Depth: {tree_stats[2]}')
    print('  Full SVG :', OUT/'complete_decision_tree.svg')
    print('  Full PDF :', OUT/'complete_decision_tree.pdf')
    print('  Open HTML:', OUT/'open_complete_decision_tree.html')
    print('  Preview  :', OUT/'decision_tree_preview.png')
    print('  All rules:', OUT/'complete_decision_tree_rules.txt')
    suspect=data.loc[data.price>5e6,['price','sqft_living','city','statezip']].copy()
    suspect['price_per_sqft_diagnostic_ONLY']=suspect.price/suspect.sqft_living
    suspect.sort_values('price',ascending=False).to_csv(OUT/'high_price_rows_to_verify.csv',index=False)
    metadata=dict(info,training_rows=len(Xtrain),test_rows=len(Xtest),split_seed=42,cv_seed=2026,cv_folds=5,
                  selection_metric='training-only five-fold mean absolute error',
                  selected_models={k:v['name'] for k,v in best.items()},
                  caveat='This held-out test set had already been examined in previous experiments. Treat test results as exploratory rather than untouched confirmation.',
                  target='price (USD)',not_used=['price_per_sqft','street','date'])
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nResults saved in:',OUT)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,default=DEFAULT_CSV)
    args=parser.parse_args();run(args.csv)
