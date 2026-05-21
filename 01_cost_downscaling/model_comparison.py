# -*- coding: utf-8 -*-
"""Multi-model comparison: Ridge vs RF vs XGBoost under LOYO CV.
   15 x-variables → total_cost_residual (detrended).
   Output: metrics table + fold summary + scatter panel + bar charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ================= CONFIG =================
BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\07_total_cost_detrended_modeling')
INPUT_DIR = BASE_DIR / 'tables' / 'gt100'
TABLES_OUT = BASE_DIR / 'tables' / 'model_comparison'
FIG_OUT = BASE_DIR / 'figures' / 'gt100'
CROPS = ('maize', 'rice', 'wheat')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

TABLES_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    'Ridge': (
        Ridge(alpha=1.0, random_state=42),
        {'needs_scaling': True}
    ),
    'Random Forest': (
        RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                              n_jobs=4, random_state=42),
        {}
    ),
    'XGBoost': (
        XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                     random_state=42, n_jobs=4),
        {}
    ),
}

COLORS = {'Ridge': '#4A90E2', 'Random Forest': '#50B86C', 'XGBoost': '#E24A4A'}
DISPLAY_ORDER = ['Ridge', 'Random Forest', 'XGBoost']


def run_loyo_comparison():
    all_metrics = []
    all_folds = []

    for crop in CROPS:
        # ---- load data ----
        input_path = INPUT_DIR / f'{crop}_model_input_total_cost_residual_gt100_2000_2019.csv'
        df = pd.read_csv(input_path)
        x_cols = [c for c in df.columns if c.startswith('x_')]
        years = sorted(df['year'].unique())

        for model_name, (model_cls, kwargs) in MODELS.items():
            for fold_idx, held_out_year in enumerate(years, start=1):
                train = df[df['year'] != held_out_year]
                test = df[df['year'] == held_out_year]
                if train.empty or test.empty:
                    continue

                X_train = train[x_cols].values
                y_train = train['total_cost_residual'].values
                X_test = test[x_cols].values
                y_test = test['total_cost_residual'].values
                year_mean = test['national_year_mean_total_cost'].values

                # ---- scaling for Ridge ----
                if kwargs.get('needs_scaling'):
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_test_sc = scaler.transform(X_test)
                else:
                    X_train_sc, X_test_sc = X_train, X_test

                # ---- fit & predict ----
                m = model_cls
                if kwargs.get('needs_scaling'):
                    m = Ridge(alpha=1.0, random_state=42)
                m.fit(X_train_sc, y_train)
                y_pred_resid = m.predict(X_test_sc)
                y_pred_total = y_pred_resid + year_mean

                # ---- residual metrics ----
                resid_r2 = r2_score(y_test, y_pred_resid)
                resid_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_resid)))
                resid_mae = float(mean_absolute_error(y_test, y_pred_resid))

                # ---- absolute cost metrics ----
                total_r2 = r2_score(test['y_value'], y_pred_total)
                total_rmse = float(np.sqrt(mean_squared_error(test['y_value'], y_pred_total)))

                all_folds.append({
                    'crop': crop,
                    'threshold_scheme': 'gt100',
                    'model': model_name,
                    'fold_idx': fold_idx,
                    'held_out_year': int(held_out_year),
                    'test_sample_count': len(test),
                    'residual_r2': resid_r2,
                    'residual_rmse': resid_rmse,
                    'residual_mae': resid_mae,
                    'total_cost_r2': total_r2,
                    'total_cost_rmse': total_rmse,
                })

    # ---- pooled metrics across all folds (per crop-model) ----
    for crop in CROPS:
        input_path = INPUT_DIR / f'{crop}_model_input_total_cost_residual_gt100_2000_2019.csv'
        df = pd.read_csv(input_path)
        x_cols = [c for c in df.columns if c.startswith('x_')]
        years = sorted(df['year'].unique())

        for model_name, (model_cls, kwargs) in MODELS.items():
            y_true_resid_all, y_pred_resid_all = [], []
            y_true_total_all, y_pred_total_all = [], []

            for held_out_year in years:
                train = df[df['year'] != held_out_year]
                test = df[df['year'] == held_out_year]
                if train.empty or test.empty:
                    continue
                X_train = train[x_cols].values
                X_test = test[x_cols].values
                y_train = train['total_cost_residual'].values
                y_test = test['total_cost_residual'].values
                year_mean = test['national_year_mean_total_cost'].values
                if kwargs.get('needs_scaling'):
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    m = Ridge(alpha=1.0, random_state=42)
                else:
                    X_train_s, X_test_s = X_train, X_test
                    m = model_cls
                m.fit(X_train_s, y_train)
                pred = m.predict(X_test_s)
                y_true_resid_all.extend(y_test.tolist())
                y_pred_resid_all.extend(pred.tolist())
                y_true_total_all.extend(test['y_value'].tolist())
                y_pred_total_all.extend((pred + year_mean).tolist())

            y_true_resid_all = np.array(y_true_resid_all)
            y_pred_resid_all = np.array(y_pred_resid_all)
            y_true_total_all = np.array(y_true_total_all)
            y_pred_total_all = np.array(y_pred_total_all)

            crop_folds = [f for f in all_folds
                          if f['crop'] == crop and f['model'] == model_name]
            n_samples = len(y_true_resid_all)
            all_metrics.append({
                'crop': crop,
                'threshold_scheme': 'gt100',
                'model': model_name,
                'sample_count': n_samples,
                'fold_count': len(crop_folds),
                'residual_r2': float(r2_score(y_true_resid_all, y_pred_resid_all)),
                'residual_rmse': float(np.sqrt(mean_squared_error(y_true_resid_all, y_pred_resid_all))),
                'residual_mae': float(mean_absolute_error(y_true_resid_all, y_pred_resid_all)),
                'total_cost_r2': float(r2_score(y_true_total_all, y_pred_total_all)),
                'total_cost_rmse': float(np.sqrt(mean_squared_error(y_true_total_all, y_pred_total_all))),
            })

    # ---- save tables ----
    metrics_df = pd.DataFrame(all_metrics).round(4)
    folds_df = pd.DataFrame(all_folds).round(4)

    metrics_path = TABLES_OUT / 'multicomp_metrics_gt100_2000_2019.csv'
    folds_path = TABLES_OUT / 'multicomp_fold_summary_gt100_2000_2019.csv'
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    folds_df.to_csv(folds_path, index=False, encoding='utf-8-sig')
    print(f'[OK] {metrics_path}')
    print(f'[OK] {folds_path}')
    print(metrics_df.to_string(index=False))

    # ---- plot 1a: scatter panel (absolute cost) ----
    _plot_scatter_panel('absolute', 'multicomp_scatter_panel_gt100.png')

    # ---- plot 1b: scatter panel (residual) ----
    _plot_scatter_panel('residual', 'multicomp_scatter_residual_panel_gt100.png')

    # ---- plot 2: grouped bar R² (total cost) ----
    _plot_bar(metrics_df, 'total_cost_r2', 'Total Cost R²', 'multicomp_bar_total_r2_gt100.png')

    # ---- plot 3: grouped bar R² (residual) ----
    _plot_bar(metrics_df, 'residual_r2', 'Residual R²', 'multicomp_bar_residual_r2_gt100.png')

    # ---- plot 4: grouped bar RMSE (residual) ----
    _plot_bar(metrics_df, 'residual_rmse', 'Residual RMSE (yuan/mu)', 'multicomp_bar_residual_rmse_gt100.png')

    return metrics_df, folds_df


def _plot_scatter_panel(mode, filename):
    """3 crops × 3 models scatter panel.
       mode: 'absolute' = total cost (residual + year_mean)
             'residual' = detrended spatial residual only."""
    suffix = 'Absolute Total Cost' if mode == 'absolute' else 'Detrended Residual'
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.25, wspace=0.22)

    for row, crop in enumerate(CROPS):
        for col, model_name in enumerate(DISPLAY_ORDER):
            ax = axes[row, col]
            ax.set_box_aspect(1)

            input_path = INPUT_DIR / f'{crop}_model_input_total_cost_residual_gt100_2000_2019.csv'
            df = pd.read_csv(input_path)
            x_cols = [c for c in df.columns if c.startswith('x_')]
            years = sorted(df['year'].unique())

            y_true_all, y_pred_all = [], []
            for held_out_year in years:
                train = df[df['year'] != held_out_year]
                test = df[df['year'] == held_out_year]
                if train.empty or test.empty:
                    continue

                X_train = train[x_cols].values
                X_test = test[x_cols].values
                y_train = train['total_cost_residual'].values
                y_test = test['total_cost_residual'].values
                year_mean = test['national_year_mean_total_cost'].values

                model_cls, kwargs = MODELS[model_name]
                if kwargs.get('needs_scaling'):
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    m = Ridge(alpha=1.0, random_state=42)
                else:
                    X_train_s, X_test_s = X_train, X_test
                    m = model_cls
                m.fit(X_train_s, y_train)
                pred = m.predict(X_test_s)

                if mode == 'absolute':
                    y_true_all.extend(test['y_value'].tolist())
                    y_pred_all.extend((pred + year_mean).tolist())
                else:
                    y_true_all.extend(test['total_cost_residual'].tolist())
                    y_pred_all.extend(pred.tolist())

            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            r2 = r2_score(y_true_all, y_pred_all)
            rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))

            vmin = min(y_true_all.min(), y_pred_all.min())
            vmax = max(y_true_all.max(), y_pred_all.max())
            pad = (vmax - vmin) * 0.08
            pmin, pmax = vmin - pad, vmax + pad

            ax.scatter(y_true_all, y_pred_all, s=8, alpha=0.40, color=COLORS[model_name],
                       linewidth=0, zorder=2)
            ax.plot([pmin, pmax], [pmin, pmax], '--', color='#555555', lw=1.0, zorder=1)

            # fit line
            slope, intercept = np.polyfit(y_true_all, y_pred_all, 1)
            ax.plot([pmin, pmax], [slope * pmin + intercept, slope * pmax + intercept],
                    color='#222222', lw=1.3, zorder=3)

            ax.set_xlim(pmin, pmax)
            ax.set_ylim(pmin, pmax)
            ax.grid(True, linestyle=':', linewidth=0.4, color='#d9d9d9', alpha=0.7)
            ax.tick_params(labelsize=7.5)

            stats_text = f'$R^2$={r2:.3f}\nRMSE={rmse:.1f}'
            ax.text(0.97, 0.04, stats_text, transform=ax.transAxes, fontsize=7.5,
                    va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='#cccccc', alpha=0.9))

            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold', pad=6)
            if row == 2:
                ax.set_xlabel(f'Observed ({suffix}) (yuan/mu)', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{crop.capitalize()} Predicted ({suffix}) (yuan/mu)', fontsize=8)

    fig.suptitle(f'Multi-model comparison — Reconstruction of {suffix} (gt100, LOYO CV)',
                 fontsize=13, y=0.985)
    out_path = FIG_OUT / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


def _plot_bar(metrics_df, metric_col, metric_label, filename):
    """Grouped bar chart: 3 models × 3 crops."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.22
    x = np.arange(len(CROPS))

    for i, model_name in enumerate(DISPLAY_ORDER):
        vals = []
        for crop in CROPS:
            row = metrics_df[(metrics_df['crop'] == crop) & (metrics_df['model'] == model_name)]
            vals.append(row[metric_col].values[0] if len(row) > 0 else 0)
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=model_name, color=COLORS[model_name],
                      edgecolor='white', linewidth=0.5)
        # annotate
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CROPS], fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis='y', linestyle=':', linewidth=0.4, color='#d9d9d9', alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    fig.tight_layout()
    out_path = FIG_OUT / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


if __name__ == '__main__':
    run_loyo_comparison()
    print('\nDone.')
