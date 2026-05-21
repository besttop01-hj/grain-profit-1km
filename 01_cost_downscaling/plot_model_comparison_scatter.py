# -*- coding: utf-8 -*-
"""3×3 multi-model residual scatter panel (V8 style).
   Rows = crops (maize, rice, wheat), Columns = models (Ridge, RF, XGBoost).
   Each panel: scatter (obs vs pred residual) + 1:1 line + fit line + R²/RMSE.
   Marginal density on top and right, single-color per model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from matplotlib.lines import Line2D
import os, warnings
warnings.filterwarnings("ignore")

# ================= CONFIG =================
INPUT_DIR = r"F:\zhongliangshouyi\3zuowu_chuli\07_total_cost_detrended_modeling\tables\gt100"
OUTPUT_DIR = r"F:\zhongliangshouyi\3zuowu_chuli\07_total_cost_detrended_modeling\figures\gt100"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CROPS = ('maize', 'rice', 'wheat')
DISPLAY_NAMES = {'maize': 'Maize', 'rice': 'Rice', 'wheat': 'Wheat'}
MODEL_ORDER = ['Ridge', 'Random Forest', 'XGBoost']

# ================= 配色 =================
COLORS = {
    'Ridge':          '#4A90E2',   # 蓝
    'Random Forest':  '#50B86C',   # 绿
    'XGBoost':        '#E24A4A',   # 红
}
FILL_ALPHA = 0.22
EDGE_ALPHA = 0.85

COLOR_1TO1 = "#555555"
COLOR_FIT = "#222222"
TEXT_COLOR = "#333333"

plt.rcParams['font.family'] = 'Arial'

MODELS = {
    'Ridge':          Ridge(alpha=1.0, random_state=42),
    'Random Forest':  RandomForestRegressor(n_estimators=300, max_depth=6,
                                            min_samples_leaf=3, n_jobs=4,
                                            random_state=42),
    'XGBoost':        XGBRegressor(n_estimators=400, max_depth=4,
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=42, n_jobs=4),
}

# ================= LOYO 预测 =================
def get_loyo_residual_predictions(crop, model_name):
    """Return (y_true_resid, y_pred_resid) pooled across 20 LOYO folds."""
    df = pd.read_csv(os.path.join(INPUT_DIR, f'{crop}_model_input_total_cost_residual_gt100_2000_2019.csv'))
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

        if model_name == 'Ridge':
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            m = Ridge(alpha=1.0, random_state=42)
        else:
            X_train_s, X_test_s = X_train, X_test
            m = MODELS[model_name]
        m.fit(X_train_s, y_train)
        pred = m.predict(X_test_s)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
    return np.array(y_true_all), np.array(y_pred_all)

# ================= 出图 =================
def plot():
    fig = plt.figure(figsize=(20, 17))
    outer = fig.add_gridspec(3, 3, wspace=0.16, hspace=0.22)

    for row, crop in enumerate(CROPS):
        for col, model_name in enumerate(MODEL_ORDER):
            y_true, y_pred = get_loyo_residual_predictions(crop, model_name)

            r2 = r2_score(y_true, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            n = len(y_true)

            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())
            pad = (vmax - vmin) * 0.08
            pmin, pmax = vmin - pad, vmax + pad

            # --- Inner subgrid: main + top density + right density ---
            inner = outer[row, col].subgridspec(2, 2,
                        width_ratios=[7, 1], height_ratios=[1, 7],
                        wspace=0.03, hspace=0.03)
            ax_main = fig.add_subplot(inner[1, 0])
            ax_top  = fig.add_subplot(inner[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

            color = COLORS[model_name]
            fill_c = (*plt.cm.colors.to_rgba(color)[:3], FILL_ALPHA)
            edge_c = (*plt.cm.colors.to_rgba(color)[:3], EDGE_ALPHA)

            # --- Main scatter ---
            ax_main.set_box_aspect(1)
            ax_main.plot([pmin, pmax], [pmin, pmax], c=COLOR_1TO1, ls='--', lw=2.0, zorder=1)
            ax_main.scatter(y_true, y_pred, s=22, facecolor=fill_c, edgecolor=edge_c,
                            linewidth=0.6, zorder=2)
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            ax_main.plot([pmin, pmax], [slope*pmin+intercept, slope*pmax+intercept],
                         c=COLOR_FIT, lw=2.2, zorder=3)
            ax_main.set_xlim(pmin, pmax)
            ax_main.set_ylim(pmin, pmax)
            ax_main.grid(True, linestyle=':', linewidth=0.5, color='#d9d9d9', alpha=0.8)
            ax_main.tick_params(labelsize=9)

            # --- Marginal densities ---
            sns.kdeplot(y_true, ax=ax_top, color=color, fill=True,
                        facecolor=fill_c, linewidth=2.0)
            ax_top.axis('off')
            sns.kdeplot(y=y_pred, ax=ax_right, color=color, fill=True,
                        facecolor=fill_c, linewidth=2.0)
            ax_right.axis('off')

            # --- Labels ---
            ax_main.text(0.03, 0.96, f'{DISPLAY_NAMES[crop]} — {model_name}',
                         transform=ax_main.transAxes, fontsize=13, fontweight='bold',
                         va='top', ha='left', color=TEXT_COLOR)
            stats = f"$R^2$={r2:.3f}   RMSE={rmse:.1f}   n={n}"
            ax_main.text(0.03, 0.86, stats, transform=ax_main.transAxes, fontsize=10,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='#cccccc', alpha=0.9))

            if row == 2:
                ax_main.set_xlabel('Observed residual (yuan/mu)', fontsize=10)
            if col == 0:
                ax_main.set_ylabel(f'{DISPLAY_NAMES[crop]} predicted residual (yuan/mu)', fontsize=10)

    # --- Global legend ---
    legend_els = [
        Line2D([0],[0], color=COLORS[m], lw=3, label=m) for m in MODEL_ORDER
    ]
    legend_els += [
        Line2D([0],[0], color=COLOR_1TO1, lw=2, ls='--', label='1:1 Line'),
        Line2D([0],[0], color=COLOR_FIT, lw=2.2, label='Fit Line'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=5, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 0.005), columnspacing=3.0,
               handletextpad=0.5)

    fig.suptitle('Multi-model Comparison — Detrended Residual (gt100, LOYO CV)',
                 fontsize=18, y=0.99)
    out_path = os.path.join(OUTPUT_DIR, 'multicomp_residual_scatter_v8_style.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')

if __name__ == '__main__':
    plot()
