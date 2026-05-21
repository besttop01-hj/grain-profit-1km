from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\07_total_cost_detrended_modeling')
RESULTS_DIR = BASE_DIR / 'results' / 'gt100'
TABLES_DIR = BASE_DIR / 'tables' / 'gt100'
CROPS = ('wheat', 'maize', 'rice')
MODEL_PARAMS = {
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': 4,
}


def run_loyo(df: pd.DataFrame, crop: str):
    x_cols = [col for col in df.columns if col.startswith('x_')]
    years = sorted(df['year'].dropna().unique())

    prediction_rows = []
    fold_rows = []
    gain_sum = pd.Series(0.0, index=x_cols, dtype=float)

    for fold_idx, held_out_year in enumerate(years, start=1):
        train_df = df[df['year'] != held_out_year].copy()
        test_df = df[df['year'] == held_out_year].copy()
        if train_df.empty or test_df.empty:
            continue

        x_train = train_df[x_cols]
        y_train = train_df['total_cost_residual']
        x_test = test_df[x_cols]
        y_test = test_df['total_cost_residual']

        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(x_train, y_train)
        y_pred_resid = model.predict(x_test)
        y_pred_total_cost = y_pred_resid + test_df['national_year_mean_total_cost'].to_numpy()

        for row, pred_resid, pred_total in zip(test_df.itertuples(index=False), y_pred_resid, y_pred_total_cost):
            prediction_rows.append(
                {
                    'sample_id': row.sample_id,
                    'crop': row.crop,
                    'year': row.year,
                    'province_code': row.province_code,
                    'province_name_en_norm': row.province_name_en_norm,
                    'threshold_scheme': row.threshold_scheme,
                    'y_true_residual': row.total_cost_residual,
                    'y_pred_residual': float(pred_resid),
                    'residual_error': float(row.total_cost_residual - pred_resid),
                    'national_year_mean_total_cost': row.national_year_mean_total_cost,
                    'y_true_total_cost': row.y_value,
                    'y_pred_total_cost': float(pred_total),
                    'total_cost_error': float(row.y_value - pred_total),
                    'fold_held_out_year': int(held_out_year),
                }
            )

        fold_rows.append(
            {
                'crop': crop,
                'threshold_scheme': 'gt100',
                'fold_idx': fold_idx,
                'held_out_year': int(held_out_year),
                'test_sample_count': len(test_df),
                'residual_r2': float(r2_score(y_test, y_pred_resid)) if len(test_df) > 1 else np.nan,
                'residual_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_resid))),
                'residual_mae': float(mean_absolute_error(y_test, y_pred_resid)),
                'total_cost_r2': float(r2_score(test_df['y_value'], y_pred_total_cost)) if len(test_df) > 1 else np.nan,
                'total_cost_rmse': float(np.sqrt(mean_squared_error(test_df['y_value'], y_pred_total_cost))),
                'total_cost_mae': float(mean_absolute_error(test_df['y_value'], y_pred_total_cost)),
            }
        )

        gain = pd.Series(model.feature_importances_, index=x_cols, dtype=float)
        gain_sum = gain_sum.add(gain, fill_value=0.0)
        print(f'[gt100][{crop}] LOYO fold {fold_idx}/{len(years)} | held_out_year={held_out_year} | test_n={len(test_df)}')

    predictions_df = pd.DataFrame(prediction_rows).sort_values(['year', 'province_code']).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(['fold_idx']).reset_index(drop=True)
    importance_df = pd.DataFrame(
        {
            'crop': crop,
            'threshold_scheme': 'gt100',
            'feature_name': x_cols,
            'importance_gain': (gain_sum / max(len(fold_df), 1)).values,
        }
    ).sort_values('importance_gain', ascending=False).reset_index(drop=True)
    return predictions_df, fold_df, importance_df


def build_metrics(predictions_df: pd.DataFrame, crop: str) -> pd.DataFrame:
    metrics = {
        'crop': crop,
        'threshold_scheme': 'gt100',
        'sample_count': len(predictions_df),
        'residual_r2': float(r2_score(predictions_df['y_true_residual'], predictions_df['y_pred_residual'])),
        'residual_rmse': float(np.sqrt(mean_squared_error(predictions_df['y_true_residual'], predictions_df['y_pred_residual']))),
        'residual_mae': float(mean_absolute_error(predictions_df['y_true_residual'], predictions_df['y_pred_residual'])),
        'total_cost_r2': float(r2_score(predictions_df['y_true_total_cost'], predictions_df['y_pred_total_cost'])),
        'total_cost_rmse': float(np.sqrt(mean_squared_error(predictions_df['y_true_total_cost'], predictions_df['y_pred_total_cost']))),
        'total_cost_mae': float(mean_absolute_error(predictions_df['y_true_total_cost'], predictions_df['y_pred_total_cost'])),
    }
    return pd.DataFrame([metrics])


def main() -> None:
    for crop in CROPS:
        crop_result_dir = RESULTS_DIR / crop
        crop_result_dir.mkdir(parents=True, exist_ok=True)
        input_path = TABLES_DIR / f'{crop}_model_input_total_cost_residual_gt100_2000_2019.csv'
        df = pd.read_csv(input_path)

        predictions_df, fold_df, importance_df = run_loyo(df, crop)
        metrics_df = build_metrics(predictions_df, crop)

        predictions_path = crop_result_dir / f'total_cost_residual_predictions_gt100_{crop}_2000_2019.csv'
        folds_path = crop_result_dir / f'total_cost_residual_fold_metrics_gt100_{crop}_2000_2019.csv'
        metrics_path = crop_result_dir / f'total_cost_residual_metrics_gt100_{crop}_2000_2019.csv'
        importance_path = crop_result_dir / f'total_cost_residual_feature_importance_gt100_{crop}_2000_2019.csv'

        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        fold_df.to_csv(folds_path, index=False, encoding='utf-8-sig')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')

        print('predictions_path =', predictions_path)
        print('folds_path =', folds_path)
        print('metrics_path =', metrics_path)
        print('importance_path =', importance_path)
        print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
