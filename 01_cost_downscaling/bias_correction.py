from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from xgboost import XGBRegressor

BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\08_total_cost_rasterization')
PRED_DIR = BASE_DIR / 'predictions' / 'gt100'
LOG_DIR = BASE_DIR / 'logs'
MODEL_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\07_total_cost_detrended_modeling\model_objects\gt100')
BASELINE_PATH = Path(r'F:\zhongliangshouyi\3zuowu_chuli\07_total_cost_detrended_modeling\tables\final_model\gt100_total_cost_year_baseline.csv')
MASK_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\03_effective_crop_mask\gt100')
X_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\04_x_mask_by_effective_crop\masked_v21\gt100')
REF = Path(r'F:\zhongliangshouyi\2chuli\00_target_grid\target_grid_1km_china_albers_v21.tif')
CROPS = ('wheat', 'maize', 'rice')
YEARS = list(range(2000, 2020))
FEATURES = [
    'x_slope_mean',
    'x_dist_road_mean',
    'x_dist_city_mean',
    'x_dist_water_mean',
    'x_soc_mean',
    'x_ph_mean',
    'x_tn_mean',
    'x_irrigation_cirrmap250_mean',
    'x_ed_cropland_mean',
    'x_ssm_surface_mean',
    'x_tvdi_mean',
    'x_nightlight_longntl_mean',
    'x_pop_density_worldpop_mean',
    'x_precip_anomaly_mean',
    'x_gdd_mean',
]
FEATURE_TO_LAYER = {
    'x_slope_mean': 'slope',
    'x_dist_road_mean': 'dist_road',
    'x_dist_city_mean': 'dist_city',
    'x_dist_water_mean': 'dist_water',
    'x_soc_mean': 'soc',
    'x_ph_mean': 'ph',
    'x_tn_mean': 'tn',
    'x_irrigation_cirrmap250_mean': 'irrigation_cirrmap250',
    'x_ed_cropland_mean': 'ed_cropland',
    'x_ssm_surface_mean': 'ssm_surface',
    'x_tvdi_mean': 'tvdi',
    'x_nightlight_longntl_mean': 'nightlight_longntl',
    'x_pop_density_worldpop_mean': 'pop_density_worldpop',
    'x_precip_anomaly_mean': 'precip_anomaly',
    'x_gdd_mean': 'gdd',
}
OUT_NODATA = -9999.0


def get_mask_path(crop: str, year: int) -> Path:
    return MASK_DIR / f'{crop}_effective_mask_{year}_gt100.tif'


def get_x_path(crop: str, year: int, feature_name: str) -> Path:
    layer = FEATURE_TO_LAYER[feature_name]
    return X_DIR / crop / layer / f'{layer}_{crop}_{year}_effective_masked_gt100_v21.tif'


def get_model_path(crop: str) -> Path:
    return MODEL_DIR / f'gt100_{crop}_total_cost_residual_xgb_model.json'


def load_baseline() -> pd.DataFrame:
    return pd.read_csv(BASELINE_PATH)


def predict_one_crop_year(crop: str, year: int, baseline_lookup: dict[tuple[str, int], float]) -> tuple[Path, dict]:
    mask_path = get_mask_path(crop, year)
    if not mask_path.exists():
        raise FileNotFoundError(f'Missing effective mask: {mask_path}')

    with rasterio.open(mask_path) as mask_ds:
        mask_arr = mask_ds.read(1)
        profile = mask_ds.profile.copy()
        valid_mask = mask_arr == 1
        height = mask_ds.height
        width = mask_ds.width

    feature_arrays = []
    valid = valid_mask.copy()
    for feature_name in FEATURES:
        x_path = get_x_path(crop, year, feature_name)
        if not x_path.exists():
            raise FileNotFoundError(f'Missing masked X raster: {x_path}')
        with rasterio.open(x_path) as ds:
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        layer_valid = np.isfinite(arr)
        if nodata is not None:
            layer_valid &= arr != nodata
        valid &= layer_valid
        feature_arrays.append(arr)

    model = XGBRegressor()
    model.load_model(str(get_model_path(crop)))

    pred_arr = np.full((height, width), OUT_NODATA, dtype=np.float32)
    if np.any(valid):
        X_stack = np.column_stack([arr[valid] for arr in feature_arrays])
        pred_resid = model.predict(X_stack)
        year_mean = baseline_lookup[(crop, year)]
        pred_total = pred_resid + year_mean
        pred_arr[valid] = pred_total.astype(np.float32)

    out_dir = PRED_DIR / crop
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{crop}_total_cost_pred_{year}_gt100.tif'
    profile.update(dtype='float32', count=1, compress='lzw', nodata=OUT_NODATA)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(pred_arr, 1)

    valid_pixel_count = int(np.count_nonzero(valid))
    return out_path, {
        'crop': crop,
        'year': year,
        'output_path': str(out_path),
        'valid_pixel_count': valid_pixel_count,
        'year_mean_total_cost': baseline_lookup[(crop, year)],
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    baseline_df = load_baseline()
    baseline_lookup = {
        (row.crop, int(row.year)): float(row.national_year_mean_total_cost)
        for row in baseline_df.itertuples(index=False)
    }

    log_rows = []
    for crop in CROPS:
        for year in YEARS:
            out_path, log_row = predict_one_crop_year(crop, year, baseline_lookup)
            log_rows.append(log_row)
            print(f'done: {crop} | {year} -> {out_path.name}')

    log_df = pd.DataFrame(log_rows).sort_values(['crop', 'year']).reset_index(drop=True)
    log_path = LOG_DIR / 'gt100_total_cost_raster_inventory_2000_2019.csv'
    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    print('log_path =', log_path)


if __name__ == '__main__':
    main()
