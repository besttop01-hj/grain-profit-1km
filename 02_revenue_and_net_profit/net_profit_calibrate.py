from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\10_2_net_profit_with_byproduct')
PRED_DIR = BASE_DIR / 'predictions' / 'gt100'
CHECK_DIR = BASE_DIR / 'check'
CALIBRATED_DIR = CHECK_DIR / 'calibrated' / 'gt100'
PROVINCE_SHP = Path(r'F:\zhongliangshouyi\1yuanshishuju\2020\province\T2020province.shp')
REF = Path(r'F:\zhongliangshouyi\2chuli\00_target_grid\target_grid_1km_china_albers_v21.tif')
JOINT_MODEL_BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\00_crop_distribution_align\10_match_diagnosis\tables\08_model_ready\gt100')
CROPS = ('wheat', 'maize', 'rice')
YEARS = list(range(2000, 2020))
OUT_NODATA = -9999.0
PROVINCE_NAME_FIX = {
    'Neimenggu': 'Inner Mongolia',
    'Xizang': 'Tibet',
    'Xinjiang Uygur': 'Xinjiang',
    'Ningxia Hui': 'Ningxia',
    'Guangxi Zhuang': 'Guangxi',
}


def normalize_province_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        return name
    return PROVINCE_NAME_FIX.get(name, name)


def build_province_masks() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    gdf = gpd.read_file(PROVINCE_SHP)
    with rasterio.open(REF) as ref:
        gdf = gdf.to_crs(ref.crs)
        gdf['province_code'] = gdf['FIRST_GID'].astype(str).str.strip()
        gdf['province_name_en_from_shp'] = gdf['ENG_NAME'].astype(str).str.strip()
        gdf['province_name_en_norm'] = gdf['province_name_en_from_shp'].map(normalize_province_name)
        province_masks = {}
        for _, row in gdf.iterrows():
            province_masks[row['province_name_en_norm']] = rasterize(
                [(row.geometry, 1)],
                out_shape=(ref.height, ref.width),
                transform=ref.transform,
                fill=0,
                dtype='uint8',
                all_touched=False,
            ) == 1
    province_info = gdf[['province_code', 'province_name_en_norm']].drop_duplicates()
    return province_info, province_masks


def get_pred_path(crop: str, year: int) -> Path:
    return PRED_DIR / crop / f'{crop}_net_profit_full_pred_{year}_gt100.tif'


def get_out_path(crop: str, year: int) -> Path:
    return CALIBRATED_DIR / crop / f'{crop}_net_profit_full_calibrated_{year}_gt100.tif'


def load_yearbook() -> pd.DataFrame:
    rows = []
    for crop in CROPS:
        path = JOINT_MODEL_BASE_DIR / f'{crop}_joint_model_base_gt100_2000_2019.csv'
        df = pd.read_csv(path)
        df['province_code'] = df['province_code'].astype(str)
        rows.append(df[['crop', 'year', 'province_code', 'province_name_en_norm', 'net_profit_y_value_final']])
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={'net_profit_y_value_final': 'target_net_profit_per_mu'})
    return out


def main() -> None:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)
    yearbook_df = load_yearbook()
    province_info, province_masks = build_province_masks()

    inventory_rows = []
    validation_rows = []

    for crop in CROPS:
        out_crop_dir = CALIBRATED_DIR / crop
        out_crop_dir.mkdir(parents=True, exist_ok=True)
        for year in YEARS:
            pred_path = get_pred_path(crop, year)
            if not pred_path.exists():
                raise FileNotFoundError(f'Missing pre-calibration raster: {pred_path}')
            with rasterio.open(pred_path) as ds:
                arr = ds.read(1).astype(np.float32)
                profile = ds.profile.copy()
                nodata = ds.nodata if ds.nodata is not None else OUT_NODATA
            valid_arr = np.isfinite(arr) & (arr != nodata)
            calibrated_arr = arr.copy()

            yearbook_sub = yearbook_df[(yearbook_df['crop'] == crop) & (yearbook_df['year'] == year)].copy()
            for province in province_info.itertuples(index=False):
                province_code = str(province.province_code)
                province_mask = province_masks[province.province_name_en_norm]
                valid = province_mask & valid_arr
                pre_mean = float(arr[valid].mean()) if np.any(valid) else np.nan
                yb_match = yearbook_sub[yearbook_sub['province_code'] == province_code]
                if yb_match.empty:
                    target_value = np.nan
                else:
                    target_value = float(yb_match['target_net_profit_per_mu'].iloc[0])

                if np.any(valid) and np.isfinite(pre_mean) and np.isfinite(target_value):
                    delta = target_value - pre_mean
                    calibrated_arr[valid] = calibrated_arr[valid] + delta
                    post_mean = float(calibrated_arr[valid].mean())
                else:
                    delta = np.nan
                    post_mean = np.nan

                validation_rows.append(
                    {
                        'crop': crop,
                        'year': year,
                        'province_code': province_code,
                        'province_name_en_norm': province.province_name_en_norm,
                        'pre_calibration_mean': pre_mean,
                        'target_net_profit_per_mu': target_value,
                        'additive_delta': delta,
                        'post_calibration_mean': post_mean,
                        'difference_post_minus_target': post_mean - target_value if np.isfinite(post_mean) and np.isfinite(target_value) else np.nan,
                    }
                )

            out_path = get_out_path(crop, year)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(calibrated_arr, 1)
            valid_pixel_count = int(np.count_nonzero(valid_arr))
            inventory_rows.append(
                {
                    'crop': crop,
                    'year': year,
                    'input_pred_path': str(pred_path),
                    'output_calibrated_path': str(out_path),
                    'valid_pixel_count': valid_pixel_count,
                }
            )
            print(f'done: {crop} | {year} -> {out_path.name}')

    inventory_df = pd.DataFrame(inventory_rows).sort_values(['crop', 'year']).reset_index(drop=True)
    inventory_path = CHECK_DIR / 'net_profit_full_calibrated_inventory_gt100_2000_2019.csv'
    inventory_df.to_csv(inventory_path, index=False, encoding='utf-8-sig')

    validation_df = pd.DataFrame(validation_rows).sort_values(['crop', 'year', 'province_code']).reset_index(drop=True)
    validation_path = CHECK_DIR / 'net_profit_full_calibrated_validation_gt100_2000_2019.csv'
    validation_df.to_csv(validation_path, index=False, encoding='utf-8-sig')

    print('inventory_path =', inventory_path)
    print('validation_path =', validation_path)


if __name__ == '__main__':
    main()
