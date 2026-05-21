from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\10_2_net_profit_with_byproduct')
OUT_DIR = BASE_DIR / 'byproduct_output_value' / 'gt100'
LOG_DIR = BASE_DIR / 'logs'
PROVINCE_SHP = Path(r'F:\zhongliangshouyi\1yuanshishuju\2020\province\T2020province.shp')
REF = Path(r'F:\zhongliangshouyi\2chuli\00_target_grid\target_grid_1km_china_albers_v21.tif')
INPUT_TABLE_PATH = Path(r'F:\zhongliangshouyi\3zuowu_chuli\00_crop_distribution_align\10_match_diagnosis\tables\08_model_ready\gt100\province_crop_year_joint_model_base_gt100_2000_2019.csv')
EFFECTIVE_MASK_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\03_effective_crop_mask\gt100')
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


def load_table() -> pd.DataFrame:
    df = pd.read_csv(INPUT_TABLE_PATH)
    df['province_code'] = df['province_code'].astype(str)
    df['byproduct_output_value_y_value_final'] = df['total_output_value_y_value_final'] - df['main_product_output_value_y_value_final']
    return df[['crop', 'year', 'province_code', 'province_name_en_norm', 'byproduct_output_value_y_value_final']].copy()


def get_mask_path(crop: str, year: int) -> Path:
    return EFFECTIVE_MASK_DIR / f'{crop}_effective_mask_{year}_gt100.tif'


def get_output_path(crop: str, year: int) -> Path:
    return OUT_DIR / crop / f'{crop}_byproduct_output_value_pred_{year}_gt100.tif'


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    input_df = load_table()
    province_info, province_masks = build_province_masks()

    baseline_rows = []
    inventory_rows = []

    for crop in CROPS:
        out_crop_dir = OUT_DIR / crop
        out_crop_dir.mkdir(parents=True, exist_ok=True)
        for year in YEARS:
            mask_path = get_mask_path(crop, year)
            if not mask_path.exists():
                raise FileNotFoundError(f'Missing effective mask: {mask_path}')
            with rasterio.open(mask_path) as mask_ds:
                mask_arr = mask_ds.read(1)
                profile = mask_ds.profile.copy()
                profile.update(dtype='float32', count=1, compress='lzw', nodata=OUT_NODATA)
            valid_mask = mask_arr == 1
            pred_arr = np.full(mask_arr.shape, OUT_NODATA, dtype=np.float32)

            year_df = input_df[(input_df['crop'] == crop) & (input_df['year'] == year)].copy()
            for province in province_info.itertuples(index=False):
                province_code = str(province.province_code)
                province_mask = province_masks[province.province_name_en_norm]
                valid = province_mask & valid_mask
                match = year_df[year_df['province_code'] == province_code]
                if match.empty:
                    byproduct_value = np.nan
                else:
                    byproduct_value = float(match['byproduct_output_value_y_value_final'].iloc[0])

                valid_pixel_count = int(np.count_nonzero(valid))
                baseline_rows.append(
                    {
                        'crop': crop,
                        'year': year,
                        'province_code': province_code,
                        'province_name_en_norm': province.province_name_en_norm,
                        'byproduct_output_value_y_value_final': byproduct_value,
                        'valid_pixel_count': valid_pixel_count,
                    }
                )

                if np.any(valid) and np.isfinite(byproduct_value):
                    pred_arr[valid] = np.round(byproduct_value, 2)

            out_path = get_output_path(crop, year)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(pred_arr, 1)
            valid_pixel_count = int(np.count_nonzero(pred_arr != OUT_NODATA))
            inventory_rows.append(
                {
                    'crop': crop,
                    'year': year,
                    'output_path': str(out_path),
                    'valid_pixel_count': valid_pixel_count,
                    'province_count': int((year_df['byproduct_output_value_y_value_final'].notna()).sum()),
                }
            )
            print(f'done: {crop} | {year} -> {out_path.name}')

    baseline_df = pd.DataFrame(baseline_rows).sort_values(['crop', 'year', 'province_code']).reset_index(drop=True)
    baseline_path = LOG_DIR / 'byproduct_output_value_baseline_gt100_2000_2019.csv'
    baseline_df.to_csv(baseline_path, index=False, encoding='utf-8-sig')

    inventory_df = pd.DataFrame(inventory_rows).sort_values(['crop', 'year']).reset_index(drop=True)
    inventory_path = LOG_DIR / 'byproduct_output_value_inventory_gt100_2000_2019.csv'
    inventory_df.to_csv(inventory_path, index=False, encoding='utf-8-sig')

    print('baseline_path =', baseline_path)
    print('inventory_path =', inventory_path)


if __name__ == '__main__':
    main()
