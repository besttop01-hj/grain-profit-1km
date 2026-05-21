from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

BASE_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\09_price_output_value_rasterization')
PRICE_DIR = BASE_DIR / 'price_predictions' / 'gt100'
YIELD_DIR = Path(r'F:\zhongliangshouyi\3zuowu_chuli\02_yield_spatialization\predictions\gt100')
OUT_DIR = BASE_DIR / 'main_product_output_value' / 'gt100'
LOG_DIR = BASE_DIR / 'logs'
CROPS = ('wheat', 'maize', 'rice')
YEARS = list(range(2000, 2020))
OUT_NODATA = -9999.0


def get_price_path(crop: str, year: int) -> Path:
    return PRICE_DIR / crop / f'{crop}_price_per_kg_pred_{year}_gt100.tif'


def get_yield_path(crop: str, year: int) -> Path:
    return YIELD_DIR / crop / f'{crop}_yield_pred_{year}_gt100.tif'


def get_output_path(crop: str, year: int) -> Path:
    return OUT_DIR / crop / f'{crop}_main_product_output_value_pred_{year}_gt100.tif'


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows = []

    for crop in CROPS:
        out_crop_dir = OUT_DIR / crop
        out_crop_dir.mkdir(parents=True, exist_ok=True)
        for year in YEARS:
            price_path = get_price_path(crop, year)
            yield_path = get_yield_path(crop, year)
            if not price_path.exists():
                raise FileNotFoundError(f'Missing price raster: {price_path}')
            if not yield_path.exists():
                raise FileNotFoundError(f'Missing yield raster: {yield_path}')

            with rasterio.open(price_path) as p_ds, rasterio.open(yield_path) as y_ds:
                if p_ds.crs != y_ds.crs or p_ds.transform != y_ds.transform or p_ds.width != y_ds.width or p_ds.height != y_ds.height:
                    raise ValueError(f'Spatial mismatch between {price_path} and {yield_path}')
                price_arr = p_ds.read(1).astype(np.float32)
                yield_arr = y_ds.read(1).astype(np.float32)
                profile = p_ds.profile.copy()
                price_nodata = p_ds.nodata if p_ds.nodata is not None else OUT_NODATA
                yield_nodata = y_ds.nodata if y_ds.nodata is not None else OUT_NODATA

            valid = np.isfinite(price_arr) & np.isfinite(yield_arr) & (price_arr != price_nodata) & (yield_arr != yield_nodata)
            out_arr = np.full(price_arr.shape, OUT_NODATA, dtype=np.float32)
            out_arr[valid] = price_arr[valid] * yield_arr[valid]

            out_path = get_output_path(crop, year)
            profile.update(dtype='float32', count=1, compress='lzw', nodata=OUT_NODATA)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(out_arr, 1)

            valid_pixel_count = int(np.count_nonzero(valid))
            inventory_rows.append(
                {
                    'crop': crop,
                    'year': year,
                    'price_path': str(price_path),
                    'yield_path': str(yield_path),
                    'output_path': str(out_path),
                    'valid_pixel_count': valid_pixel_count,
                }
            )
            print(f'done: {crop} | {year} -> {out_path.name}')

    inventory_df = pd.DataFrame(inventory_rows).sort_values(['crop', 'year']).reset_index(drop=True)
    inventory_path = LOG_DIR / 'main_product_output_value_raster_inventory_gt100_2000_2019.csv'
    inventory_df.to_csv(inventory_path, index=False, encoding='utf-8-sig')
    print('inventory_path =', inventory_path)


if __name__ == '__main__':
    main()
