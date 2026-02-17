import pandas as pd

df_old = pd.read_csv('../data/combined_fire_data.csv')

path_morocco = r'../data/Morocco_Wildfire_Predictions/Date_final_dataset_balanced_float32.parquet'
df_morocco = pd.read_parquet(path_morocco)

mapping = {
    'average_temperature_lag_1': 'temp',
    'wind_speed_lag_1': 'wind',
    'is_fire': 'occured',
    'NDVI': 'vegetation',
    'SoilMoisture': 'soil_moisture'
}

df_mor_mapped = df_morocco[list(mapping.keys())].rename(columns=mapping)

df_old['vegetation'] = 0.5
df_old['soil_moisture'] = 0.3

df_final = pd.concat([df_old, df_mor_mapped], ignore_index=True)

df_final.to_csv('data/master_fire_data.csv', index=False)

print(f"Final dataset is ready!")
print(f"Total rows: {len(df_final)}")