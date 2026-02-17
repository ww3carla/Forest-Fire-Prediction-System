import pandas as pd

df1 = pd.read_csv('../data/forest_fire_area_dataset/forestfires.csv')
df2 = pd.read_csv('../data/global_fireforest/final_dataset.csv')

# df1=Portugal, df2=Global
# column occured formed from area
df1['occured'] = df1['area'].apply(lambda x: 1 if x > 0 else 0)

df1_mapped = pd.DataFrame()
df1_mapped['temp'] = df1['temp']
df1_mapped['wind'] = df1['wind']
df1_mapped['humidity'] = df1['RH']
df1_mapped['occured'] = df1['occured']

df2_mapped = pd.DataFrame()
df2_mapped['temp'] = df2['temp_mean']
df2_mapped['wind'] = df2['wind_speed_max']
df2_mapped['humidity'] = df2['humidity_min']
df2_mapped['occured'] = df2['occured']

df_combined = pd.concat([df1_mapped, df2_mapped], ignore_index=True)

df_combined.to_csv('data/combined_fire_data.csv', index=False)

print(f"Stats after unification:")
print(f"Total rows: {len(df_combined)}")
print(f"Cases of fire (1): {df_combined['occured'].sum()}")
print(f"Sure cases (0): {len(df_combined) - df_combined['occured'].sum()}")