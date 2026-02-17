import pandas as pd

df1 = pd.read_csv('../data/forest_fire_area_dataset/forestfires.csv')

print("First 5 rows of Set1 (Portugal):")
print(df1.head())

print("\nColumn Info")
print(df1.info())

print("\n-Not found values")
print(df1.isnull().sum())