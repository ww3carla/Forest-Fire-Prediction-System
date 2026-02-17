import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('../data/forest_fire_area_dataset/forestfires.csv')

df['target'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Acuratețea modelului: {acc * 100:.2f}%")

joblib.dump(model, '../models/fire_model.pkl')
print("Modelul a fost salvat în folderul 'models/'!")