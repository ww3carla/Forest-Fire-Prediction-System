import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

start_time = time.time()

print("--- Initializing Master Model Training ---")
print("Loading dataset with over 1 million rows... Please wait.")

try:
    df = pd.read_csv('../data/master_fire_data.csv')
    df = df.dropna()
    print(f"Dataset loaded successfully. Total samples: {len(df)}")
except FileNotFoundError:
    print("Error: 'master_fire_data.csv' not found in data/ directory.")
    exit()

features = ['temp', 'wind', 'humidity', 'vegetation', 'soil_moisture']
X = df[features]
y = df['occured']

# split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print("\nStarting RandomForest training using all available CPU cores...")

model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("Training complete. Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
duration = round((end_time - start_time) / 60, 2)

print("\n" + "="*30)
print("       FINAL MODEL RESULTS       ")
print("="*30)
print(f"Master Model Accuracy: {accuracy * 100:.2f}%")
print(f"Total Training Time: {duration} minutes")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, '../models/fire_model.pkl')
print("="*30)
print("SUCCESS: Master model saved to 'models/fire_model.pkl'")
print("="*30)