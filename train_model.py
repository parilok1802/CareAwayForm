import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Train synthetic model (आप चाहो तो real Kaggle data use कर सकते हो)
data = pd.DataFrame({
    "age": np.random.randint(18, 80, 500),
    "gender": np.random.choice(["M", "F"], 500),
    "service_type": np.random.choice(["Consultation", "Physiotherapy", "Dentistry"], 500),
    "symptom": np.random.choice(["Pain", "Injury", "Checkup"], 500),
    "history": np.random.choice([0, 1], 500),
    "next_service": np.random.choice(["LabTest", "XRay", "Medication"], 500)
})

X = pd.get_dummies(data[["age", "gender", "service_type", "symptom", "history"]])
y = data["next_service"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and columns
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("✅ Model saved as model.pkl and model_columns.pkl")
