# ========== 1. Import Libraries ==========
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
#from imblearn.over_sampling import SMOTE


# ========== 2. Load Data ==========
df = pd.read_csv("Balanced_Stroke_Data.csv")
X = df.drop("stroke", axis=1)
y = df["stroke"]

# ========== 3. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ========== 6. Train Gradient Boosting Model ==========
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

# ========== 7. Predict and Evaluate ==========
y_pred = gb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("✅ Accuracy:", accuracy)
print("🎯 F1-Score:", f1)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save the Best Model and Scaler
# ===========================
import joblib

joblib.dump(gb_model, 'stroke_model.pkl')

print("\n✅ Model Saved Successfully!")