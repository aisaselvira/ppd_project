import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
# 🔍 Tambahkan evaluasi akurasi di sini
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015 (1).csv")

# Drop kolom target dan fitur yang tidak digunakan
X = df.drop(["Diabetes_012", "Education", "Income", "NoDocbcCost"], axis=1)
y = df["Diabetes_012"]

# Simpan urutan kolom fitur
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Oversample data
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Hitung class weight
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl", compress=1)
# Train Random Forest (versi middle)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=17,
    class_weight=class_weight_dict,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)
y_pred = rf.predict(X_test_scaled)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 💾 Simpan model setelah evaluasi
joblib.dump(rf, "random_forest_model.pkl", compress=3)

print("✅ Model dan scaler berhasil disimpan.")
