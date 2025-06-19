import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015 (1).csv")

# 2. Drop kolom target dan fitur yang tidak digunakan
X = df.drop(["Diabetes_012", "Education", "Income", "NoDocbcCost"], axis=1)
y = df["Diabetes_012"]

# 3. Simpan urutan kolom fitur
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# 4. Oversampling dengan SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# 5. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 6. Hitung class weight
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 7. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl", compress=1)

# =======================
# 🔁 Train Random Forest
# =======================
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=17,
    class_weight=class_weight_dict,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("🎯 [Random Forest] Accuracy:", accuracy_score(y_test, y_pred_rf))
print("📊 [Random Forest] Classification Report:\n", classification_report(y_test, y_pred_rf))

# Simpan model Random Forest
joblib.dump(rf, "random_forest_model.pkl", compress=3)

# =========================
# 🌳 Train Decision Tree
# =========================
dt = DecisionTreeClassifier(
    max_depth=17,
    class_weight=class_weight_dict,
    random_state=42
)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print("🎯 [Decision Tree] Accuracy:", accuracy_score(y_test, y_pred_dt))
print("📊 [Decision Tree] Classification Report:\n", classification_report(y_test, y_pred_dt))

# Simpan model Decision Tree
joblib.dump(dt, "decision_tree_model.pkl", compress=3)

# =========================
print("✅ Semua model dan scaler berhasil disimpan.")
