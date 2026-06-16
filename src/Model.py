import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import joblib
from Featuer_selection import (
    X_train_final, X_test_final, y_log_train, y_log_test, y_test_raw, 
    selected_features, global_mean_log, X_train, CATEGORICAL_TE_COLS
)

# Visualization setup
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

models = {
    "Ridge Regression": Ridge(alpha=10.0),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=15,
        min_samples_split=10
    ),

    "Tuned Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.6,
        random_state=42,
        n_jobs=-1
    ),

    "Advanced Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    ),

    "LightGBM": lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=9,
        num_leaves=200,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.05,
        random_state=42,
        verbose=-1
    ),

    "XGBoost": xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    ),
}

predictions   = {}
trained_models = {}

print("⏳ Training Models …")

for name, model in models.items():
    # XGBoost needs eval_set for early stopping
    if name == "XGBoost":
        model.fit(
            X_train_final, y_log_train,
            eval_set=[(X_test_final, y_log_test)],
            verbose=False
        )
    elif name == "LightGBM":
        model.fit(
            X_train_final, y_log_train,
            eval_set=[(X_test_final, y_log_test)]
        )
    else:
        model.fit(X_train_final, y_log_train)

    trained_models[name] = model

    log_pred = model.predict(X_test_final)
    y_pred   = np.maximum(np.expm1(log_pred), 0)   # back-transform; clip negatives
    predictions[name] = y_pred

    print(f"✅ {name} done")

print("🎉 All models trained successfully")

# ── Evaluation ────────────────────────────────────────────────────────────────
results = []
for name, y_pred in predictions.items():
    r2   = r2_score(y_test_raw, y_pred)
    mae  = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    results.append({
        "Model":      name,
        "R2 Score":   round(r2,   4),
        "MAE (EGP)":  round(mae,  0),
        "RMSE (EGP)": round(rmse, 0)
    })

comparison_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)

print("🚀 FINAL MODEL COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# ── Save best model ───────────────────────────────────────────────────────────
best_model_name = comparison_df.iloc[0]["Model"]
best_model      = trained_models[best_model_name]

joblib.dump(best_model,        "egypt_houses_best_model.pkl")
joblib.dump(selected_features, "egypt_houses_feature_names.pkl")

print(f"🏆 Best Model  : {best_model_name}")
print(f"⭐ Best R²      : {comparison_df.iloc[0]['R2 Score']}")
print("💾 Model and feature list saved successfully")

# 1. حساب الـ Full Mapping من الـ Training Set بأكملها (بدون الـ CV لأننا بنسيف للإنتاج)
te_maps = {}
for col in CATEGORICAL_TE_COLS:
    te_maps[col] = y_log_train.groupby(X_train[col]).mean().to_dict()

# 2. حساب الـ Interaction TEs Mapping
inter_maps = {}
for g1, g2, name in [("City", "Type", "City_Type_TE"), ("City", "Delivery_Term", "City_DT_TE")]:
    inter_series = X_train[g1] + "_" + X_train[g2]
    inter_maps[name] = y_log_train.groupby(inter_series).mean().to_dict()

# 3. حساب الـ City aggregates المعتمدة على الـ Training Data
city_aggregates = {
    "City_Area_mean": X_train.groupby("City")["Area"].mean().to_dict(),
    "City_Beds_mean": X_train.groupby("City")["Bedrooms"].mean().to_dict(),
}
# بالنسبة للميديان الخاص بالسعر (باستخدام الـ proxy)
agg_series = np.expm1(y_log_train)
agg_series.index = X_train.index
city_aggregates["City_Price_median"] = agg_series.groupby(X_train["City"]).median().to_dict()

# سيف كل الحاجات دي في فايل واحد للـ Pipeline
pipeline_assets = {
    "te_maps": te_maps,
    "inter_maps": inter_maps,
    "city_aggregates": city_aggregates,
    "global_mean_log": global_mean_log,
    "global_area_mean": X_train["Area"].mean(),
    "global_beds_mean": X_train["Bedrooms"].mean(),
    "global_price_median": agg_series.median()
}

joblib.dump(pipeline_assets, "target_encoding_assets.pkl")
print("💾 Saved Target Encoding assets successfully!")