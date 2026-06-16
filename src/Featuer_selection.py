import pandas as pd
import numpy as np
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
from Featuer_engineering import df_featured

# Visualization setup
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

CATEGORICAL_TE_COLS  = ["City", "Type", "Delivery_Term", "Payment_Option", "Compound"]
CATEGORICAL_DROP_COLS = ["Furnished", "Delivery_Date"]  # already binary-encoded

X = df_featured.drop(columns=["Price", "Log_Price"] + CATEGORICAL_DROP_COLS)
y_log = df_featured["Log_Price"]
y_raw = df_featured["Price"]

X_train, X_test, y_log_train, y_log_test, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.20,
    random_state=42
)

print(f"📊 Train Shape : {X_train.shape}")
print(f"📊 Test  Shape : {X_test.shape}")


KF = KFold(n_splits=5, shuffle=True, random_state=42)

def apply_target_encoding(X_tr, X_te, y_tr, group_cols, global_mean):
    """
    Fit target encoding on X_tr / y_tr using 5-fold CV to avoid leakage,
    then apply the full-train mean map to X_te.
    Returns copies of X_tr and X_te with new *_TE columns added.
    """
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    # ── Single-column TEs ──────────────────────────────────────────────────
    for col in group_cols:
        te_col = col + "_TE"
        X_tr[te_col] = 0.0
        for tr_idx, val_idx in KF.split(X_tr):
            mean_map = y_tr.iloc[tr_idx].groupby(X_tr[col].iloc[tr_idx]).mean()
            X_tr.iloc[val_idx, X_tr.columns.get_loc(te_col)] = (
                X_tr[col].iloc[val_idx].map(mean_map).fillna(global_mean).values
            )
        # Apply the full-train mean to the test set
        full_map = y_tr.groupby(X_tr[col]).mean()
        X_te[te_col] = X_te[col].map(full_map).fillna(global_mean)

    # ── Interaction TEs ────────────────────────────────────────────────────
    for g1, g2, name in [("City", "Type", "City_Type_TE"),
                          ("City", "Delivery_Term", "City_DT_TE")]:
        inter_tr = X_tr[g1] + "_" + X_tr[g2]
        inter_te = X_te[g1] + "_" + X_te[g2]
        X_tr[name] = 0.0
        for tr_idx, val_idx in KF.split(X_tr):
            mean_map = y_tr.iloc[tr_idx].groupby(inter_tr.iloc[tr_idx]).mean()
            X_tr.iloc[val_idx, X_tr.columns.get_loc(name)] = (
                inter_tr.iloc[val_idx].map(mean_map).fillna(global_mean).values
            )
        full_map = y_tr.groupby(inter_tr).mean()
        X_te[name] = inter_te.map(full_map).fillna(global_mean)

    # ── City-level aggregate stats ─────────────────────────────────────────
    # (also computed from training data only to avoid leakage)
    for stat_col, stat_fn, out_col in [
        ("Area",  "mean",   "City_Area_mean"),
        ("Price",  "median", "City_Price_median"),
        ("Bedrooms", "mean", "City_Beds_mean"),
    ]:
        # We need the raw price column only for median; for others use X cols
        agg_src = X_tr[stat_col] if stat_col in X_tr.columns else None
        # Price is not in X – use y_train_raw proxy via index alignment
        if stat_col == "Price":
            # approximate: re-exponentiate Log_Price
            agg_series = np.expm1(y_tr)
            agg_series.index = X_tr.index
            full_agg = agg_series.groupby(X_tr["City"]).agg(stat_fn)
            X_tr[out_col] = 0.0
            for tr_idx, val_idx in KF.split(X_tr):
                agg_ = agg_series.iloc[tr_idx].groupby(X_tr["City"].iloc[tr_idx]).agg(stat_fn)
                X_tr.iloc[val_idx, X_tr.columns.get_loc(out_col)] = (
                    X_tr["City"].iloc[val_idx].map(agg_).fillna(agg_series.mean()).values
                )
        else:
            full_agg = X_tr.groupby("City")[stat_col].agg(stat_fn)
            X_tr[out_col] = 0.0
            for tr_idx, val_idx in KF.split(X_tr):
                agg_ = X_tr.iloc[tr_idx].groupby("City")[stat_col].agg(stat_fn)
                X_tr.iloc[val_idx, X_tr.columns.get_loc(out_col)] = (
                    X_tr["City"].iloc[val_idx].map(agg_).fillna(X_tr[stat_col].mean()).values
                )
        X_te[out_col] = X_te["City"].map(full_agg).fillna(X_tr[out_col].mean())

    # ── Derived ratio feature ──────────────────────────────────────────────
    X_tr["Area_vs_City_avg"] = X_tr["Area"] / (X_tr["City_Area_mean"] + 1)
    X_te["Area_vs_City_avg"] = X_te["Area"] / (X_te["City_Area_mean"] + 1)
    X_tr["Log_City_Price_median"] = np.log1p(X_tr["City_Price_median"])
    X_te["Log_City_Price_median"] = np.log1p(X_te["City_Price_median"])

    # ── Drop raw categorical columns (replaced by TEs) ────────────────────
    drop_cats = [c for c in group_cols if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_cats)
    X_te = X_te.drop(columns=drop_cats)

    return X_tr, X_te

global_mean_log = y_log_train.mean()

X_train_enc, X_test_enc = apply_target_encoding(
    X_train.copy(), X_test.copy(), y_log_train,
    CATEGORICAL_TE_COLS, global_mean_log
)

print("✅ Target Encoding applied.")
print(f"   Train shape : {X_train_enc.shape}")
print(f"   Test  shape : {X_test_enc.shape}")
print()

# ── Feature Importance via baseline RF ────────────────────────────────────
print("⏳ Computing feature importances …")
selector_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
selector_model.fit(X_train_enc, y_log_train)

importance_df = pd.DataFrame({
    "Feature":    X_train_enc.columns,
    "Importance": selector_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            hue="Feature", palette="viridis", legend=False)
plt.title("Feature Importance (Random Forest on Encoded Features)", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.axvline(x=0.003, color="red", linestyle="--", label="Threshold (0.3%)")
plt.legend()
plt.tight_layout()
plt.show()

# ── Apply selection threshold ──────────────────────────────────────────────
threshold = 0.003
selected_features = importance_df[
    importance_df["Importance"] >= threshold
]["Feature"].tolist()

X_train_final = X_train_enc[selected_features].copy()
X_test_final  = X_test_enc[selected_features].copy()

print(f"🎯 Total features evaluated : {X_train_enc.shape[1]}")
print(f"   Features selected         : {len(selected_features)}")
dropped = list(set(X_train_enc.columns) - set(selected_features))
print(f"   Dropped (below 0.3%)      : {dropped}")
print(f"Final feature list:{selected_features}")