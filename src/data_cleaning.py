import os

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

# Visualization setup
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# ── DYNAMIC PATH RESOLUTION FOR SERVER COMPATIBILITY ──────────────────
# Get the directory where data_cleaning.py lives (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory of the project (Egypt-Houses-Price)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Construct the relative cloud-safe path to the data folder
CSV_PATH = os.path.join(BASE_DIR, "data", "Egypt_Houses_Price.csv")

# Load the dataframe using the safe path
df = pd.read_csv(CSV_PATH)

def basic_cleaning(dataframe):
    # 1. Remove duplicate rows
    dataframe = dataframe.drop_duplicates()

    # 2. Drop rows where the target variable (Price) is missing
    dataframe = dataframe.dropna(subset=["Price"])

    # 3. Handle Numerical Columns (Convert to numeric, impute with Median)
    num_cols = ["Bedrooms", "Bathrooms", "Area", "Price"]
    for col in num_cols:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
        dataframe[col] = dataframe[col].fillna(dataframe[col].median())

    # 4. Handle Categorical Columns (Impute missing with 'Unknown')
    cat_cols = ["Type", "Furnished", "Level", "Compound", "Payment_Option",
                "Delivery_Date", "Delivery_Term", "City"]
    for col in cat_cols:
        dataframe[col] = dataframe[col].fillna("Unknown")

    # 5. Normalize inconsistent Type labels
    dataframe["Type"] = dataframe["Type"].replace({
        "Standalone Villa": "Stand Alone Villa",
        "Twin house": "Twin House"
    })

    return dataframe

df_clean = basic_cleaning(df)
print(f"✅ Basic cleaning completed. Current shape: {df_clean.shape}")

def remove_outliers_automated(dataframe):
    """
    Remove outliers using percentile-based thresholds.
    Uses both a lower (0.5%) and upper (99%) bound to cut extreme values
    from both ends, which improves model accuracy significantly.
    """
    df_out = dataframe.copy()

    constraints = {
        # Upper bounds – remove top extreme luxury / data errors
        'Price_upper':     df_out['Price'].quantile(0.99),
        'Area_upper':      df_out['Area'].quantile(0.995),
        'Bedrooms_upper':  df_out['Bedrooms'].quantile(0.995),
        'Bathrooms_upper': df_out['Bathrooms'].quantile(0.995),
        # Lower bounds – remove near-zero / likely data errors
        'Price_lower':     df_out['Price'].quantile(0.005),
        'Area_lower':      df_out['Area'].quantile(0.005),
    }

    print("📊 --- Automated Outlier Thresholds Discovered ---")
    for k, v in constraints.items():
        print(f"  ✔️  {k}: {v:,.2f}")

    # Apply filters
    df_out = df_out[
        (df_out['Price']     >= constraints['Price_lower'])  &
        (df_out['Price']     <= constraints['Price_upper'])  &
        (df_out['Area']      >= constraints['Area_lower'])   &
        (df_out['Area']      <= constraints['Area_upper'])   &
        (df_out['Bedrooms']  <= constraints['Bedrooms_upper'])  &
        (df_out['Bathrooms'] <= constraints['Bathrooms_upper'])
    ]

    return df_out

df_filtered = remove_outliers_automated(df_clean)
print(f"🧹 Outliers removed. Shape before: {df_clean.shape}  →  After: {df_filtered.shape}")