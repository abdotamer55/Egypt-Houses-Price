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
from sklearn.preprocessing import LabelEncoder
from data_cleaning import df_filtered

# Visualization setup
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

def engineer_features(dataframe):
    """
    Build structural + behavioral features and prepare target encoding helpers.
    NOTE: Target Encoding (TE) is applied AFTER the train/test split
    in the next section to prevent data leakage.
    """
    df_out = dataframe.copy().reset_index(drop=True)

    # ── 1. Structural Features ──────────────────────────────────────────────
    df_out['Total_Rooms']      = df_out['Bedrooms'] + df_out['Bathrooms']
    df_out['Area_per_Room']    = df_out['Area'] / (df_out['Total_Rooms'] + 0.1)
    df_out['Area_per_Bedroom'] = df_out['Area'] / (df_out['Bedrooms']    + 0.1)
    df_out['Bath_to_Bed_Ratio']= df_out['Bathrooms'] / (df_out['Bedrooms'] + 0.1)
    df_out['Bed_x_Bath']       = df_out['Bedrooms'] * df_out['Bathrooms']
    df_out['Area_x_Beds']      = df_out['Area']     * df_out['Bedrooms']

    # ── 2. Market Status Indicators (Binary Encoding) ──────────────────────
    df_out['Is_Furnished']    = (df_out['Furnished']     == 'Yes').astype(int)
    df_out['Is_Ready']        = (df_out['Delivery_Date'] == 'Ready to move').astype(int)
    df_out['In_Compound']     = (df_out['Compound']      != 'Unknown').astype(int)
    df_out['Is_Compound']     = df_out['In_Compound']   # backward compat
    df_out['Is_Finished']     = (df_out['Delivery_Term'] == 'Finished').astype(int)
    df_out['Is_Semi_Finished']= (df_out['Delivery_Term'] == 'Semi Finished').astype(int)
    df_out['Is_Cash']         = (df_out['Payment_Option']== 'Cash').astype(int)
    df_out['Is_Installment']  = df_out['Payment_Option'].apply(
        lambda x: 1 if 'Installment' in str(x) else 0)

    # ── 3. Property Type Flags ──────────────────────────────────────────────
    df_out['Is_Villa']     = df_out['Type'].apply(lambda x: 1 if 'Villa' in str(x) else 0)
    df_out['Is_Apartment'] = (df_out['Type'] == 'Apartment').astype(int)
    df_out['Is_Chalet']    = (df_out['Type'] == 'Chalet').astype(int)

    # ── 4. Level Parsing ───────────────────────────────────────────────────
    def clean_level(val):
        val = str(val).strip().lower()
        if 'ground' in val: return 0
        elif '10+'   in val: return 10
        elif 'unknown' in val: return 0
        else:
            try:    return int(val)
            except: return 0

    df_out['Level'] = df_out['Level'].apply(clean_level)

    # ── 5. Target Variable Transformation (fixes price skewness) ───────────
    df_out['Log_Price'] = np.log1p(df_out['Price'])

    return df_out

df_featured = engineer_features(df_filtered)
print("🚀 Feature engineering completed successfully.")
print(f"   Shape: {df_featured.shape}")