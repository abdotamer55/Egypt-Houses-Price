import os
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # Project root directory
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Path to your 'models' folder

# Define complete file paths
MODEL_PATH = os.path.join(MODELS_DIR, "egypt_houses_best_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "egypt_houses_feature_names.pkl")
ASSETS_PATH = os.path.join(MODELS_DIR, "target_encoding_assets.pkl")

# 2. Load trained model, feature lists, and target encoding assets
try:
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    assets = joblib.load(ASSETS_PATH)
    print("✅ All models and assets loaded successfully from the /models folder!")
except FileNotFoundError:
    raise FileNotFoundError(
        f"❌ Could not find the required .pkl files!\n"
        f"Please verify that they exist at the following location:\n"
        f"📍 {MODELS_DIR}\n"
        f"If they are missing, re-run your training scripts to generate them."
    )


def pipeline_predict(input_data):
    """
    Processes raw dictionary inputs from the UI, applies data preprocessing,
    feature engineering, maps target encodings, and returns the final predicted price.

    Parameters:
    -----------
    input_data : dict
        A dictionary containing user inputs from the Streamlit form.
        Example:
        {
            'Type': 'Apartment', 'Bedrooms': 3, 'Bathrooms': 2, 'Area': 150,
            'Furnished': 'Yes', 'Level': '3', 'Compound': 'Unknown',
            'Payment_Option': 'Cash', 'Delivery_Date': 'Ready to move',
            'Delivery_Term': 'Finished', 'City': 'New Cairo'
        }
    """
    # Convert input dictionary into a single-row DataFrame
    df_input = pd.DataFrame([input_data])

    # ── 1. BASIC DATA CLEANING ───────────────────────────────────────────────
    # Standardize inconsistent property type labels
    df_input["Type"] = df_input["Type"].replace(
        {"Standalone Villa": "Stand Alone Villa", "Twin house": "Twin House"}
    )

    # Coerce numerical columns to correct types
    num_cols = ["Bedrooms", "Bathrooms", "Area"]
    for col in num_cols:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    # ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────
    # Structural features
    df_input["Total_Rooms"] = df_input["Bedrooms"] + df_input["Bathrooms"]
    df_input["Area_per_Room"] = df_input["Area"] / (df_input["Total_Rooms"] + 0.1)
    df_input["Area_per_Bedroom"] = df_input["Area"] / (df_input["Bedrooms"] + 0.1)
    df_input["Bath_to_Bed_Ratio"] = df_input["Bathrooms"] / (
        df_input["Bedrooms"] + 0.1
    )
    df_input["Bed_x_Bath"] = df_input["Bedrooms"] * df_input["Bathrooms"]
    df_input["Area_x_Beds"] = df_input["Area"] * df_input["Bedrooms"]

    # Market status indicators (Binary encoding maps)
    df_input["Is_Furnished"] = (df_input["Furnished"] == "Yes").astype(int)
    df_input["Is_Ready"] = (df_input["Delivery_Date"] == "Ready to move").astype(
        int
    )
    df_input["In_Compound"] = (df_input["Compound"] != "Unknown").astype(int)
    df_input["Is_Compound"] = df_input["In_Compound"]
    df_input["Is_Finished"] = (df_input["Delivery_Term"] == "Finished").astype(int)
    df_input["Is_Semi_Finished"] = (
        df_input["Delivery_Term"] == "Semi Finished"
    ).astype(int)
    df_input["Is_Cash"] = (df_input["Payment_Option"] == "Cash").astype(int)
    df_input["Is_Installment"] = df_input["Payment_Option"].apply(
        lambda x: 1 if "Installment" in str(x) else 0
    )

    # Property type indicators
    df_input["Is_Villa"] = df_input["Type"].apply(
        lambda x: 1 if "Villa" in str(x) else 0
    )
    df_input["Is_Apartment"] = (df_input["Type"] == "Apartment").astype(int)
    df_input["Is_Chalet"] = (df_input["Type"] == "Chalet").astype(int)

    # Level parsing logic
    def clean_level(val):
        val = str(val).strip().lower()
        if "ground" in val:
            return 0
        elif "10+" in val:
            return 10
        elif "unknown" in val:
            return 0
        else:
            try:
                return int(val)
            except ValueError:
                return 0

    df_input["Level"] = df_input["Level"].apply(clean_level)

    # ── 3. TARGET ENCODING MAPPING via SAVED ASSETS ──────────────────────────
    # Retrieve the global fallback training mean log value
    global_mean = assets["global_mean_log"]

    # Map single categorical values to target encoding scores
    for col in assets["te_maps"].keys():
        te_col = col + "_TE"
        mapping = assets["te_maps"][col]
        df_input[te_col] = df_input[col].map(mapping).fillna(global_mean)

    # Map interactive/combined features target encodings
    for name, mapping in assets["inter_maps"].items():
        if name == "City_Type_TE":
            inter_val = df_input["City"] + "_" + df_input["Type"]
        elif name == "City_DT_TE":
            inter_val = df_input["City"] + "_" + df_input["Delivery_Term"]
        df_input[name] = inter_val.map(mapping).fillna(global_mean)

    # Extract regional descriptive aggregations from historical training stats
    city_val = df_input["City"].iloc[0]

    df_input["City_Area_mean"] = assets["city_aggregates"]["City_Area_mean"].get(
        city_val, assets["global_area_mean"]
    )
    df_input["City_Beds_mean"] = assets["city_aggregates"]["City_Beds_mean"].get(
        city_val, assets["global_beds_mean"]
    )
    df_input["City_Price_median"] = assets["city_aggregates"][
        "City_Price_median"
    ].get(city_val, assets["global_price_median"])

    # Derived context indicators
    df_input["Area_vs_City_avg"] = df_input["Area"] / (
        df_input["City_Area_mean"] + 1
    )
    df_input["Log_City_Price_median"] = np.log1p(df_input["City_Price_median"])

    # ── 4. FEATURE SELECTION & PREDICTION ────────────────────────────────────
    # Order final feature set precisely to match structural shape expected by ML model
    X_final = df_input[selected_features]

    # Predict target Log Price
    log_pred = model.predict(X_final)[0]

    # Exponentially revert price transformation to raw EGP metrics (and guard against negatives)
    final_price = np.maximum(np.expm1(log_pred), 0)

    return round(final_price, 2)