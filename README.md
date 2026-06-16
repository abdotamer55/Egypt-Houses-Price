An end-to-end Machine Learning web application designed to predict residential property prices in Egypt. The project processes raw real estate data through a robust data pipeline—encompassing rigorous cleaning, advanced feature engineering, target encoding to prevent data leakage, and feature importance selection—before serving predictions via an intuitive Streamlit multi-page interface.

---

## 🚀 Features & Architecture

The project is structured into a modular, production-ready pipeline split into distinct operational layers:

1. **Data Cleaning (`data_cleaning.py`)**: Outlier rejection, duplicate filtering, and structural numeric coercion.
2. **Feature Engineering (`featuer_engineering.py`)**: Construction of non-linear interaction features and domain-specific ratios.
3. **Feature Selection & Encoding (`featuer_selection.py`)**: Safe Target Encoding split mechanics to prevent data leakage and Random Forest feature importance filtering.
4. **Model Training & Evaluation (`Model.py`)**: Benchmarking multiple regression algorithms (Ridge, Decision Trees, Random Forest, LightGBM, XGBoost).
5. **Prediction Pipeline (`Predection.py`)**: A backend routing script that passes raw web user input safely through the saved preprocessing assets.
6. **Multi-page UI Application (`App.py`)**: An elegant interactive client interface built using Streamlit.

---

## 📊 Detailed Pipeline Breakdown

### 1. Data Cleaning & Outlier Rejection
* **Handling Missing Values**: Drops missing target instances (`Price`) and standardizes continuous attributes via median statistical imputation.
* **Percentile-Based Trimming**: Extreme luxury properties or pricing anomalies are cut dynamically using defensive percentile filters (Lower: 0.5%, Upper: 99%) across Price, Area, Bedrooms, and Bathrooms to stabilize gradients.

### 2. Advanced Feature Engineering
To extract maximum predictive power from the real estate attributes, several engineering steps were implemented:
* **Ratios & Proportions**: Computed structural metrics such as `Total_Rooms`, `Area_per_Room`, `Area_per_Bedroom`, and `Bath_to_Bed_Ratio`.
* **Interaction Features**: Captured multi-dimensional effects like `Bed_x_Bath` and `Area_x_Beds`.
* **Binary Status Mapping**: Encoded high-signal textual states (`Furnished`, `Delivery_Date`, `Payment_Option`, `Delivery_Term`) into binary indicators (`Is_Furnished`, `Is_Ready`, `Is_Cash`, etc.).
* **Level Harmonization**: Built a custom text parser to harmonize floor descriptions (e.g., mapping "Ground" to `0` and "10+" to `10`).

### 3. Leakage-Free Target Encoding & Selection
* **Target Transformation**: The target variable `Price` is heavily skewed, so a logarithmic transformation ($Log\_Price = \ln(Price + 1)$) is applied to enforce normality.
* **Leakage Avoidance**: Out-of-fold categorical targets (`City`, `Type`, `Compound`, and interaction groups like `City_Type`) are mapped to their respective target means calculated *strictly* from the training partition.
* **Feature Importance Threshold**: A baseline Random Forest Regressor fits the encoded features. Features failing to meet an importance threshold of `0.003` (0.3% variance explanation) are safely pruned.

### 4. Model Benchmarking & Deployment
The final pipeline benchmarks multiple machine learning architectures using $R^2$ Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Continuous evaluation ensures that the model with the highest generalization capability is saved to disk (`egypt_houses_best_model.pkl`).

---

## 📁 Project Structure

```text
Egypt-Houses-Price/
├── App.py                  # The main entry point for the Streamlit application (Home Page)
├── README.md               # Project documentation and architecture overview
├── data/                   # Directory containing raw historical real estate datasets
│   └── Egypt_Houses_Price.csv
├── models/                 # Serialized machine learning models and encoding assets
│   ├── egypt_houses_best_model.pkl
│   ├── egypt_houses_feature_names.pkl
│   └── target_encoding_assets.pkl
└── src/                    # Core source code for the data pipeline and machine learning layers
    ├── data_cleaning.py     # Initial preprocessing, missing value imputation, and outlier filtering
    ├── Featuer_engineering.py  # Production of structural interaction terms and business metrics
    ├── Featuer_selection.py    # Target encoding implementation and Random Forest feature pruning
    ├── Model.py             # Model benchmarking suite, evaluation metrics, and export logic
    └── Predection.py        # Centralized production inference pipeline for live predictions