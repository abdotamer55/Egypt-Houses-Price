# Egypt House Price Prediction

A Python project combining Egyptian property-data analysis, regression modeling, a Streamlit prediction interface and a Power BI report.

[Explore the notebook](notebooks/eda_improved.ipynb) · [Open the Power BI file](dashboard/Dashboard%20project.pbix) · [Application source](App.py)

## Problem Statement

Property prices vary with location, size, property type and other listing attributes. This project explores those relationships and builds a regression workflow for estimating prices from the available features.

## Methodology

1. **Clean the source data:** standardize numeric fields, handle missing values and trim extreme values.
2. **Engineer features:** create room totals, area ratios, interactions and indicators for listing attributes.
3. **Encode and select:** split the data, apply out-of-fold target encoding on the training partition and use feature importance for selection.
4. **Compare regressors:** the training code includes Ridge, decision-tree, random-forest, LightGBM and XGBoost models, evaluated with regression metrics.
5. **Serve predictions:** load the saved model, selected features and encoding assets through the Streamlit interface.
6. **Explore visually:** use the notebook and included Power BI report for further inspection.

The report contains Dashboard, Overview and Prediction pages, alongside navigation and tooltip pages. No report screenshot is included here because the repository did not contain a verified report capture.

## Insights

The committed cleaned CSV contains **25,111 rows and 12 columns**. Descriptive statistics computed from that file:

| Field | Median | Mean |
| --- | ---: | ---: |
| Price, in dataset units | 2,750,000 | 4,379,580.71 |
| Area, in dataset units | 178 | 226.30 |

The mean price is above the median, consistent with a right-skewed distribution. The training code uses a log-transformed price target. These figures describe the committed cleaned sample, not current Egyptian market prices or independently verified valuations.

## Business Recommendations

- Compare properties within the same location and type before interpreting a price difference.
- Use estimates as an exploratory reference, not a formal valuation.
- Validate on newer listings and a separate holdout before using predictions for pricing decisions.
- Review how trimming affects expensive or unusual properties before applying the model to those cases.

These are analytical recommendations, not evidence of achieved financial returns.

## Run Locally

```bash
git clone https://github.com/abdotamer55/Egypt-Houses-Price.git
cd Egypt-Houses-Price
pip install -r requirements.txt
streamlit run App.py
```

Keep the files in `models/` together. Only load serialized model files from sources you trust.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `data/` | Original and cleaned property CSV files |
| `notebooks/eda_improved.ipynb` | Exploratory analysis |
| `src/data_cleaning.py` | Data cleaning and filtering |
| `src/Feature_engineering.py` | Derived features |
| `src/Feature_selection.py` | Target encoding and feature selection |
| `src/Model.py` | Model comparison and artifact export |
| `src/Prediction.py` | Prediction pipeline |
| `models/` | Committed model and preprocessing artifacts |
| `dashboard/` | Power BI report |

The old misspelled module paths remain as small compatibility wrappers so earlier imports and file links still work. New code uses the corrected filenames.

## Limitations and Reproducibility

This documentation does not claim a new model score or a production deployment. Training modules contain top-level processing, so importing them may perform data preparation. The training script writes artifacts to its working directory; place regenerated artifacts in `models/` before running the app. A full training and application regression test is separate from this documentation and filename cleanup.
