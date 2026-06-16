import os
import sys
import streamlit as st
from src.Predection import pipeline_predict

# ── DYNAMIC PATH INJECTION FOR MULTI-PAGE STRUCTURE ────────────────────
# CURRENT_DIR resolves to the absolute path of the 'pages' directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR resolves to the root project folder (Egypt-Houses-Price)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Append the neighboring 'src' directory to python path environment safely
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ── PAGE CONFIGURATION ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Price Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Machine Learning Price Prediction Form")
st.write("Select the precise real estate features below to calculate the estimated property market value:")

# ── STREAMLIT GRAPHICAL USER INTERFACE FORM ─────────────────────────────
with st.form("property_detailed_form"):
    st.markdown("### 🏢 Location & Structural Attributes")
    
    # Dropdown menus for geographical and categorical property types
    city = st.selectbox(
        "Select City Location",
        ["New Cairo", "6th of October", "Cairo", "Giza", "Shorouk City", "Madinaty", "Sheikh Zayed"]
    )
    prop_type = st.selectbox(
        "Select Property Type",
        ["Apartment", "Stand Alone Villa", "Twin House", "Chalet", "Town House"]
    )
    
    # Numerical fields grouped into 3 symmetric columns
    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("Built-up Area (Sqm)", min_value=10, max_value=2000, value=150)
    with col2:
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=15, value=3)
    with col3:
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=15, value=2)
        
    st.markdown("---")
    st.markdown("### 🛠️ Finishing, Compound & Contract Terms")
    
    # Converted text inputs into clean dropdown selectboxes to enforce data consistency
    col4, col5, col6 = st.columns(3)
    with col4:
        furnished = st.selectbox("Is it Furnished?", ["No", "Yes", "Unknown"])
        level = st.selectbox(
            "Floor Level", 
            ["Ground", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
        )
    with col5:
        delivery_term = st.selectbox("Finishing Status", ["Finished", "Semi Finished", "Core & Shell", "Unknown"])
        delivery_date = st.selectbox(
            "Delivery Date / Status", 
            ["Ready to move", "Soon", "2026", "2027", "2028", "Unknown"]
        )
    with col6:
        payment_option = st.selectbox("Payment Terms", ["Cash", "Installment", "Unknown"])
        compound = st.selectbox(
            "Compound Name", 
            ["Unknown", "Mountain View", "Palm Hills", "SODIC", "Madinaty", "Mivida", "Zayed Regency", "Villette"]
        )

    # Form submission trigger button
    submit_button = st.form_submit_button("Calculate Estimated Market Price")

# ── PIPELINE INFERENCE EXECUTION ON CLICK ───────────────────────────────
if submit_button:
    # Package all graphical user inputs into the identical dictionary format expected by the pipeline
    raw_payload = {
        'Type': prop_type,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Area': area,
        'Furnished': furnished,
        'Level': level,
        'Compound': compound,
        'Payment_Option': payment_option,
        'Delivery_Date': delivery_date,
        'Delivery_Term': delivery_term,
        'City': city
    }
    
    with st.spinner("Processing input data and executing model inference..."):
        try:
            # Route payload directly into the saved feature engineering and model weights pipeline
            predicted_price = pipeline_predict(raw_payload)
            
            # Render the final financial calculation in EGP with proper decimal string formatting
            st.success("🎉 Valuation computed successfully!")
            st.metric(
                label="💰 Estimated Property Market Price Value",
                value=f"{predicted_price:,.2f} EGP"
            )
        except Exception as e:
            # Fallback error messaging in case of missing model assets or corrupt values
            st.error(
                f"⚠️ An error occurred during inference pipeline execution.\n"
                f"Please verify your saved model assets. Details: {str(e)}"
            )