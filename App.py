import os
import sys
import streamlit as st

# ── DYNAMIC PATH INJECTION FOR THE INFERENCE PIPELINE ──────────────────
# Identify the absolute path of the root directory (Egypt-Houses-Price)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inject 'src' folder into the python environment system paths safely
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# FIX: Import directly from Predection without the "src." prefix
from Predection import pipeline_predict

# ── GLOBAL PAGE CONFIGURATION ──────────────────────────────────────────
st.set_page_config(
    page_title="Egypt Houses Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── MAIN TITLE & APP TABS CREATION ──────────────────────────────────────
st.title("🏡 Egypt Property Price Predictor Platform")
st.write("An integrated platform bridging comprehensive real estate analytics with Machine Learning valuation.")

# Creating two core layout tabs to consolidate the application into a single page
tab1, tab2 = st.tabs(["📋 Project Overview", "🔮 Machine Learning Predictor"])

# ────────────────────────────────────────────────────────────────────────
# 📂 TAB 1: PROJECT OVERVIEW & DOCUMENTATION (FORMER HOME PAGE)
# ────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Project Documentation & Architecture")
    
    # Resolve and parse the README.md file
    README_PATH = os.path.join(BASE_DIR, "README.md")
    
    def load_readme(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return """
        # ⚠️ README.md file not found!
        Please make sure that the README.md file exists in the root directory next to App.py.
        """
        
    readme_content = load_readme(README_PATH)
    st.markdown(readme_content, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# 🔮 TAB 2: MACHINE LEARNING VALUATION FORM (FORMER PREDICTION PAGE)
# ────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Machine Learning Price Prediction Form")
    st.write("Select the precise real estate features below to calculate the estimated property market value:")
    
    # Graphical User Interface Form
    with st.form("property_detailed_form"):
        st.markdown("### 🏢 Location & Structural Attributes")
        
        city = st.selectbox(
            "Select City Location",
            ["New Cairo", "6th of October", "Cairo", "Giza", "Shorouk City", "Madinaty", "Sheikh Zayed"]
        )
        prop_type = st.selectbox(
            "Select Property Type",
            ["Apartment", "Stand Alone Villa", "Twin House", "Chalet", "Town House"]
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Built-up Area (Sqm)", min_value=10, max_value=2000, value=150)
        with col2:
            bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=15, value=3)
        with col3:
            bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=15, value=2)
            
        st.markdown("---")
        st.markdown("### 🛠️ Finishing, Compound & Contract Terms")
        
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

    # Pipeline Inference Execution on Click
    if submit_button:
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
                # Route payload directly into the pipeline function inside src/Predection.py
                predicted_price = pipeline_predict(raw_payload)
                
                st.success("🎉 Valuation computed successfully!")
                st.metric(
                    label="💰 Estimated Property Market Price Value",
                    value=f"{predicted_price:,.2f} EGP"
                )
            except Exception as e:
                st.error(
                    f"⚠️ An error occurred during inference pipeline execution.\n"
                    f"Please verify your saved model assets. Details: {str(e)}"
                )