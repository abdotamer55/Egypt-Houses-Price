import os
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Egypt Houses Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏡 Egypt Property Price Predictor - Home Page")

# Resolve path to README.md (located in the same root directory as App.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
README_PATH = os.path.join(BASE_DIR, "README.md")

def load_readme(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        # ⚠️ README.md file not found!
        Please make sure that the README.md file exists in the root directory next to App.py.
        """

# Read and render the content of the markdown file
readme_content = load_readme(README_PATH)
st.markdown(readme_content, unsafe_allow_html=True)

# Sidebar navigation tip - Streamlit automatically builds the sidebar using the 'pages' folder
st.sidebar.success("Main")