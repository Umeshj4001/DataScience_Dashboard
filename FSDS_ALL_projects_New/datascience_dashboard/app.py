import streamlit as st
import importlib
import os
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #5C5C5C;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1rem;
        color: #696969;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<div class="main-header">Data Science Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="description">A comprehensive dashboard to showcase and run various data science projects.</div>', unsafe_allow_html=True)

# Define available categories
categories = {
    "Exploratory Data Analysis (EDA)": "eda",
    "Regression Projects": "regression",
    "Clustering Projects": "clustering"
}

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Project Categories")

# Initialize session state for category if it doesn't exist
if 'current_category' not in st.session_state:
    st.session_state.current_category = "eda"  # Default to EDA

# Create sidebar navigation
selected_category_name = st.sidebar.radio("Select a category:", list(categories.keys()))
selected_category = categories[selected_category_name]

# Update session state if category changed
if st.session_state.current_category != selected_category:
    st.session_state.current_category = selected_category

# Display author info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This interactive dashboard showcases various data science projects and techniques.")

# Load the selected module dynamically
try:
    # Import the module based on the selected category
    module_path = f"pages.{selected_category}"
    module = importlib.import_module(module_path)
    
    # Run the app function from the imported module
    module.app()
except ImportError:
    st.warning(f"The module for {selected_category_name} is not implemented yet.")
    st.info("Please select another category or check back later.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please try another category or report this issue.") 