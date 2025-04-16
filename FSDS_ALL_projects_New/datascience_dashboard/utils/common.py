import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64
from sklearn.datasets import load_iris, load_diabetes, load_wine, make_blobs, make_regression
import os

# Set the style for plots
sns.set_style("whitegrid")

def get_sample_dataset(dataset_name):
    """
    Load a sample dataset based on the name.
    
    Args:
        dataset_name (str): Name of the dataset to load
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    if dataset_name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = [data.target_names[i] for i in data.target]
        return df
    
    elif dataset_name == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "synthetic_regression":
        X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
        df['target'] = y
        return df
    
    elif dataset_name == "synthetic_clusters":
        X, y = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['cluster'] = y
        return df
    
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

def file_uploader(key, help_text="Upload your CSV data file"):
    """
    Create a file uploader that returns a DataFrame.
    
    Args:
        key (str): Unique key for the uploader widget
        help_text (str): Help text to display
        
    Returns:
        pandas.DataFrame or None: The uploaded data or None if no file is uploaded
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key=key, help=help_text)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

def display_dataset_info(df):
    """
    Display basic information about a dataset.
    
    Args:
        df (pandas.DataFrame): The dataset to display information for
    """
    st.write("### Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
    
    with col2:
        st.write(f"**Numeric columns:** {len(df.select_dtypes(include=['number']).columns)}")
        st.write(f"**Categorical columns:** {len(df.select_dtypes(exclude=['number']).columns)}")
    
    with st.expander("Preview Data"):
        st.dataframe(df.head())
    
    with st.expander("Data Types"):
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
        dtypes_df.index.name = "Column"
        dtypes_df.reset_index(inplace=True)
        st.dataframe(dtypes_df)
    
    with st.expander("Summary Statistics"):
        st.dataframe(df.describe())

def create_download_link(df, filename="data.csv"):
    """
    Create a download link for a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame to download
        filename (str): The name of the file to download
        
    Returns:
        str: HTML code for the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def plot_correlation_matrix(df, numeric_only=True):
    """
    Plot a correlation matrix for a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame to plot
        numeric_only (bool): Whether to include only numeric columns
    """
    if numeric_only:
        df_numeric = df.select_dtypes(include=['number'])
    else:
        df_numeric = df
    
    if df_numeric.shape[1] < 2:
        st.warning("Not enough numeric columns to create a correlation matrix.")
        return
    
    # Calculate correlation
    corr = df_numeric.corr()
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax)
    
    # Set title
    plt.title('Correlation Matrix', fontsize=15)
    
    # Show the plot in Streamlit
    st.pyplot(fig)
    
def show_code(code):
    """
    Display the code in an expandable section.
    
    Args:
        code (str): The code to display
    """
    with st.expander("Show Code"):
        st.code(code, language="python") 