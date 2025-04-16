import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
import os

# Add parent directory to path to import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_sample_dataset, file_uploader, display_dataset_info, plot_correlation_matrix, show_code

def app():
    """
    EDA application module.
    """
    st.markdown('<div class="sub-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown("""
    Exploratory Data Analysis (EDA) is a crucial step in any data science project. 
    It helps you understand the data, find patterns, spot anomalies, test hypotheses, 
    and check assumptions through summary statistics and graphical representations.
    """)
    
    # Data source selection
    st.subheader("Select Data Source")
    data_source = st.radio(
        "Choose a data source:",
        ["Sample Dataset", "Upload Your Own Data"],
        horizontal=True
    )
    
    # Data loading based on source selection
    if data_source == "Sample Dataset":
        sample_dataset = st.selectbox(
            "Choose a sample dataset:",
            ["iris", "diabetes", "wine"],
            help="Select one of the built-in datasets to analyze"
        )
        df = get_sample_dataset(sample_dataset)
        st.success(f"Loaded {sample_dataset} dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    else:
        df = file_uploader("eda_upload", "Upload your CSV file for exploratory data analysis")
        if df is None:
            st.info("Please upload a CSV file to begin analysis or select a sample dataset.")
            st.stop()
        else:
            st.success(f"Loaded your dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Display dataset info
    display_dataset_info(df)
    
    # EDA actions in tabs
    eda_tabs = st.tabs(["Distribution Analysis", "Correlation Analysis", "Outlier Detection", "Missing Values"])
    
    with eda_tabs[0]:  # Distribution Analysis
        st.write("### Distribution Analysis")
        
        # Numeric column selection for distribution analysis
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_columns:
            st.warning("No numeric columns found in the dataset for distribution analysis.")
        else:
            selected_column = st.selectbox("Select a column to analyze:", numeric_columns, key="dist_select")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                st.write("#### Histogram")
                bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30, key="hist_bins")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[selected_column], bins=bins, kde=True, ax=ax)
                ax.set_title(f'Distribution of {selected_column}')
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
            with col2:
                # Box plot
                st.write("#### Box Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=df[selected_column], ax=ax)
                ax.set_title(f'Box Plot of {selected_column}')
                st.pyplot(fig)
            
            # Statistics
            st.write("#### Summary Statistics")
            stats_df = pd.DataFrame({
                "Mean": [df[selected_column].mean()],
                "Median": [df[selected_column].median()],
                "Std Dev": [df[selected_column].std()],
                "Min": [df[selected_column].min()],
                "Max": [df[selected_column].max()],
                "Q1 (25%)": [df[selected_column].quantile(0.25)],
                "Q3 (75%)": [df[selected_column].quantile(0.75)],
                "IQR": [df[selected_column].quantile(0.75) - df[selected_column].quantile(0.25)],
                "Skewness": [df[selected_column].skew()],
                "Kurtosis": [df[selected_column].kurt()]
            })
            st.dataframe(stats_df.T)
    
    with eda_tabs[1]:  # Correlation Analysis
        st.write("### Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            st.warning("At least two numeric columns are required for correlation analysis.")
        else:
            plot_correlation_matrix(df)
            
            # Scatter plot between two selected variables
            st.write("#### Scatter Plot")
            
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis:", numeric_df.columns, key="scatter_x")
            with col2:
                remaining_cols = [col for col in numeric_df.columns if col != x_col]
                y_col = st.selectbox("Select Y-axis:", remaining_cols, key="scatter_y")
            
            # Color by categorical variable if available
            color_col = None
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'target_name' in df.columns:
                categorical_cols.append('target_name')
            
            if categorical_cols:
                use_color = st.checkbox("Color by category", value=True if 'target_name' in df.columns else False)
                if use_color:
                    color_col = st.selectbox("Select color variable:", categorical_cols)
            
            # Create scatter plot
            if color_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                                title=f'Scatter Plot: {x_col} vs {y_col} (colored by {color_col})',
                                opacity=0.7)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, 
                                title=f'Scatter Plot: {x_col} vs {y_col}',
                                opacity=0.7)
            
            fig.update_layout(height=500, width=700)
            st.plotly_chart(fig, use_container_width=True)
    
    with eda_tabs[2]:  # Outlier Detection
        st.write("### Outlier Detection")
        
        if not numeric_columns:
            st.warning("No numeric columns found in the dataset for outlier detection.")
        else:
            selected_column = st.selectbox("Select a column to detect outliers:", numeric_columns, key="outlier_select")
            
            # Calculate outlier bounds using IQR
            Q1 = df[selected_column].quantile(0.25)
            Q3 = df[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
            outlier_count = len(outliers)
            
            # Display outlier information
            st.write(f"**Found {outlier_count} outliers** using the IQR method (values outside the range of {lower_bound:.2f} to {upper_bound:.2f}).")
            
            # Plot with outliers highlighted
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(range(len(df)), df[selected_column], alpha=0.5, label='Normal')
            if outlier_count > 0:
                ax.scatter(outliers.index, outliers[selected_column], color='red', alpha=0.8, label='Outliers')
            ax.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.3, label='Upper Bound')
            ax.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.3, label='Lower Bound')
            ax.set_title(f'Outlier Detection for {selected_column}')
            ax.set_xlabel('Index')
            ax.set_ylabel(selected_column)
            ax.legend()
            st.pyplot(fig)
            
            # Show outliers in a table
            if outlier_count > 0:
                with st.expander("View Outliers"):
                    st.dataframe(outliers[[selected_column]])
    
    with eda_tabs[3]:  # Missing Values
        st.write("### Missing Value Analysis")
        
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if missing_df.empty:
            st.success("No missing values found in the dataset!")
        else:
            st.write(f"Found {missing_df['Missing Values'].sum()} missing values across {len(missing_df)} columns.")
            
            # Display missing values table
            st.dataframe(missing_df)
            
            # Plot missing values
            if not missing_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_df = missing_df.sort_values('Missing Values', ascending=False)
                sns.barplot(x=missing_df.index, y='Percentage (%)', data=missing_df, ax=ax)
                ax.set_title('Missing Values by Column')
                ax.set_xlabel('Columns')
                ax.set_ylabel('Missing Percentage (%)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
    
    # Code explanation
    show_code("""
# Example EDA code for your own analysis:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('your_data.csv')

# Basic data information
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Distribution of a numeric column
plt.figure(figsize=(10, 6))
sns.histplot(df['your_column'], kde=True)
plt.title('Distribution of your_column')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=['number']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
    """) 