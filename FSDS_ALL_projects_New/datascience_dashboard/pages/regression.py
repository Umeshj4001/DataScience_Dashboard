import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
import joblib
from io import BytesIO

# Add parent directory to path to import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_sample_dataset, file_uploader, display_dataset_info, plot_correlation_matrix, show_code

def app():
    """
    Regression application module.
    """
    st.markdown('<div class="sub-header">Regression Projects</div>', unsafe_allow_html=True)
    st.markdown("""
    Regression analysis is a supervised learning technique used to predict continuous values. 
    This module allows you to build, train, and evaluate various regression models on your data.
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
            ["diabetes", "synthetic_regression"],
            help="Select one of the built-in datasets for regression analysis"
        )
        df = get_sample_dataset(sample_dataset)
        st.success(f"Loaded {sample_dataset} dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    else:
        df = file_uploader("regression_upload", "Upload your CSV file for regression analysis")
        if df is None:
            st.info("Please upload a CSV file to begin analysis or select a sample dataset.")
            st.stop()
        else:
            st.success(f"Loaded your dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Display dataset info
    display_dataset_info(df)
    
    # Define tabs for different sections
    tabs = st.tabs(["Feature Selection", "Model Training", "Evaluation & Visualization"])
    
    # Initialize session state for model results
    if 'regression_results' not in st.session_state:
        st.session_state.regression_results = {}
    
    if 'regression_feature_importance' not in st.session_state:
        st.session_state.regression_feature_importance = {}
    
    if 'regression_best_model' not in st.session_state:
        st.session_state.regression_best_model = None
    
    if 'regression_X_train' not in st.session_state:
        st.session_state.regression_X_train = None
        
    if 'regression_y_train' not in st.session_state:
        st.session_state.regression_y_train = None
        
    if 'regression_X_test' not in st.session_state:
        st.session_state.regression_X_test = None
        
    if 'regression_y_test' not in st.session_state:
        st.session_state.regression_y_test = None
    
    if 'regression_target' not in st.session_state:
        st.session_state.regression_target = None
    
    if 'regression_features' not in st.session_state:
        st.session_state.regression_features = None
    
    # Feature Selection Tab
    with tabs[0]:
        st.subheader("Feature Selection")
        
        # Target variable selection
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_columns:
            st.error("No numeric columns found in the dataset. Regression requires numeric features.")
            st.stop()
        
        st.write("### Select Target Variable")
        target_col = st.selectbox("Choose the target variable to predict:", numeric_columns, key="reg_target")
        
        # Feature selection
        st.write("### Select Features")
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Select features to include in the model:", 
            available_features,
            default=available_features[:min(5, len(available_features))],  # Default selects first 5 features
            key="reg_features"
        )
        
        if not selected_features:
            st.warning("Please select at least one feature for the model.")
            st.stop()
        
        # Save selections to session state
        st.session_state.regression_target = target_col
        st.session_state.regression_features = selected_features
        
        # Show correlation with target
        st.write("### Feature Correlation with Target")
        feature_corrs = df[selected_features + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_corrs.values, y=feature_corrs.index, ax=ax)
        ax.set_title(f'Feature Correlation with {target_col}')
        ax.set_xlabel('Correlation')
        st.pyplot(fig)
        
        # Data splitting
        st.write("### Data Splitting")
        test_size = st.slider("Test set size (%):", min_value=10, max_value=40, value=20, key="reg_test_size") / 100
        random_state = st.number_input("Random state for reproducibility:", min_value=0, max_value=100, value=42, key="reg_random_state")
        
        # Split the data
        X = df[selected_features]
        y = df[target_col]
        
        # Check for categorical features
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = None
        if cat_features:
            st.info(f"Detected {len(cat_features)} categorical features: {', '.join(cat_features)}. These will be one-hot encoded.")
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, num_features),
                    ('cat', categorical_transformer, cat_features)
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features)
                ]
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
        
        # Save to session state (original data)
        st.session_state.regression_X_train = X_train
        st.session_state.regression_y_train = y_train
        st.session_state.regression_X_test = X_test
        st.session_state.regression_y_test = y_test
        st.session_state.regression_preprocessor = preprocessor
        
        # Display split information
        st.write(f"Training set: {X_train.shape[0]} samples")
        st.write(f"Test set: {X_test.shape[0]} samples")
        
        if st.button("Proceed to Model Training", key="reg_to_training"):
            st.success("Features selected and data split successfully! Click on the 'Model Training' tab to continue.")
    
    # Model Training Tab
    with tabs[1]:
        st.subheader("Model Training")
        
        # Check if features and target are selected
        if not st.session_state.regression_features or not st.session_state.regression_target:
            st.warning("Please complete the feature selection step first.")
            st.stop()
        
        # Model selection
        st.write("### Choose Regression Models")
        
        # Model checkboxes
        col1, col2 = st.columns(2)
        with col1:
            use_linear = st.checkbox("Linear Regression", value=True)
            use_ridge = st.checkbox("Ridge Regression", value=False)
            use_lasso = st.checkbox("Lasso Regression", value=False)
            use_elastic = st.checkbox("ElasticNet", value=False)
        
        with col2:
            use_rf = st.checkbox("Random Forest", value=True)
            use_gbm = st.checkbox("Gradient Boosting", value=False)
            use_svr = st.checkbox("Support Vector Regression", value=False)
        
        # Check if at least one model is selected
        if not any([use_linear, use_ridge, use_lasso, use_elastic, use_rf, use_gbm, use_svr]):
            st.warning("Please select at least one regression model.")
            st.stop()
        
        # Hyperparameters section
        with st.expander("Hyperparameters"):
            # Different hyperparameters based on selected models
            params = {}
            
            if use_ridge:
                params['ridge_alpha'] = st.slider("Ridge alpha:", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="ridge_alpha")
            
            if use_lasso:
                params['lasso_alpha'] = st.slider("Lasso alpha:", min_value=0.001, max_value=1.0, value=0.1, step=0.001, key="lasso_alpha")
            
            if use_elastic:
                params['elastic_alpha'] = st.slider("ElasticNet alpha:", min_value=0.001, max_value=1.0, value=0.1, step=0.001, key="elastic_alpha")
                params['elastic_l1'] = st.slider("ElasticNet L1 ratio:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="elastic_l1")
            
            if use_rf:
                params['rf_n_estimators'] = st.slider("Random Forest n_estimators:", min_value=10, max_value=500, value=100, step=10, key="rf_n_estimators")
                params['rf_max_depth'] = st.slider("Random Forest max_depth:", min_value=1, max_value=30, value=10, key="rf_max_depth")
            
            if use_gbm:
                params['gbm_n_estimators'] = st.slider("Gradient Boosting n_estimators:", min_value=10, max_value=500, value=100, step=10, key="gbm_n_estimators")
                params['gbm_learning_rate'] = st.slider("Gradient Boosting learning_rate:", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="gbm_learning_rate")
            
            if use_svr:
                params['svr_C'] = st.slider("SVR C:", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="svr_C")
        
        st.write("### Cross-Validation Settings")
        cv_folds = st.slider("Number of cross-validation folds:", min_value=2, max_value=10, value=5, key="reg_cv_folds")
        
        # Train models button
        if st.button("Train Models", key="train_reg_models"):
            # Dictionary to store models
            models = {}
            
            # Create the models based on selections
            if use_linear:
                models['Linear Regression'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', LinearRegression())
                ])
            
            if use_ridge:
                models['Ridge Regression'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', Ridge(alpha=params.get('ridge_alpha', 1.0)))
                ])
            
            if use_lasso:
                models['Lasso Regression'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', Lasso(alpha=params.get('lasso_alpha', 0.1)))
                ])
            
            if use_elastic:
                models['ElasticNet'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', ElasticNet(
                        alpha=params.get('elastic_alpha', 0.1),
                        l1_ratio=params.get('elastic_l1', 0.5)
                    ))
                ])
            
            if use_rf:
                models['Random Forest'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', RandomForestRegressor(
                        n_estimators=params.get('rf_n_estimators', 100),
                        max_depth=params.get('rf_max_depth', 10),
                        random_state=42
                    ))
                ])
            
            if use_gbm:
                models['Gradient Boosting'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=params.get('gbm_n_estimators', 100),
                        learning_rate=params.get('gbm_learning_rate', 0.1),
                        random_state=42
                    ))
                ])
            
            if use_svr:
                models['SVR'] = Pipeline([
                    ('preprocessor', st.session_state.regression_preprocessor),
                    ('regressor', SVR(C=params.get('svr_C', 1.0)))
                ])
            
            # Progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Results dictionary
            results = {}
            feature_importance = {}
            
            # Train and evaluate each model
            for i, (name, model) in enumerate(models.items()):
                progress_text.text(f"Training {name}...")
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(
                        model, st.session_state.regression_X_train, st.session_state.regression_y_train, 
                        cv=cv_folds, scoring='neg_mean_squared_error'
                    )
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    
                    # Fit on full training set
                    model.fit(st.session_state.regression_X_train, st.session_state.regression_y_train)
                    
                    # Predict on test set
                    y_pred = model.predict(st.session_state.regression_X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(st.session_state.regression_y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(st.session_state.regression_y_test, y_pred)
                    r2 = r2_score(st.session_state.regression_y_test, y_pred)
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'cv_rmse': cv_rmse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'predictions': y_pred
                    }
                    
                    # Get feature importance if available
                    if name in ['Random Forest', 'Gradient Boosting']:
                        # For tree-based models
                        importances = model.named_steps['regressor'].feature_importances_
                        # Get feature names (accounting for one-hot encoding)
                        feature_names = (
                            st.session_state.regression_preprocessor.get_feature_names_out()
                            if hasattr(st.session_state.regression_preprocessor, 'get_feature_names_out')
                            else st.session_state.regression_features
                        )
                        feature_importance[name] = pd.Series(
                            importances, index=feature_names
                        ).sort_values(ascending=False)
                    
                    if name == 'Linear Regression':
                        # For linear models
                        if not st.session_state.regression_preprocessor.transformers_[0][0] == 'num':
                            st.warning("Feature importance for Linear Regression is only available with numeric features only.")
                        else:
                            coefs = model.named_steps['regressor'].coef_
                            feature_importance[name] = pd.Series(
                                np.abs(coefs), index=st.session_state.regression_features
                            ).sort_values(ascending=False)
                    
                except Exception as e:
                    st.error(f"Error training {name}: {e}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(models))
            
            progress_text.text("Model training complete!")
            
            # Find best model based on R2 score
            if results:
                best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
                st.session_state.regression_best_model = best_model_name
                st.session_state.regression_results = results
                st.session_state.regression_feature_importance = feature_importance
                
                st.success(f"Models trained successfully! Best model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
                st.write("Go to the 'Evaluation & Visualization' tab to see detailed results.")
    
    # Evaluation & Visualization Tab
    with tabs[2]:
        st.subheader("Evaluation & Visualization")
        
        if not st.session_state.regression_results:
            st.warning("Please train models in the 'Model Training' tab first.")
            st.stop()
        
        # Show evaluation metrics in a table
        st.write("### Model Performance Comparison")
        metrics_df = pd.DataFrame({
            model_name: {
                'R² Score': results['r2'],
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'CV RMSE': results['cv_rmse']
            }
            for model_name, results in st.session_state.regression_results.items()
        }).T
        
        # Show the table and highlight the best model
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R² Score']).highlight_min(axis=0, subset=['RMSE', 'MAE', 'CV RMSE']))
        
        # Visualize predictions vs actual
        st.write("### Predictions vs Actual Values")
        
        # Model selector
        available_models = list(st.session_state.regression_results.keys())
        selected_model = st.selectbox(
            "Select model to visualize:", 
            available_models,
            index=available_models.index(st.session_state.regression_best_model) if st.session_state.regression_best_model in available_models else 0
        )
        
        model_results = st.session_state.regression_results[selected_model]
        
        # Create scatter plot of predictions vs actual
        fig = px.scatter(
            x=st.session_state.regression_y_test,
            y=model_results['predictions'],
            labels={"x": "Actual Values", "y": "Predicted Values"},
            title=f"{selected_model}: Actual vs Predicted Values"
        )
        
        # Add perfect prediction line
        min_val = min(st.session_state.regression_y_test.min(), model_results['predictions'].min())
        max_val = max(st.session_state.regression_y_test.max(), model_results['predictions'].max())
        fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual plot
        st.write("### Residual Analysis")
        
        residuals = st.session_state.regression_y_test - model_results['predictions']
        
        fig = px.scatter(
            x=model_results['predictions'],
            y=residuals,
            labels={"x": "Predicted Values", "y": "Residuals"},
            title=f"{selected_model}: Residual Plot"
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance if available
        st.write("### Feature Importance")
        
        if selected_model in st.session_state.regression_feature_importance:
            importance = st.session_state.regression_feature_importance[selected_model]
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, len(importance) * 0.4 + 2))
            importance.sort_values().plot(kind='barh', ax=ax)
            ax.set_title(f"Feature Importance - {selected_model}")
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        else:
            st.info(f"Feature importance visualization not available for {selected_model}.")
        
        # Model download section
        st.write("### Download Trained Model")
        
        if st.button("Download Selected Model", key="download_reg_model"):
            # Save model to BytesIO
            model_file = BytesIO()
            joblib.dump(model_results['model'], model_file)
            model_file.seek(0)
            
            # Create download button
            st.download_button(
                label="Click to Download Model",
                data=model_file,
                file_name=f"{selected_model.replace(' ', '_').lower()}_model.joblib",
                mime="application/octet-stream"
            )
            
            st.info("""
            The downloaded model is a scikit-learn pipeline that includes preprocessing.
            To use it for predictions:
            ```python
            import joblib
            model = joblib.load('downloaded_model.joblib')
            predictions = model.predict(new_data)
            ```
            """)
    
    # Code explanation
    show_code("""
# Example regression code for your own analysis:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('your_data.csv')

# Prepare features and target
X = df[['feature1', 'feature2', 'feature3']]  # Replace with your features
y = df['target_variable']  # Replace with your target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
print(coef_df)

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Prediction vs Actual')
plt.show()
    """) 