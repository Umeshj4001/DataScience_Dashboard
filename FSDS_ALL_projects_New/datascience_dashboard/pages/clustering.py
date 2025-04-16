import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sys
import os
from io import BytesIO

# Add parent directory to path to import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_sample_dataset, file_uploader, display_dataset_info, show_code

def app():
    """
    Clustering application module.
    """
    st.markdown('<div class="sub-header">Clustering Projects</div>', unsafe_allow_html=True)
    st.markdown("""
    Clustering is an unsupervised learning technique that groups similar data points together based on 
    their features. This module allows you to apply various clustering algorithms to your data and 
    visualize the results.
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
            ["iris", "wine", "synthetic_clusters"],
            help="Select one of the built-in datasets for clustering analysis"
        )
        df = get_sample_dataset(sample_dataset)
        st.success(f"Loaded {sample_dataset} dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    else:
        df = file_uploader("clustering_upload", "Upload your CSV file for clustering analysis")
        if df is None:
            st.info("Please upload a CSV file to begin analysis or select a sample dataset.")
            st.stop()
        else:
            st.success(f"Loaded your dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Display dataset info
    display_dataset_info(df)
    
    # Define tabs for different sections
    tabs = st.tabs(["Feature Selection", "Clustering", "Evaluation & Visualization"])
    
    # Initialize session state for clustering results
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    
    if 'clustering_data' not in st.session_state:
        st.session_state.clustering_data = None
    
    if 'clustering_features' not in st.session_state:
        st.session_state.clustering_features = None
    
    if 'clustering_pca_data' not in st.session_state:
        st.session_state.clustering_pca_data = None
        
    if 'clustering_scaler' not in st.session_state:
        st.session_state.clustering_scaler = None
        
    if 'clustering_pca' not in st.session_state:
        st.session_state.clustering_pca = None
    
    # Feature Selection Tab
    with tabs[0]:
        st.subheader("Feature Selection")
        
        # Select features for clustering
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove target column from numeric columns if it exists (for sample datasets)
        if 'target' in numeric_columns:
            numeric_columns.remove('target')
        if 'cluster' in numeric_columns:
            numeric_columns.remove('cluster')
        
        if not numeric_columns:
            st.error("No numeric columns found in the dataset. Clustering requires numeric features.")
            st.stop()
        
        st.write("### Select Features")
        selected_features = st.multiselect(
            "Select features to include in clustering:", 
            numeric_columns,
            default=numeric_columns[:min(5, len(numeric_columns))],  # Default selects first 5 features
            key="cluster_features"
        )
        
        if not selected_features:
            st.warning("Please select at least two features for clustering.")
            st.stop()
        elif len(selected_features) < 2:
            st.warning("Please select at least two features for effective clustering.")
            st.stop()
        
        # Data preprocessing
        st.write("### Data Preprocessing")
        
        # Scaling method
        scaling_method = st.radio(
            "Select scaling method:",
            ["StandardScaler", "MinMaxScaler", "None"],
            horizontal=True,
            key="scaling_method"
        )
        
        # Apply PCA for dimensionality reduction if more than 2 features
        apply_pca = False
        pca_components = 2
        
        if len(selected_features) > 2:
            apply_pca = st.checkbox("Apply PCA for dimensionality reduction", value=True, key="apply_pca")
            
            if apply_pca:
                max_components = min(len(selected_features), 10)  # Limit to 10 components max
                pca_components = st.slider(
                    "Number of PCA components:", 
                    min_value=2, 
                    max_value=max_components, 
                    value=2, 
                    key="pca_components"
                )
        
        # Process the data
        if st.button("Process Data", key="process_data"):
            # Extract selected features
            X = df[selected_features].copy()
            
            # Handle missing values
            if X.isnull().sum().sum() > 0:
                st.warning(f"Found {X.isnull().sum().sum()} missing values. These will be replaced with the mean.")
                X = X.fillna(X.mean())
            
            # Scale the data
            scaler = None
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            else:
                X_scaled = X
            
            # Apply PCA if selected
            pca_model = None
            X_pca = None
            
            if apply_pca:
                pca_model = PCA(n_components=pca_components)
                pca_result = pca_model.fit_transform(X_scaled)
                pca_cols = [f"PC{i+1}" for i in range(pca_components)]
                X_pca = pd.DataFrame(pca_result, columns=pca_cols)
                
                # Show explained variance
                explained_variance = pca_model.explained_variance_ratio_
                cum_explained_variance = np.cumsum(explained_variance)
                
                st.write("### PCA Explained Variance")
                
                # Create a bar chart of explained variance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(pca_cols, explained_variance * 100)
                ax.set_ylabel("Explained Variance (%)")
                ax.set_xlabel("Principal Components")
                ax.set_title("Explained Variance by Principal Component")
                
                # Add a line for cumulative explained variance
                ax2 = ax.twinx()
                ax2.plot(pca_cols, cum_explained_variance * 100, 'r-', marker='o')
                ax2.set_ylabel("Cumulative Explained Variance (%)")
                ax2.grid(False)
                
                st.pyplot(fig)
                
                st.write(f"**Total explained variance:** {cum_explained_variance[-1] * 100:.2f}%")
            
            # Save processed data to session state
            st.session_state.clustering_data = X_scaled
            st.session_state.clustering_features = selected_features
            st.session_state.clustering_pca_data = X_pca
            st.session_state.clustering_scaler = scaler
            st.session_state.clustering_pca = pca_model
            
            st.success("Data processed successfully! Proceed to the Clustering tab.")
    
    # Clustering Tab
    with tabs[1]:
        st.subheader("Clustering")
        
        # Check if data has been processed
        if st.session_state.clustering_data is None:
            st.warning("Please process your data in the Feature Selection tab first.")
            st.stop()
        
        # Choose clustering algorithm
        st.write("### Select Clustering Algorithm")
        clustering_algo = st.selectbox(
            "Choose a clustering algorithm:",
            ["K-Means", "DBSCAN", "Agglomerative Clustering"],
            key="clustering_algo"
        )
        
        # Algorithm-specific parameters
        if clustering_algo == "K-Means":
            n_clusters = st.slider(
                "Number of clusters (k):", 
                min_value=2, 
                max_value=10, 
                value=3, 
                key="kmeans_clusters"
            )
            
            # Elbow method for K-Means
            with st.expander("Elbow Method for K Selection"):
                if st.button("Run Elbow Method", key="elbow_method"):
                    data = st.session_state.clustering_pca_data if st.session_state.clustering_pca_data is not None else st.session_state.clustering_data
                    
                    # Calculate inertia for different k values
                    inertia = []
                    silhouette = []
                    k_range = range(2, 11)
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(data)
                        inertia.append(kmeans.inertia_)
                        
                        # Calculate silhouette score if k > 1
                        if k > 1:
                            labels = kmeans.labels_
                            sil_score = silhouette_score(data, labels)
                            silhouette.append(sil_score)
                        else:
                            silhouette.append(0)
                    
                    # Plot elbow method
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Inertia plot (left y-axis)
                    ax1.plot(list(k_range), inertia, 'bo-', label='Inertia')
                    ax1.set_xlabel('Number of clusters (k)')
                    ax1.set_ylabel('Inertia (within-cluster sum of squares)', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')
                    
                    # Silhouette plot (right y-axis)
                    ax2 = ax1.twinx()
                    ax2.plot(list(k_range), silhouette, 'ro-', label='Silhouette Score')
                    ax2.set_ylabel('Silhouette Score', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    plt.title('Elbow Method for K-Means Clustering')
                    fig.tight_layout()
                    
                    # Add both legends
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                    st.pyplot(fig)
                    
                    st.info("""
                    **How to choose k using the Elbow Method:**
                    1. Look for the "elbow" point where adding more clusters doesn't significantly reduce inertia
                    2. Higher silhouette scores indicate better clustering
                    3. Balance between these metrics to find the optimal k
                    """)
        
        elif clustering_algo == "DBSCAN":
            col1, col2 = st.columns(2)
            
            with col1:
                eps = st.slider(
                    "eps (maximum distance between points):", 
                    min_value=0.05, 
                    max_value=2.0, 
                    value=0.5, 
                    step=0.05, 
                    key="dbscan_eps"
                )
            
            with col2:
                min_samples = st.slider(
                    "min_samples (minimum points in neighborhood):", 
                    min_value=2, 
                    max_value=15, 
                    value=5, 
                    key="dbscan_min_samples"
                )
        
        elif clustering_algo == "Agglomerative Clustering":
            n_clusters = st.slider(
                "Number of clusters:", 
                min_value=2, 
                max_value=10, 
                value=3, 
                key="agg_clusters"
            )
            
            linkage = st.selectbox(
                "Linkage criteria:",
                ["ward", "complete", "average", "single"],
                key="agg_linkage"
            )
            
            affinity = st.selectbox(
                "Affinity metric:",
                ["euclidean", "manhattan", "cosine"],
                key="agg_affinity"
            )
        
        # Run clustering
        if st.button("Run Clustering", key="run_clustering"):
            # Get the data to cluster
            data = st.session_state.clustering_pca_data if st.session_state.clustering_pca_data is not None else st.session_state.clustering_data
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize model based on algorithm choice
            status_text.text("Initializing clustering model...")
            progress_bar.progress(25)
            
            if clustering_algo == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            elif clustering_algo == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
            
            elif clustering_algo == "Agglomerative Clustering":
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    affinity=affinity if linkage != 'ward' else 'euclidean'  # ward only works with euclidean
                )
            
            # Fit the model
            status_text.text("Fitting clustering model...")
            progress_bar.progress(50)
            
            cluster_labels = model.fit_predict(data)
            
            status_text.text("Processing results...")
            progress_bar.progress(75)
            
            # Add cluster labels to the data
            if st.session_state.clustering_pca_data is not None:
                result_df = st.session_state.clustering_pca_data.copy()
            else:
                result_df = st.session_state.clustering_data.copy()
                
            result_df['cluster'] = cluster_labels
            
            # Calculate clustering metrics if applicable
            metrics = {}
            
            if clustering_algo != "DBSCAN" or len(np.unique(cluster_labels)) > 1:
                try:
                    metrics['silhouette'] = silhouette_score(data, cluster_labels)
                    metrics['calinski_harabasz'] = calinski_harabasz_score(data, cluster_labels)
                    metrics['davies_bouldin'] = davies_bouldin_score(data, cluster_labels)
                except:
                    st.warning("Could not calculate some clustering metrics.")
            
            # Store results in session state
            st.session_state.clustering_results = {
                'labels': cluster_labels,
                'data': result_df,
                'algorithm': clustering_algo,
                'metrics': metrics,
                'model': model
            }
            
            progress_bar.progress(100)
            status_text.text("Clustering complete!")
            
            # Display basic cluster statistics
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            
            st.write("### Cluster Distribution")
            
            # Convert series to DataFrame for better display
            cluster_df = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Count': cluster_counts.values,
                'Percentage': (cluster_counts.values / len(cluster_labels) * 100).round(2)
            })
            
            # Handle special case for DBSCAN where -1 is noise
            if clustering_algo == "DBSCAN" and -1 in cluster_df['Cluster'].values:
                cluster_df.loc[cluster_df['Cluster'] == -1, 'Cluster'] = 'Noise'
            
            st.dataframe(cluster_df)
            
            # Show clustering metrics if available
            if metrics:
                st.write("### Clustering Metrics")
                st.write(f"**Silhouette Score:** {metrics.get('silhouette', 'N/A'):.4f} (higher is better, range: -1 to 1)")
                st.write(f"**Calinski-Harabasz Index:** {metrics.get('calinski_harabasz', 'N/A'):.4f} (higher is better)")
                st.write(f"**Davies-Bouldin Index:** {metrics.get('davies_bouldin', 'N/A'):.4f} (lower is better)")
            
            st.success("Clustering complete! Go to the Evaluation & Visualization tab to explore the results.")
    
    # Evaluation & Visualization Tab
    with tabs[2]:
        st.subheader("Evaluation & Visualization")
        
        # Check if clustering results exist
        if st.session_state.clustering_results is None:
            st.warning("Please run clustering in the Clustering tab first.")
            st.stop()
        
        # Get results from session state
        results = st.session_state.clustering_results
        original_data = st.session_state.clustering_data
        
        # Visualization options
        st.write("### Cluster Visualization")
        
        # Determine which data to visualize
        if st.session_state.clustering_pca_data is not None:
            # If PCA was applied, use PCA data
            vis_data = st.session_state.clustering_pca_data.copy()
            vis_data['cluster'] = results['labels']
            
            # Default to first two PCA components
            default_x = vis_data.columns[0]  # PC1
            default_y = vis_data.columns[1]  # PC2
            
            # Create a 2D scatter plot of clusters
            x_col = st.selectbox("Select X-axis:", vis_data.columns[:-1], index=0)
            y_col = st.selectbox("Select Y-axis:", vis_data.columns[:-1], index=1)
        
        else:
            # If no PCA, use original features
            vis_data = original_data.copy()
            vis_data['cluster'] = results['labels']
            
            # Select columns to plot
            available_cols = list(original_data.columns)
            x_col = st.selectbox("Select X-axis:", available_cols, index=min(0, len(available_cols)-1))
            y_col = st.selectbox("Select Y-axis:", available_cols, index=min(1, len(available_cols)-1))
        
        # 2D Scatter plot of clusters
        cluster_labels = vis_data['cluster'].copy()
        
        # For DBSCAN, replace -1 with "Noise" for better visualization
        if results['algorithm'] == "DBSCAN":
            cluster_labels = cluster_labels.replace(-1, "Noise")
            
        # Create scatter plot with Plotly
        fig = px.scatter(
            vis_data, 
            x=x_col, 
            y=y_col, 
            color=cluster_labels if results['algorithm'] != "DBSCAN" else cluster_labels.astype(str),
            title=f"Cluster Visualization ({results['algorithm']})",
            opacity=0.7,
            height=600,
            labels={"color": "Cluster"}
        )
        
        # Update layout for better visualization
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization if PCA has at least 3 components
        if st.session_state.clustering_pca_data is not None and st.session_state.clustering_pca_data.shape[1] >= 3:
            st.write("### 3D Cluster Visualization")
            
            with st.expander("Show 3D Visualization"):
                # Select dimensions for 3D plot
                cols = st.session_state.clustering_pca_data.columns
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col_3d = st.selectbox("X-axis (3D):", cols, index=0)
                with col2:
                    y_col_3d = st.selectbox("Y-axis (3D):", cols, index=1)
                with col3:
                    z_col_3d = st.selectbox("Z-axis (3D):", cols, index=2)
                
                # Create 3D scatter plot
                fig_3d = px.scatter_3d(
                    vis_data,
                    x=x_col_3d,
                    y=y_col_3d,
                    z=z_col_3d,
                    color=cluster_labels if results['algorithm'] != "DBSCAN" else cluster_labels.astype(str),
                    opacity=0.7,
                    title=f"3D Cluster Visualization ({results['algorithm']})",
                    height=700
                )
                
                # Update layout
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=x_col_3d,
                        yaxis_title=y_col_3d,
                        zaxis_title=z_col_3d
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
        
        # Cluster profiling
        st.write("### Cluster Profiling")
        
        with st.expander("Cluster Characteristics"):
            # Add the clusters back to the original features
            profiling_data = pd.DataFrame(original_data)
            profiling_data['cluster'] = results['labels']
            
            # For DBSCAN, replace -1 with "Noise" for better labeling
            if results['algorithm'] == "DBSCAN":
                profiling_data['cluster'] = profiling_data['cluster'].replace(-1, -999)
                cluster_name_map = {-999: "Noise"}
                cluster_names = ["Noise"] + [f"Cluster {i}" for i in sorted(list(set(profiling_data['cluster'].unique()) - {-999}))]
            else:
                cluster_name_map = {}
                cluster_names = [f"Cluster {i}" for i in sorted(profiling_data['cluster'].unique())]
            
            # Calculate cluster means for each feature
            cluster_means = profiling_data.groupby('cluster').mean()
            
            # Rename cluster index if needed
            if cluster_name_map:
                cluster_means = cluster_means.rename(index=cluster_name_map)
            
            # Display cluster means
            st.write("#### Cluster Means")
            st.dataframe(cluster_means)
            
            # Create a radar chart or parallel coordinates plot for cluster profiles
            st.write("#### Cluster Profiles")
            
            # Normalize data for the plot
            if st.session_state.clustering_scaler is None:
                scaler = MinMaxScaler()
                normalized_means = pd.DataFrame(
                    scaler.fit_transform(cluster_means),
                    index=cluster_means.index,
                    columns=cluster_means.columns
                )
            else:
                normalized_means = cluster_means
            
            # Parallel coordinates plot
            # Prepare data for parallel coordinates
            parallel_data = normalized_means.reset_index()
            parallel_data = parallel_data.melt(
                id_vars=['cluster'],
                var_name='Feature',
                value_name='Value'
            )
            
            # Create the parallel coordinates plot
            fig = px.parallel_coordinates(
                parallel_data,
                color='cluster',
                labels={'cluster': 'Cluster', 'Value': 'Normalized Value', 'Feature': 'Feature'},
                title='Cluster Profiles Comparison'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance per cluster (which features distinguish each cluster)
            st.write("#### Feature Importance for Clusters")
            
            # Calculate standard deviations for each feature across clusters
            feature_std = cluster_means.std()
            feature_importance = feature_std.sort_values(ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importance.plot(kind='bar', ax=ax)
            ax.set_title('Feature Importance for Cluster Separation')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Standard Deviation Across Clusters')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **Interpreting Feature Importance:**
            Features with higher standard deviation across clusters are more important for distinguishing between clusters.
            These features show more variation between different clusters and help define the cluster boundaries.
            """)
            
        # Downloadable results
        st.write("### Export Results")
        
        # Add the clusters back to the original features for export
        export_data = pd.DataFrame(st.session_state.clustering_data, columns=st.session_state.clustering_features)
        export_data['cluster'] = results['labels']
        
        # For DBSCAN, replace -1 with "Noise" for better labeling in export
        if results['algorithm'] == "DBSCAN":
            export_data['cluster'] = export_data['cluster'].replace(-1, "Noise")
        
        # Create download button
        csv = export_data.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Clustering Results as CSV",
            data=csv,
            file_name=f"clustering_results_{results['algorithm']}.csv",
            mime="text/csv"
        )
    
    # Code explanation
    show_code("""
# Example clustering code for your own analysis:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('your_data.csv')

# Select features for clustering
features = ['feature1', 'feature2', 'feature3', 'feature4']
X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for visualization (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_}")

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the data
df['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
plt.title('Cluster Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analyze cluster characteristics
cluster_means = df.groupby('cluster')[features].mean()
print("Cluster means:")
print(cluster_means)
    """) 