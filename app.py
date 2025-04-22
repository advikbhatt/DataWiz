# Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the fucntions 
from integration_handler import database_and_upload_interface
from create_visualization import create_visualization
from suggest_visualizations import suggest_visualizations
from generate_column_stats import generate_column_stats
from generate_data_profile import generate_data_profile
from handle_missing_values import handle_missing_values
from parse_uploaded_file import parse_uploaded_file 
from detect_outlier import detect_outliers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
import plotly.figure_factory as ff
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import DBSCAN, AgglomerativeClustering


st.set_page_config(
    page_title="Data Wiz",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
    }
    
    h1 {
        color: #f0f2f6;
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #f0f2f6;
        font-size: 1.6rem;
        margin-bottom: 1rem;
    }
    
    .card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d323e;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background-color: #252a37;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #343b4d;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #64b5f6;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
    }
    
    div.stButton > button {
        background-color: #1a1c24;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    div.stButton > button:hover {
        background-color: #1a1c24;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c24;
        border-radius: 8px;
        color: #a0aec0;
        padding: 0.75rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #252a37;
        color: #f0f2f6;
    }
    
    .upload-container {
        border: 2px dashed #343b4d;
        border-radius: 10px;
        padding: 2rem 1rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.2s;
    }
    
    .upload-container:hover {
        border-color: #4a56a6;
        background-color: #252a37;
    }
    
    .stProgress > div > div {
        background-color: #4a56a6;
    }
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .connection-panel {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .connection-card {
        background-color: #252a37;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #343b4d;
        transition: all 0.2s;
    }
    
    .connection-card:hover {
        border-color: #4a56a6;
        transform: translateY(-2px);
    }
    
    .info-box {
        background-color: rgba(74, 86, 166, 0.1);
        border-left: 4px solid #4a56a6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .controls-bar {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .header {
        padding: 10px;
        z-index: 100;
        color: white;
        text-align: center;
        display: flex;
        justify-content: space-between;
        align-items: center       
    }
    .title-container {
        display: flex;
        align-items: center;
    }
    </style>

""", unsafe_allow_html=True)

if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'upload'
with st.container():

        st.markdown("""
            <div class="title-container">
                <h1 style="display:flex; align-items:center;">
                    <img src="data:image/png;base64,{}" style="width:100px;height:100px;margin-right:10px;"/> DataWiz
                </h1>
            </div>
        """.format(base64.b64encode(open("logo.png", "rb").read()).decode()), unsafe_allow_html=True)

if st.session_state.data is None:
    database_and_upload_interface()
    # st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    # st.markdown('</div>', unsafe_allow_html=True)
    # if uploaded_file is not None:
    #     with st.spinner("Processing file..."):
    #         data, file_name, file_type = parse_uploaded_file(uploaded_file)
            
    #         if data is not None:
    #             st.session_state.data = data
    #             st.session_state.file_name = file_name
    #             st.session_state.file_type = file_type
    #             st.success(f"File '{file_name}' loaded successfully")
    #             st.rerun()
else:
    data = st.session_state.data
    st.title("Dashboard")
    if st.session_state.file_name:
        col1, col2 = st.columns([0.9,0.1])
        col1.markdown(f"**Source:** {st.session_state.file_name}")
        if col2.button("❌"):
            st.session_state.data = None
            st.session_state.file_name = None
            st.session_state.file_type = None
            st.rerun()
    
    tabs = st.tabs(["Overview", "Data Explorer", "Error Handling", "Visualizations","ML"])
    
    with tabs[0]:
        cleaned_data, missing_info = handle_missing_values(data)
        profile = generate_data_profile(data)
        metric1, metric2, metric3, metric4, metric5,metric6,metric7= st.columns(7)
        
        with metric1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{profile['row_count']:,}</div>
                <div class="metric-label">Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{profile['column_count']}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{profile['memory_usage']}</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            """, unsafe_allow_html=True)
   
        with metric4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(profile['data_types'])}</div>
                <div class="metric-label">Unique Data Types</div>
            </div>
            """, unsafe_allow_html=True)

        with metric5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sum(profile['data_types'].values())}</div>
                <div class="metric-label">Total Data Points</div>
            </div>
            """, unsafe_allow_html=True)
   
        with metric6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sum(abs(v) for v in profile['skewness'].values()) / len(profile['skewness']):.2f}</div>
                <div class="metric-label">Avg. Skewness</div>
            </div>
            """, unsafe_allow_html=True)
   
        with metric7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sum(profile['kurtosis'].values()) / len(profile['kurtosis']):.2f}</div>
                <div class="metric-label">Avg. Kurtosis</div>
            </div>
            """, unsafe_allow_html=True)

    with tabs[1]:                
        search_term = st.text_input("🔍 Search in data", "")
        
        if search_term:
            filtered_data = data[data.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)]
            if len(filtered_data) == 0:
                st.warning("No results found for your search term.")
            else:
                st.dataframe(filtered_data.head(100), use_container_width=True)
                if len(filtered_data) > 100:
                    st.caption(f"Showing 100 of {len(filtered_data)} matching rows")
        else:
            st.dataframe(data.head(100), use_container_width=True)
            if len(data) > 100:
                st.caption(f"Showing 100 of {len(data)} rows")
        
        st.subheader("Column Analysis")
        
        selected_column = st.selectbox("Select a column to analyze", data.columns)
        col_stats = generate_column_stats(data, selected_column)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Type:** {col_stats['type'].capitalize()}")
            st.markdown(f"**Unique Values:** {col_stats['unique_values']}")
            st.markdown(f"**Missing Values:** {col_stats['missing_count']} ({col_stats['missing_percent']:.2f}%)")
            
            if col_stats['type'] == 'numeric':
                stats_cols = st.columns(3)
                
                with stats_cols[0]:
                    st.metric("Min", f"{col_stats.get('min', 'N/A'):.2f}" if col_stats.get('min') is not None else "N/A")
                    st.metric("Median", f"{col_stats.get('median', 'N/A'):.2f}" if col_stats.get('median') is not None else "N/A")
                    
                with stats_cols[1]:
                    st.metric("Max", f"{col_stats.get('max', 'N/A'):.2f}" if col_stats.get('max') is not None else "N/A")
                    st.metric("Mean", f"{col_stats.get('mean', 'N/A'):.2f}" if col_stats.get('mean') is not None else "N/A")
                    
                with stats_cols[2]:
                    st.metric("Std Dev", f"{col_stats.get('std', 'N/A'):.2f}" if col_stats.get('std') is not None else "N/A")
                    st.metric("Skewness", f"{col_stats.get('skewness', 'N/A'):.2f}" if col_stats.get('skewness') is not None else "N/A")
        
        with col2:
            if col_stats['type'] == 'numeric':
                fig = create_visualization(data, 'histogram', {'x': selected_column})
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"{selected_column}_hist")

            elif col_stats['type'] in ['categorical', 'boolean']:
                fig = create_visualization(data, 'bar', {'x': selected_column})
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"{selected_column}_bar")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:

        tab1, tab2, tab3= st.tabs(["Missing Values", "Duplicate Values", "Outliers"])
        with tab1:
            if len(missing_info) > 0:
                missing_df = pd.DataFrame(missing_info)

                missing_df = missing_df.sort_values('missing_percent', ascending=False)

                fig = px.bar(
                    missing_df, 
                    x='column', 
                    y='missing_percent',
                    title='Missing Values by Column (%)',
                    color_discrete_sequence=['#4a56a6'],
                    labels={'column': 'Column', 'missing_percent': 'Missing (%)'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f0f2f6'),
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{selected_column}_missing")
                st.subheader("Handle Missing Values")
                col1, col2 = st.columns(2)
                with col1:
                    missing_col = st.selectbox("Select column with missing values", missing_df['column'])

                with col2:
                    handling_method = st.selectbox(
                        "Choose handling method", 
                        ["Don't modify", "Drop rows", "Fill with mean/mode", "Fill with median", "Fill with zero", "Fill with custom value"]
                    )

                if handling_method != "Don't modify":
                    preview_data = data.copy()

                    if handling_method == "Drop rows":
                        preview_data = preview_data.dropna(subset=[missing_col])
                        st.markdown(f"**Result:** {len(data) - len(preview_data)} rows would be dropped")

                    elif handling_method == "Fill with mean/mode":
                        if pd.api.types.is_numeric_dtype(preview_data[missing_col]):
                            fill_value = preview_data[missing_col].mean()
                            preview_data[missing_col] = preview_data[missing_col].fillna(fill_value)
                            st.markdown(f"**Result:** Missing values would be filled with mean ({fill_value:.2f})")
                        else:
                            mode_value = preview_data[missing_col].mode()[0] if not preview_data[missing_col].mode().empty else "N/A"
                            preview_data[missing_col] = preview_data[missing_col].fillna(mode_value)
                            st.markdown(f"**Result:** Missing values would be filled with mode ({mode_value})")

                    elif handling_method == "Fill with median":
                        if pd.api.types.is_numeric_dtype(preview_data[missing_col]):
                            fill_value = preview_data[missing_col].median()
                            preview_data[missing_col] = preview_data[missing_col].fillna(fill_value)
                            st.markdown(f"**Result:** Missing values would be filled with median ({fill_value:.2f})")
                        else:
                            st.warning("Median can only be used with numeric columns.")

                    elif handling_method == "Fill with zero":
                        preview_data[missing_col] = preview_data[missing_col].fillna(0)
                        st.markdown(f"**Result:** Missing values would be filled with zero")

                    elif handling_method == "Fill with custom value":
                        custom_value = st.text_input("Enter custom value")
                        if custom_value:
                            if pd.api.types.is_numeric_dtype(preview_data[missing_col]):
                                try:
                                    custom_value = float(custom_value)
                                except ValueError:
                                    st.warning("Please enter a valid number for this column")

                            preview_data[missing_col] = preview_data[missing_col].fillna(custom_value)
                            st.markdown(f"**Result:** Missing values would be filled with '{custom_value}'")

                    if st.button("Apply Changes"):
                        st.session_state.data = preview_data
                        st.success(f"Missing values handled for column: {missing_col}")
                        st.rerun()
            else:
                st.info("No missing values found in this dataset.")
        
        with tab2:
            duplicate_rows = data[data.duplicated(keep=False)] 
            total_duplicates = data.duplicated().sum()

            if total_duplicates > 0:
                st.subheader("Duplicate Values in Dataset")

                st.write(f"Total Duplicate Rows: **{total_duplicates}**")
                st.dataframe(duplicate_rows, use_container_width=True)

                duplicate_summary = data.duplicated(subset=data.columns, keep=False).value_counts().reset_index()
                duplicate_summary.columns = ['Is Duplicate', 'Count']

                fig = px.bar(
                    duplicate_summary,
                    x='Is Duplicate',
                    y='Count',
                    title='Duplicate Row Distribution',
                    color='Is Duplicate',
                    color_discrete_map={True: '#D9534F', False: '#5CB85C'},
                    labels={'Is Duplicate': 'Duplicate?', 'Count': 'Number of Rows'}
                )

                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f0f2f6'),
                    margin=dict(l=40, r=40, t=50, b=40),
                )

                st.plotly_chart(fig, use_container_width=True, key="duplicate_distribution")
                st.subheader("Handle Duplicate Values")

                handling_method = st.selectbox(
                    "Choose a method to handle duplicates:",
                    ["Don't modify", "Remove all duplicates", "Keep first occurrence", "Keep last occurrence"]
                )

                if handling_method != "Don't modify":
                    preview_data = data.copy()

                    if handling_method == "Remove all duplicates":
                        preview_data = preview_data.drop_duplicates(keep=False)
                        st.markdown(f"**Result:** {total_duplicates} rows would be completely removed.")

                    elif handling_method == "Keep first occurrence":
                        preview_data = preview_data.drop_duplicates(keep="first")
                        st.markdown(f"**Result:** {total_duplicates} rows would be modified, keeping the first occurrence.")

                    elif handling_method == "Keep last occurrence":
                        preview_data = preview_data.drop_duplicates(keep="last")
                        st.markdown(f"**Result:** {total_duplicates} rows would be modified, keeping the last occurrence.")

                    if st.button("Apply Change"):
                        st.session_state.data = preview_data
                        st.success("Duplicate values have been handled successfully!")
                        st.rerun()
            else:
                st.info("No duplicate values found in this dataset.")

        with tab3:            
            data = st.session_state.get("data", None)
            if data is None:
                st.warning("No dataset found. Please upload a dataset.")
                st.stop()
            
            num_columns = data.select_dtypes(include=["number"]).columns.tolist()
            if not num_columns:
                st.warning("No numerical columns found for outlier detection.")
                st.stop()
            
            selected_column = st.selectbox("Select a numeric column for outlier detection", num_columns)
            outliers, lower_bound, upper_bound = detect_outliers(data, selected_column)
            
            if not outliers.empty:
                st.markdown(f"### {len(outliers)} Outliers detected in '{selected_column}'")
            
                fig = px.box(data, y=selected_column, title=f"Outlier Detection in {selected_column}")
                st.plotly_chart(fig, use_container_width=True, key=f"{selected_column}_boxplot")
            
                handling_method = st.selectbox("Choose an outlier handling method", 
                                               ["Don't modify", "Drop outliers", "Cap at bounds", "Replace with mean", "Replace with median"])
            
                preview_data = data.copy()
            
                if handling_method == "Drop outliers":
                    preview_data = preview_data[(preview_data[selected_column] >= lower_bound) & (preview_data[selected_column] <= upper_bound)]
                    st.markdown(f"**Result:** {len(data) - len(preview_data)} outliers would be dropped.")
            
                elif handling_method == "Cap at bounds":
                    preview_data[selected_column] = np.where(preview_data[selected_column] < lower_bound, lower_bound, preview_data[selected_column])
                    preview_data[selected_column] = np.where(preview_data[selected_column] > upper_bound, upper_bound, preview_data[selected_column])
                    st.markdown(f"**Result:** Outliers would be capped at [{lower_bound:.2f}, {upper_bound:.2f}].")
            
                elif handling_method == "Replace with mean":
                    mean_value = preview_data[selected_column].mean()
                    preview_data[selected_column] = np.where((preview_data[selected_column] < lower_bound) | (preview_data[selected_column] > upper_bound), mean_value, preview_data[selected_column])
                    st.markdown(f"**Result:** Outliers would be replaced with mean ({mean_value:.2f}).")
            
                elif handling_method == "Replace with median":
                    median_value = preview_data[selected_column].median()
                    preview_data[selected_column] = np.where((preview_data[selected_column] < lower_bound) | (preview_data[selected_column] > upper_bound), median_value, preview_data[selected_column])
                    st.markdown(f"**Result:** Outliers would be replaced with median ({median_value:.2f}).")
            
                before_after = pd.DataFrame({
                    'Before': data.loc[outliers.index, selected_column],
                    'After': preview_data.loc[outliers.index, selected_column]
                })
            
                st.markdown("**Before/After Preview (Outlier Rows Only):**")
                st.dataframe(before_after, use_container_width=True)
            
                if st.button("Apply Changess"):
                    st.session_state.data = preview_data
                    st.success(f"Outliers handled for column: {selected_column}")
                    st.rerun()
            else:
                st.info("No outliers detected in this column.")


    with tabs[3]:
        viz_tabs = st.tabs(["Custom", "Suggested", "Distribution", "Relationship"])

        with viz_tabs[0]:
            col1, col2, col3= st.columns(3)

            with col1:
                chart_type = st.selectbox(
                    "Select Chart Type", 
                    ["bar", "histogram", "pie", "scatter", "boxplot", "correlation", "line", "area", "violin"]
                )

            with col2:
                default_value = min(20, len(data))  
                max_rows = st.number_input("Number of Rows to Display", min_value=1, max_value=len(data), value=default_value, step=1)
                            
            with col3:
                filter_columns = st.multiselect("Filter Data By Column", data.columns.tolist())
                filters = {}
                for col in filter_columns:
                    unique_vals = data[col].dropna().unique()
                    filters[col] = st.selectbox(f"Select value for {col}", [None] + unique_vals.tolist())
            
            selected_cols = {}

            if chart_type == 'bar':
                x_col = st.selectbox("Select Column", data.columns, key="bar_x")
                selected_cols = {'x': x_col}
            elif chart_type == 'histogram':
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                x_col = st.selectbox("Select Numeric Column", numeric_cols, key="hist_x")
                selected_cols = {'x': x_col}
            elif chart_type == 'pie':
                cat_col = st.selectbox("Select Categorical Column", data.columns, key="pie_cat")
                selected_cols = {'sector': cat_col}
            elif chart_type == 'scatter':
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                x_col = st.selectbox("X-Axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-Axis", numeric_cols, key="scatter_y")
                selected_cols = {'x': x_col, 'y': y_col}
            elif chart_type == 'boxplot':
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                y_col = st.selectbox("Select Numeric Column", numeric_cols, key="box_y")
                selected_cols = {'y': y_col}
            elif chart_type in ['line', 'area']:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                x_col = st.selectbox("X-Axis", numeric_cols, key="line_x")
                y_col = st.selectbox("Y-Axis", numeric_cols, key="line_y")
                selected_cols = {'x': x_col, 'y': y_col}
            elif chart_type == 'violin':
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                y_col = st.selectbox("Select Numeric Column", numeric_cols, key="violin_y")
                selected_cols = {'y': y_col}
            elif chart_type == 'correlation':
                selected_cols = {}

            if selected_cols or chart_type == 'correlation':
                fig = create_visualization(data, chart_type, selected_cols, filters=filters, max_rows=max_rows)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"{chart_type}_custom")
                else:
                    st.warning("Could not create visualization with the selected options.")

        with viz_tabs[1]:
            suggested_viz = suggest_visualizations(data)
            if suggested_viz:
                st.subheader("Visualization Suggestions")
                
                suggestion_titles = [s['title'] for s in suggested_viz]
                selected_suggestion = st.selectbox("Select a suggestion", suggestion_titles)
                
                selected_idx = suggestion_titles.index(selected_suggestion)
                suggestion = suggested_viz[selected_idx]
                
                fig = create_visualization(data, suggestion['type'], suggestion['columns'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"suggestion_{selected_idx}")
                else:
                    st.warning("Could not create the suggested visualization.")
            else:
                st.info("No visualization suggestions available for this dataset.")
        with viz_tabs[2]:
            st.write("## Distribution Analysis")
            for col in data.select_dtypes(include=['number']).columns:
                hist_fig = create_visualization(data, 'histogram', {'x': col})
                if hist_fig:
                    st.plotly_chart(hist_fig, use_container_width=True)

        with viz_tabs[3]:
            st.write("## Relationship Analysis")

            col1, col2 = st.columns(2)

            col_types = {
                'Numeric vs Numeric': ['Scatter Plot', 'Line Chart'],
                'Numeric vs Categorical': ['Box Plot', 'Violin Plot', 'Bar Chart'],
                'Categorical vs Categorical': ['Heatmap', 'Stacked Bar Chart']
            }

            analysis_type = col1.selectbox("Select Analysis Type", list(col_types.keys()))
            chart_type = col2.selectbox("Select Chart Type", col_types[analysis_type])

            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            if analysis_type == 'Numeric vs Numeric' and len(numeric_cols) >= 2:
                x_col = st.selectbox("X-Axis", numeric_cols, key="rel_x")
                y_col = st.selectbox("Y-Axis", numeric_cols, key="rel_y")
                chart_map = {'Scatter Plot': 'scatter', 'Line Chart': 'line'}
                fig = create_visualization(data, chart_map[chart_type], {'x': x_col, 'y': y_col})

            elif analysis_type == 'Numeric vs Categorical' and numeric_cols and categorical_cols:
                num_col = st.selectbox("Numeric Column", numeric_cols, key="num_col")
                cat_col = st.selectbox("Categorical Column", categorical_cols, key="cat_col")
                chart_map = {'Box Plot': 'boxplot', 'Violin Plot': 'violin', 'Bar Chart': 'bar'}
                fig = create_visualization(data, chart_map[chart_type], {'y': num_col, 'group': cat_col, 'x': cat_col})

            elif analysis_type == 'Categorical vs Categorical' and len(categorical_cols) >= 2:
                cat_x = st.selectbox("Categorical Column 1", categorical_cols, key="cat_x")
                cat_y = st.selectbox("Categorical Column 2", categorical_cols, key="cat_y")
                chart_map = {'Heatmap': 'correlation', 'Stacked Bar Chart': 'bar'}
                fig = create_visualization(data, chart_map[chart_type], {'x': cat_x, 'group': cat_y})

            else:
                fig = None
                st.warning("Not enough columns available for the selected analysis.")

            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"relationship_{chart_type}")


        
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[4]:
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("No dataset found. Please upload a dataset on the main page.")
            st.stop()

        data = st.session_state.data.copy()

        learning_type = st.tabs(["Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning", "Reinforcement Learning"])

        def evaluate_regression(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                "MAE": np.mean(abs(y_test - y_pred)),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R² Score": model.score(X_test, y_test)
            }

        def evaluate_classification(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            if y_prob is not None:
                metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
            return metrics, confusion_matrix(y_test, y_pred)

        with learning_type[0]:  
            ml_problem = st.selectbox("Choose Task Type", ["Regression", "Classification"], key="ml_problem")
            train_size = st.slider("Select Training Data Percentage", 50, 90, 80, step=5) / 100

            if ml_problem == "Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(),
                    "Support Vector Regression": SVR(),
                    "Random Forest Regression": RandomForestRegressor(),
                    "XGBoost Regression": GradientBoostingRegressor()
                }
            else:
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Support Vector Machine": SVC(probability=True),
                    "Gradient Boosting": GradientBoostingClassifier()
                }

            selected_model_name = st.selectbox("Select Model", list(models.keys()))
            selected_model = models[selected_model_name]

            valid_target_cols = data.select_dtypes(include=[np.number]).columns if ml_problem == 'Regression' else data.select_dtypes(exclude=[np.number]).columns
            target_col = st.selectbox("Select Target Column", valid_target_cols)
            valid_feature_cols = data.select_dtypes(include=[np.number]).columns if ml_problem == 'Regression' else data.columns
            feature_cols = st.multiselect("Select Feature Columns", [col for col in valid_feature_cols if col != target_col])

            if feature_cols:
                X = data[feature_cols]
                y = data[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

                if ml_problem == "Regression":
                    results = evaluate_regression(selected_model, X_train, X_test, y_train, y_test)
                    st.write("### Model Performance Metrics")
                    st.write(results)
                else:
                    results, conf_matrix = evaluate_classification(selected_model, X_train, X_test, y_train, y_test)
                    st.write("### Model Performance Metrics")
                    st.write(results)

                    st.write("### Confusion Matrix")
                    fig = ff.create_annotated_heatmap(z=conf_matrix, colorscale='Blues')
                    st.plotly_chart(fig, use_container_width=True, key="confusion_matrix")

        with learning_type[1]:  
            unsupervised_task = st.selectbox("Select Task", ["Clustering", "Dimensionality Reduction"], key="unsupervised_task")

            if unsupervised_task == "Clustering":
                models = {
                    "K-Means Clustering": KMeans(n_clusters=3),
                    "Hierarchical Clustering": AgglomerativeClustering(n_clusters=3),
                    "DBSCAN": DBSCAN(),
                    "Gaussian Mixture Model": GaussianMixture(n_components=3)
                }
            else:
                models = {
                    "Principal Component Analysis (PCA)": PCA(n_components=2)
                }

            selected_model_name = st.selectbox("Select Model", list(models.keys()))
            selected_model = models[selected_model_name]
            feature_cols = st.multiselect("Select Feature Columns", data.columns)

            if feature_cols:
                X = data[feature_cols]
                if unsupervised_task == "Clustering":
                    cluster_labels = selected_model.fit_predict(X)
                    metrics = {
                        "Silhouette Score": silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else "N/A",
                        "Davies-Bouldin Index": davies_bouldin_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else "N/A"
                    }
                    st.write("### Cluster Evaluation Metrics")
                    st.write(metrics)
                else:
                    transformed_data = selected_model.fit_transform(X)
                    st.write("### Dimensionality Reduction Output")
                    st.write("Explained Variance Ratio:", selected_model.explained_variance_ratio_)

        with learning_type[2]: 
            models = {
                "Label Propagation": None,
                "Label Spreading": None
            }
            selected_model = st.selectbox("Select Model", list(models.keys()), key="unsupervised_model")

        with learning_type[3]:
            reinforcement_task = st.selectbox("Select Task", ["Value-Based Methods", "Policy-Based Methods", "Actor-Critic Methods"], key="reinforcement_task")

            if reinforcement_task == "Value-Based Methods":
                models = {
                    "Q-Learning": None,
                    "Deep Q Networks (DQN)": None
                }
            elif reinforcement_task == "Policy-Based Methods":
                models = {
                    "REINFORCE Algorithm": None,
                    "Proximal Policy Optimization (PPO)": None
                }
            elif reinforcement_task == "Actor-Critic Methods":
                models = {
                    "Advantage Actor-Critic (A2C)": None,
                    "Deep Deterministic Policy Gradient (DDPG)": None
                }
            selected_model = st.selectbox("Select Model", list(models.keys()))


st.markdown("---")
st.markdown('<p style="text-align: center; color: gray;">Data Wiz • @2025 </p>', unsafe_allow_html=True)