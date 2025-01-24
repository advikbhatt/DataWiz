import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Data handling and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Metrics and evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# AutoML
from tpot import TPOTClassifier

# Miscellaneous
import numpy as np
import os


# Page configuration
st.set_page_config(page_title="Data Visualization App", layout="wide")

# Title and description
st.title("📊 Data Visualization Web App")
st.write("Upload your CSV or Excel file to generate visualizations interactively and get AI-based model suggestions.")

# Custom styling for aesthetics
st.markdown("""
    <style>
        .container {
            display: flex;
            gap: 20px;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .graph-container {
            width: 48%;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #f9f9f9;
        }
        .graph-container h3 {
            font-size: 16px;
            color: #333;
        }
        .graph-container p {
            font-size: 12px;
            color: #555;
        }
        .button-container {
            display: flex;
            gap: 20px;
        }
        .button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            font-size: 14px;
            cursor: pointer;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("Upload a file (CSV or Excel)", type=["csv", "xlsx"])
ml=uploaded_file
maxUploadSize = 1024

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Dataset Preview
        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(data)

        # Dataset Summary
        with st.expander("Dataset Summary", expanded=False):
            col1, col2,col3,col4 = st.columns(4)

            with col1:
                
                numeric_data = data.select_dtypes(include=["number"])
                st.write(f"**Mean of Numeric Columns:**")
                st.write(numeric_data.mean())

                
            with col2:
                st.write(f"**Median of Numeric Columns:**")
                st.write(numeric_data.median())

               
            with col3:
                st.write(f"**Mode of Numeric Columns:**")
                st.write(numeric_data.mode().iloc[0])
                
            with col4:
                st.write(f"**Data types:**")
                st.dataframe(data.dtypes)
            st.write(f"**Shape of the dataset:** {data.shape}")

            st.write(f"**Summary Statistics for Numeric Columns:**")
            st.write(numeric_data.describe())


 

        # Visualizations inside Expander
        with st.expander("Visualizations", expanded=False):
            st.write("### Visualizations")
            col1, col2 = st.columns(2)

            with col1:
                # Bar Plot
                st.subheader("Bar Plot")
                numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
                categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

                num_rows_bar = st.number_input("Number of rows for Bar Plot:", min_value=1, max_value=len(data), value=10, step=1)
                cat_column = st.selectbox("Select a categorical column for bar plot:", categorical_columns)
                num_column = st.selectbox("Select a numerical column for bar plot:", numeric_columns)

                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=data[cat_column].head(num_rows_bar), y=data[num_column].head(num_rows_bar), palette="viridis", ax=ax)
                    plt.title(f"Bar Plot: {cat_column} vs {num_column}", fontsize=14)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(fontsize=10)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating bar plot: {e}")

                # Pie Chart
                st.subheader("Pie Chart")
                num_rows_pie = st.number_input("Number of rows for Pie Chart:", min_value=1, max_value=len(data), value=10, step=1)
                pie_column = st.selectbox("Select a categorical column for pie chart:", categorical_columns)
                try:
                    pie_data = data[pie_column].value_counts().head(num_rows_pie)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel", len(pie_data)))
                    ax.axis('equal')  
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating pie chart: {e}")

                #Scatter Plot
                st.subheader("Scatter Plot")
                num_rows_scatter = st.number_input("Number of rows for Scatter Plot:", min_value=1, max_value=len(data), value=10, step=1)
                scatter_x_column = st.selectbox("Select x-axis column for scatter plot:", numeric_columns)
                scatter_y_column = st.selectbox("Select y-axis column for scatter plot:", numeric_columns)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(data[scatter_x_column].head(num_rows_scatter), data[scatter_y_column].head(num_rows_scatter), alpha=0.6)
                    plt.title(f"Scatter Plot: {scatter_x_column} vs {scatter_y_column}", fontsize=14)
                    ax.set_xlabel(scatter_x_column, fontsize=12)
                    ax.set_ylabel(scatter_y_column, fontsize=12)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating scatter plot: {e}")
                
                #Area Chart
                st.subheader("Area Chart")
                num_rows_area = st.number_input("Number of rows for Area Chart:", min_value=1, max_value=len(data), value=10, step=1)
                area_column = st.selectbox("Select a numerical column for area chart:", numeric_columns)
                try:
                    fig = px.area(data[area_column].head(num_rows_area), title=f"Area Chart for {area_column}")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error generating area chart: {e}")
        
                # Line Graph
                st.subheader("Line Graph")
                num_rows_line = st.number_input("Number of rows for Line Graph:", min_value=1, max_value=len(data), value=10, step=1)
                line_column = st.selectbox("Select a numerical column for line graph:", numeric_columns)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(data[line_column].head(num_rows_line), label=f'{line_column} over Index', color='tab:blue')
                    ax.set_title(f"Line Graph of {line_column}", fontsize=14)
                    ax.set_xlabel('Index', fontsize=12)
                    ax.set_ylabel(line_column, fontsize=12)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating line graph: {e}")
            
            with col2:
                # Heatmap
                st.subheader("Heatmap")
                try:
                    numeric_data = data.select_dtypes(include=["number"])

                    if not numeric_data.empty:
                        corr_matrix = numeric_data.corr()
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                        plt.title("Correlation Heatmap", fontsize=14)
                        st.pyplot(fig)
                    else:
                        st.write("No numeric columns available for generating the heatmap.")
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")


                # Histogram
                st.subheader("Histogram")
                hist_column = st.selectbox("Select a numerical column for histogram:", numeric_columns)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(data[hist_column].dropna(), bins=20, color='skyblue', edgecolor='black')
                    plt.title(f"Histogram of {hist_column}", fontsize=14)
                    plt.xlabel(hist_column, fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating histogram: {e}")

                # Boxplot
                st.subheader("Boxplot")
                num_rows_box = st.number_input("Number of rows for Boxplot:", min_value=1, max_value=len(data), value=10, step=1)
                box_column = st.selectbox("Select a numerical column for boxplot:", numeric_columns)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(data=data[box_column].head(num_rows_box), color="lightgreen", ax=ax)
                    plt.title(f"Boxplot of {box_column}", fontsize=14)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(fontsize=10)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error generating boxplot: {e}")

                # Waterfall Chart
                st.subheader("Waterfall Chart")
                num_rows_waterfall = st.number_input("Number of rows for Waterfall Chart:", min_value=1, max_value=len(data), value=10, step=1)
                waterfall_column = st.selectbox("Select a numerical column for waterfall chart:", numeric_columns)
                try:
                    fig = go.Figure(go.Waterfall(
                        y=data[waterfall_column].head(num_rows_waterfall),
                        measure=["relative"]*len(data[waterfall_column].head(num_rows_waterfall)),
                        name="Waterfall Chart"
                    ))
                    fig.update_layout(title=f"Waterfall Chart for {waterfall_column}")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error generating waterfall chart: {e}")

                # Density Plot
                st.subheader("Density Plot")
                num_rows_density = st.number_input("Number of rows for Density Plot:", min_value=1, max_value=len(data), value=10, step=1)
                density_column = st.selectbox("Select a numerical column for density plot:", numeric_columns)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.kdeplot(data[density_column].head(num_rows_density), shade=True, ax=ax)
                    plt.title(f"Density Plot of {density_column}", fontsize=14)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating density plot: {e}")

                
                # Violin Plot   
                st.subheader("Violin Plot")
                num_rows_violin = st.number_input("Number of rows for Violin Plot:", min_value=1, max_value=len(data), value=10, step=1)
                violin_x_column = st.selectbox("Select x-axis column for violin plot:", categorical_columns)

                violin_y_column = st.selectbox("Select y-axis column for violin plot:", numeric_columns)

                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.violinplot(x=violin_x_column, y=violin_y_column, data=data.head(num_rows_violin), ax=ax)
                    plt.title(f"Violin Plot: {violin_x_column} vs {violin_y_column}", fontsize=14)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(fontsize=10)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating violin plot: {e}")

            
        # Machine Learning Model Suggestions
        
        

    except Exception as e:
        st.error(f"Error reading file: {e}")
