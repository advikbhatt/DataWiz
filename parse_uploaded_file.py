import pandas as pd
import streamlit as st
def parse_uploaded_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            data = pd.read_csv(uploaded_file)
        elif file_type == 'xls':
            data = pd.read_excel(uploaded_file, engine='xlrd')  
        elif file_type == 'xlsx':
            data = pd.read_excel(uploaded_file, engine='openpyxl')  
        elif file_type == 'json':
            data = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_type}")
            return None, None, None

        return data, uploaded_file.name, file_type
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None, None, None