
import streamlit as st
import pandas as pd
from parse_uploaded_file import parse_uploaded_file  

def database_and_upload_interface():
    st.markdown("## 📂 Data Source")
    db_tabs = st.tabs(["Upload File", "Connect MongoDB", "Connect SQL Database"])

    with db_tabs[0]:
        uploaded_files = st.file_uploader(
            "Upload one or more data files",
            type=["csv", "json", "xls", "xlsx"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                st.write(f"📂 **File Name:** `{file.name}`")
                data, file_name, file_type = parse_uploaded_file(file)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.file_name = file_name
                    st.session_state.file_type = file_type
                    st.success(f"File '{file_name}' loaded successfully")
                    st.rerun()

    with db_tabs[1]:
        st.subheader("🔗 Connect to MongoDB")
        with st.form("mongo_form"):
            mongo_uri = st.text_input("MongoDB URI", value="mongodb+srv://<username>:<password>@cluster.mongodb.net/")
            db_name = st.text_input("Database Name")
            collection_name = st.text_input("Collection Name")
            submitted = st.form_submit_button("Connect")
        if submitted:
            try:
                from pymongo import MongoClient
                client = MongoClient(mongo_uri)
                db = client[db_name]
                collection = db[collection_name]
                documents = list(collection.find())
                df = pd.DataFrame(documents)
                st.session_state.data = df
                st.session_state.file_name = f"{db_name}.{collection_name} (MongoDB)"
                st.success(f"✅ Connected to `{db_name}.{collection_name}`")
                st.rerun()
            except Exception as e:
                st.error(f"❌ MongoDB Connection Error: {e}")

    with db_tabs[2]:
        st.subheader("🔗 Connect to SQL Database")
        with st.form("sql_form"):
            db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL"])
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            db_name = st.text_input("Database Name")
            table_name = st.text_input("Table Name")
            submitted = st.form_submit_button("Connect")
        if submitted:
            try:
                from sqlalchemy import create_engine
                if db_type == "MySQL":
                    engine_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
                else:
                    engine_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
                engine = create_engine(engine_str)
                df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
                st.session_state.data = df
                st.session_state.file_name = f"{db_name}.{table_name} ({db_type})"
                st.success(f"✅ Connected to `{db_name}.{table_name}`")
                st.rerun()
            except Exception as e:
                st.error(f"❌ SQL Connection Error: {e}")
