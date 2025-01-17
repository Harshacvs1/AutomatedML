import streamlit as st
import pandas as pd
import os

# Import profiling and PyCaret
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStream ML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build automated ML pipelines using Streamlit, Pandas Profiling, and PyCaret")

# Load Dataset
if os.path.exists("Sourcedataset.csv"):
    df = pd.read_csv("Sourcedataset.csv", index_col=None)

# Upload Page
if choice == "Upload":
    st.title("Upload Your Data for Modeling")
    file = st.file_uploader("Upload your dataset here!")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("Sourcedataset.csv", index=None)
        st.dataframe(df)

# Profiling Page
if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

# ML Page
if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target Column", df.columns)
    task = None

    # Determine Task Type
    if df[target].dtype == "object" or len(df[target].unique()) < 20:
        task = "classification"
    else:
        task = "regression"

    st.info(f"Detected task: {task.capitalize()}")

    if st.button("Train Model"):
        if task == "classification":
            cls_setup(df, target=target)
            best_model = cls_compare()
            st.info("This is the Best Model (Classification)")
            st.dataframe(pull())  # Pull only once after model comparison
            cls_save(best_model, "best_model")
        elif task == "regression":
            reg_setup(df, target=target)
            best_model = reg_compare()
            st.info("This is the Best Model (Regression)")
            st.dataframe(pull())  # Pull only once after model comparison
            reg_save(best_model, "best_model")

# Download Page
if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download Best Model", f, "trained_model.pkl")
