import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to load CSV file
def load_csv():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# Function to detect numeric columns
def get_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return numeric_cols

# Function to edit numeric columns
def edit_columns(numeric_cols):
    edited_cols = st.multiselect("Select numeric columns to include", numeric_cols, default=numeric_cols)
    return edited_cols

# Function to clean data
def clean_data(df, numeric_cols):
    # Handle missing values
    st.write("Handling missing values...")
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)  # Replace with mean
    st.write("Missing values replaced with mean.")

    # Handle outliers
    st.write("Handling outliers...")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    st.write("Outliers removed.")

    # Handle duplicates
    st.write("Handling duplicates...")
    df.drop_duplicates(inplace=True)
    st.write("Duplicates removed.")

    return df

# Function to plot correlation graphs
def plot_correlations(df, cols):
    if len(cols) > 1:
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Select at least two numeric columns to plot correlations.")

# Main function to run the app
def main():
    st.title("Correlation Graphs for Numeric Columns")

    df = load_csv()

    if df is not None:
        st.write("DataFrame loaded successfully!")
        numeric_cols = get_numeric_columns(df)

        if numeric_cols:
            st.write("Numeric columns detected:")
            st.write(numeric_cols)

            edited_cols = edit_columns(numeric_cols)

            if st.button("Clean and Generate Correlation Graphs"):
                df = clean_data(df, edited_cols)
                plot_correlations(df, edited_cols)
        else:
            st.warning("No numeric columns found in the uploaded CSV file.")
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
