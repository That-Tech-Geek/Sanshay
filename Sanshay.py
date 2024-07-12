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

# Function to plot correlation graphs
def plot_correlations(df, cols):
    if len(cols) > 1:
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Calculate and display correlation coefficients
        corr_coefficients = corr.unstack().sort_values(ascending=False)
        st.write("Correlation Coefficients:")
        st.write(corr_coefficients)
        
        # Identify highly correlated columns
        highly_correlated_cols = [(i, j) for i in range(len(corr)) for j in range(i) if abs(corr.iloc[i, j]) > 0.8]
        st.write("Highly Correlated Columns (|r| > 0.8):")
        st.write(highly_correlated_cols)
        
        # Investment Recommendation
        investment_recommendation = []
        for col in cols:
            if corr[col].mean() > 0.5:
                investment_recommendation.append((col, "Good to invest in"))
            elif corr[col].mean() < -0.5:
                investment_recommendation.append((col, "Not good to invest in"))
            else:
                investment_recommendation.append((col, "Neutral"))
        st.write("Investment Recommendation:")
        st.write(investment_recommendation)
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

            if st.button("Generate Correlation Graphs"):
                plot_correlations(df, edited_cols)
        else:
            st.warning("No numeric columns found in the uploaded CSV file.")
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
