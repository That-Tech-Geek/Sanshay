import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to load CSV file
def load_csv():
    uploaded_file = st.file_uploader("Upload your company data file", type=["csv"])
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
    edited_cols = st.multiselect("Choose the financial metrics to analyze", numeric_cols, default=numeric_cols)
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
        st.write("How the financial metrics are related:")
        st.write(corr_coefficients)
        
        # Identify highly correlated columns
        highly_correlated_cols = [(i, j) for i in range(len(corr)) for j in range(i) if abs(corr.iloc[i, j]) > 0.8 and i!= j]
        st.write("Financial metrics that are strongly linked:")
        st.write(highly_correlated_cols)
        
        # Investment Recommendation
        investment_recommendation = []
        for col in cols:
            corr_mean = corr[col].mean()
            if corr_mean > 0.5:
                investment_recommendation.append((col, "Good investment opportunity", 0.7))  # 70% weighting for positive correlation
            elif corr_mean < -0.5:
                investment_recommendation.append((col, "Not a good investment opportunity", 0.3))  # 30% weighting for negative correlation
            else:
                investment_recommendation.append((col, "Neutral", 0.5))  # 50% weighting for neutral correlation
        
        st.write("Investment Recommendation:")
        st.write(investment_recommendation)
        
        # Final Verdict
        final_verdict = sum([x[2] for x in investment_recommendation]) / len(investment_recommendation)
        if final_verdict > 0.5:
            st.write("Final Verdict: This company is a good investment opportunity")
        elif final_verdict < 0.5:
            st.write("Final Verdict: This company is not a good investment opportunity")
        else:
            st.write("Final Verdict: Neutral")
    else:
        st.warning("Please select at least two financial metrics to analyze.")

# Main function to run the app
def main():
    st.title("Company Investment Analysis")

    df = load_csv()

    if df is not None:
        st.write("Company data loaded successfully!")
        numeric_cols = get_numeric_columns(df)

        if numeric_cols:
            st.write("Financial metrics detected:")
            st.write(numeric_cols)

            edited_cols = edit_columns(numeric_cols)

            if st.button("Analyze Company Data"):
                plot_correlations(df, edited_cols)
        else:
            st.warning("No financial metrics found in the uploaded file.")
    else:
        st.info("Please upload a company data file to get started.")

if __name__ == "__main__":
    main()
