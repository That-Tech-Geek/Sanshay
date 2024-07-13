import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import plotly.express as px

# Function to load CSV file
def load_csv():
    st.set_option("server.maxMessageSize", 1024 * 1024 * 1024)  # 1GB
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

# Function to perform Pearson correlation test
def pearson_correlation_test(df, cols):
    correlation_coefficients = {}
    p_values = {}
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            corr_coef, p_value = pearsonr(df[col1], df[col2])
            correlation_coefficients[f"{col1}_{col2}"] = corr_coef
            p_values[f"{col1}_{col2}"] = p_value
    return correlation_coefficients, p_values

# Function to plot correlation graphs
def plot_correlations(df, cols):
    try:
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
            highly_correlated_cols = [(cols[i], cols[j]) for i in range(len(corr)) for j in range(i) if abs(corr.iloc[i, j]) > 0.8 and i!= j]
            st.write("Financial metrics that are strongly linked:")
            st.write(highly_correlated_cols)
            
            # Perform Pearson correlation test
            correlation_coefficients, p_values = pearson_correlation_test(df, cols)
            data = []
            for key, value in correlation_coefficients.items():
                col1, col2 = key.split("_")
                idx1 = cols.index(col1)
                idx2 = cols.index(col2)
                data.append({
                    "Column 1": f"Column {idx1+1} ({col1})",
                    "Column 2": f"Column {idx2+1} ({col2})",
                    "Pearson Correlation Coefficient": value,
                    "p-value": p_values[key]
                })
            df = pd.DataFrame(data)
            st.write(df)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        plt.close('all')  # Close all matplotlib figures to avoid memory leak

# Function to plot time series graphs
def plot_time_series(df, cols):
    try:
        fig = px.line(df, x=df.index, y=cols)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main function to run the app
def main():
    st.title("Hey, I'm Sanshay, and I'm here to help you make Data analysis easier!")

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

            time_series_cols = st.multiselect("Select columns to plot time series graphs", numeric_cols, default=numeric_cols)
            if st.button("Generate Time Series Graphs"):
                plot_time_series(df, time_series_cols)
        else:
            st.warning("No numeric columns found in the uploaded CSV file.")
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
