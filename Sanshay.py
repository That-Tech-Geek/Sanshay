import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import plotly.express as px

# Function to load CSV file
def load_csv():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except pd.errors.EmptyDataError:
            st.error("Error: The uploaded file is empty.")
            return None
        except pd.errors.ParserError:
            st.error("Error: The uploaded file is not a valid CSV file.")
            return None
    return None

# Function to detect numeric columns
def get_numeric_columns(df):
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        return numeric_cols
    except AttributeError:
        st.error("Error: The data frame is empty.")
        return []

# Function to edit numeric columns
def edit_columns(numeric_cols):
    try:
        edited_cols = st.multiselect("Select numeric columns to include", numeric_cols, default=numeric_cols)
        return edited_cols
    except TypeError:
        st.error("Error: No numeric columns found in the data frame.")
        return []

# Function to perform Pearson correlation test
def pearson_correlation_test(df, cols):
    try:
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
    except ValueError:
        st.error("Error: The selected columns are not numeric.")
        return {}, {}

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

# Function to analyze slicer data
def analyze_slicer_data(df):
    try:
        slicer_col = st.selectbox("Select a column to analyze slicer data", df.columns)
        unique_values = df[slicer_col].unique()
        slicer_value = st.selectbox(f"Select a value for {slicer_col}", unique_values)
        
        filtered_df = df[df[slicer_col] == slicer_value]
        
        st.write("Filtered DataFrame:")
        st.write(filtered_df)
        
        st.write("Summary statistics:")
        st.write(filtered_df.describe())
        
        # Identify duplicates
        duplicate_rows = filtered_df[filtered_df.duplicated()]
        st.write("Duplicate rows:")
        st.write(duplicate_rows)
        
        # Calculate summary statistics
        st.write("Mean:")
        st.write(filtered_df.mean())
        
        st.write("Median:")
        st.write(filtered_df.median())
        
        st.write("Mode:")
        st.write(filtered_df.mode().iloc[0])
        
        st.write("Standard Deviation:")
        st.write(filtered_df.std())
        
        st.write("Variance:")
        st.write(filtered_df.var())
        
        st.write("Minimum values:")
        st.write(filtered_df.min())
        
        st.write("Maximum values:")
        st.write(filtered_df.max())
        
        st.write("Quantiles:")
        st.write(filtered_df.quantile([0.25, 0.5, 0.75]))
        
        st.write("Correlation matrix:")
        st.write(filtered_df.corr())
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main function
def main():
    st.title("Data Analysis App")
    df = load_csv()
    if df is not None:
        numeric_cols = get_numeric_columns(df)
        edited_cols = edit_columns(numeric_cols)
        if len(edited_cols) > 0:
            plot_correlations(df, edited_cols)
            plot_time_series(df, edited_cols)
            analyze_slicer_data(df)

if __name__ == "__main__":
    main()
