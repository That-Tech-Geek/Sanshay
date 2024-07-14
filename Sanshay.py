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
        
        # Select duplicates to delete
        delete_duplicates = st.multiselect("Select duplicates to delete", duplicate_rows.index)
        
        # Delete selected duplicates
        filtered_df.drop(delete_duplicates, inplace=True)
        
        # Identify rows with empty cells
        empty_rows = filtered_df[filtered_df.isnull().any(axis=1)]
        st.write("Rows with empty cells:")
        st.write(empty_rows)
        
        # Select rows with empty cells to delete
        delete_empty_rows = st.multiselect("Select rows with empty cells to delete", empty_rows.index)
        
        # Delete selected rows with empty cells
        filtered_df.drop(delete_empty_rows, inplace=True)
        
        # Identify outliers using the Z-score method
        from scipy import stats
        outliers = []
        for col in filtered_df.select_dtypes(include=['number']).columns:
            z_scores = np.abs(stats.zscore(filtered_df[col]))
            outliers.extend(filtered_df[(z_scores > 3)].index)
        outliers = list(set(outliers))
        
        st.write("Outlier rows:")
        st.write(filtered_df.loc[outliers])
        
        # Select outliers to delete
        delete_outliers = st.multiselect("Select outliers to delete", outliers)
        
        # Delete selected outliers
        filtered_df.drop(delete_outliers, inplace=True)
        
        st.write("Updated DataFrame after deleting duplicates, rows with empty cells, and outliers:")
        st.write(filtered_df)
        
        # Generate inferences using Gemini
        from gemini import Gemini
        gemini = Gemini(filtered_df)
        inferences = gemini.infer()
        
        st.write("Inferences:")
        for inference in inferences:
            st.write(f"* {inference}")
        
        # Create a CSV file
        @st.cache
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(filtered_df)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='filtered_data.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
# Main function to run the app
def main():
    st.title("Hey, I'm Sanshay, and I'm here to help you make Data analysis easier!")
    st.write("Sanshay is here to help you understand your dataset inside out, and to draw as many conclusions as you want from it.")
    df = load_csv()

    if df is not None:
        st.write("DataFrame loaded successfully!")

        # Remove rows with empty cells
        df.dropna(inplace=True)

        # Remove whitespaces from column names
        df.columns = [col.strip() for col in df.columns]

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        numeric_cols = get_numeric_columns(df)
        edited_cols = edit_columns(numeric_cols)

        if len(edited_cols) > 0:
            st.write("Selected numeric columns:")
            st.write(edited_cols)

            plot_correlations(df, edited_cols)
            plot_time_series(df, edited_cols)

            analyze_slicer_data(df)

if __name__ == "__main__":
    main()
