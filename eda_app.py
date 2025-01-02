import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from src.data_cleaning import clean_data

def summarize_data(df):
    st.write("### Dataset Summary")
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.write("**Basic Statistics:**")
    st.write(df.describe())
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())

def create_visualizations(df):
    st.write("## Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("## Pairplot")
    if df.shape[0] <= 1000:  
        pairplot_fig = sns.pairplot(df)
        st.pyplot(pairplot_fig.fig)

    st.write("## Boxplots")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        st.write(f"### Boxplot for {col}")
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    st.write("## Histograms")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        st.write(f"### Histogram for {col}")
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], kde=True, color='green', ax=ax)
        st.pyplot(fig)

def main():
    st.title("Data Summary and Visualizations")
    st.write("Upload a CSV file to analyze and visualize the data.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = clean_data(df)
        st.write("### Uploaded Dataset After Cleaning:")
        st.write(df.head(10))

        summarize_data(df)

        st.write("## Visualizations")
        create_visualizations(df)
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
