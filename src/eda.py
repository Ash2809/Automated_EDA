import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_data(df):
    summary = {
        'Data Types': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Basic Statistics': df.describe(),
        'Shape': df.shape,
        'Columns': df.columns.tolist()
    }
    return summary

def create_visualizations(df):
    visualizations = []

    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    heatmap_path = "heatmap.png"
    plt.savefig(heatmap_path)
    visualizations.append(heatmap_path)

    if df.shape[0] <= 1000:  
        pairplot_path = "pairplot.png"
        sns.pairplot(df)
        plt.savefig(pairplot_path)
        visualizations.append(pairplot_path)

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        boxplot_path = f"boxplot_{col}.png"
        plt.savefig(boxplot_path)
        visualizations.append(boxplot_path)

    return visualizations


if __name__ == "__main__":
    path = r"C:\Users\aashutosh kumar\Music\abalone.csv"
    df = pd.read_csv(path)
    summary = summarize_data(df)
    print("Dataset Summary:")
    for key, value in summary.items():
        print(f"{key}:")
        print(value)
        print("-" * 50)

    print("Creating visualizations...")
    visualizations = create_visualizations(df)
    print("Visualizations saved:")
    for viz in visualizations:
        print(viz)