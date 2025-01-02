import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == 'object':  
                data[col] = data[col].fillna(data.mode()[0])
            else:  
                data[col] = imputer.fit_transform(data[[col]]).flatten()
    return data

def remove_duplicates(data):
    return data.drop_duplicates()

def remove_outliers(data):
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    z_scores = abs(zscore(data[numerical_columns]))
    data = data[(z_scores < 3).all(axis=1)] 
    return data

def clean_data(data, remove_outliers_flag=False):
    data = handle_missing_values(data)
    data = remove_duplicates(data)

    if remove_outliers_flag:
        data = remove_outliers(data)

    return data

if __name__ == "__main__":
    path = r"C:\Users\aashutosh kumar\Music\abalone.csv"
    data = pd.read_csv(path)

    print("Before data cleaning shape:", data.shape)

    cleaned_data = clean_data(data, remove_outliers_flag=True)

    print("After data cleaning shape:", cleaned_data.shape)
    print(cleaned_data.head(10))
