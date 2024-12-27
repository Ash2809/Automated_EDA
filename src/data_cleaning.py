import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import re

def handle_missing_values(data):
    imputer = SimpleImputer(strategy = 'mean')
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
    numerical_columns = data.select_dtypes(include = ['int64', 'float64']).columns
    z_scores = abs(zscore(data[numerical_columns]).all(axis = 1))

    cleaned_data = data[(z_scores < 3)].all(axis = 1)
    return cleaned_data

# def clean_text_data(data):
#     # Apply the text cleaning to columns that are of 'object' (string) data type in the entire DataFrame
#     for col in data.select_dtypes(include=['object']).columns:
#         data[col] = data[col].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))  # Remove non-alphabet characters
#         data[col] = data[col].apply(lambda x: x.strip())  # Remove leading/trailing spaces
#         data[col] = data[col].apply(lambda x: ' '.join(x.split()))  # Remove extra spaces between words
#     return data


def clean_data(data, flag):
    data = handle_missing_values(data)
    data = remove_duplicates(data)

    if(flag):
        data = remove_outliers(data)
    
    # data = clean_text_data(data)

    return data

if __name__ == "__main__":
    path = r"C:\Users\aashutosh kumar\Music\abalone.csv"
    data = pd.read_csv(path)

    print("Before data cleaning shape", data.shape)

    cleaned_data = clean_data(data, flag=True)

    print("After data cleaning shape: ", cleaned_data.shape)
