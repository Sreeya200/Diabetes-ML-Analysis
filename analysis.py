import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_and_clean_data():
    """
    Load the Diabetes dataset and clean it for analysis
    Returns: cleaned pandas DataFrame
    """
    # Loading Diabetes dataset
    df = pd.read_csv('diabetes.csv')
    
    # Basic cleaning
    # Replace 0 values with NaN for certain columns where 0 is not possible
    zero_not_possible = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in zero_not_possible:
        df[column] = df[column].replace(0, np.nan)
    
    # Fill NaN values with median of respective columns
    for column in zero_not_possible:
        df[column] = df[column].fillna(df[column].median())
    
    return df
