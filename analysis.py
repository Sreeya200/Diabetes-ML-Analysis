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

def create_age_distribution_plot(df):
    """
    Create a histogram showing age distribution with diabetes status
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Outcome', bins=30, multiple="layer", alpha=0.6)
    plt.title('Age Distribution by Diabetes Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(labels=['No Diabetes', 'Diabetes'])
    plt.tight_layout()
    plt.savefig('age_distribution.png')
    plt.close()
    
    return df['Age'].describe()

def create_glucose_bmi_scatter(df):
    """
    Create a scatter plot of Glucose vs BMI with diabetes outcome color coding
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', alpha=0.6)
    plt.title('Glucose vs BMI by Diabetes Status')
    plt.xlabel('Glucose Level')
    plt.ylabel('BMI')
    plt.legend(labels=['No Diabetes', 'Diabetes'])
    plt.tight_layout()
    plt.savefig('glucose_bmi_scatter.png')
    plt.close()
    
    return df[['Glucose', 'BMI']].corr()

def create_correlation_heatmap(df):
    """
    Create a heatmap showing correlations between all features
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Diabetes Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    return correlation_matrix
