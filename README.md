# Diabetes Data Analysis and Classification

This project analyzes a diabetes dataset, generates visualizations, and trains a machine learning model to predict diabetes outcomes. It uses Python libraries such as Pandas, Seaborn, Matplotlib, and Scikit-Learn.

## Features

1. **Data Cleaning**: Replaces zero values with NaN for certain features where zero is invalid and fills NaN values with the median.
2. **Exploratory Data Analysis**: 
   - Histogram showing age distribution by diabetes status.
   - Scatter plot of glucose vs. BMI with diabetes outcome color coding.
   - Correlation heatmap of all features.
3. **Descriptive Statistics**: Provides basic statistics, correlation data, diabetes prevalence ratio, and feature importance.
4. **Model Training and Evaluation**: Trains a Random Forest classifier and provides a confusion matrix and classification report.

## Setup

1. Clone this repository.
2. Ensure you have the necessary libraries installed:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn numpy
   ```
3. Download and place the `diabetes.csv` dataset in the project directory.

## Usage

Run the code with:
```bash
python analysis.py
```

The script will:
- Clean the data
- Generate plots for analysis
- Train a Random Forest model and display evaluation metrics

## File Structure

- **diabetes.csv**: Dataset file (required).
- **age_distribution.png**: Saved histogram of age distribution by diabetes status.
- **glucose_bmi_scatter.png**: Saved scatter plot of glucose vs. BMI by diabetes status.
- **correlation_heatmap.png**: Saved correlation heatmap of all features.

## Functions

- **load_and_clean_data()**: Loads and cleans the dataset.
- **create_age_distribution_plot()**: Plots age distribution with diabetes status.
- **create_glucose_bmi_scatter()**: Creates scatter plot of Glucose vs BMI by diabetes status.
Here's a README for your code:

---

# Diabetes Data Analysis and Classification

This project analyzes a diabetes dataset, generates visualizations, and trains a machine learning model to predict diabetes outcomes. It uses Python libraries such as Pandas, Seaborn, Matplotlib, and Scikit-Learn.

## Features

1. **Data Cleaning**: Replaces zero values with NaN for certain features where zero is invalid and fills NaN values with the median.
2. **Exploratory Data Analysis**: 
   - Histogram showing age distribution by diabetes status.
   - Scatter plot of glucose vs. BMI with diabetes outcome color coding.
   - Correlation heatmap of all features.
3. **Descriptive Statistics**: Provides basic statistics, correlation data, diabetes prevalence ratio, and feature importance.
4. **Model Training and Evaluation**: Trains a Random Forest classifier and provides a confusion matrix and classification report.

## Setup

1. Clone this repository.
2. Ensure you have the necessary libraries installed:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn numpy
   ```
3. Download and place the `diabetes.csv` dataset in the project directory.

## Usage

Run the code with:
```bash
python script_name.py
```

The script will:
- Clean the data
- Generate plots for analysis
- Train a Random Forest model and display evaluation metrics

## File Structure

- **diabetes.csv**: Dataset file (required).
- **age_distribution.png**: Saved histogram of age distribution by diabetes status.
- **glucose_bmi_scatter.png**: Saved scatter plot of glucose vs. BMI by diabetes status.
- **correlation_heatmap.png**: Saved correlation heatmap of all features.

## Functions

- **load_and_clean_data()**: Loads and cleans the dataset.
- **create_age_distribution_plot()**: Plots age distribution with diabetes status.
- **create_glucose_bmi_scatter()**: Creates scatter plot of Glucose vs BMI by diabetes status.
- **create_correlation_heatmap()**: Plots heatmap of feature correlations.
- **train_and_evaluate_model()**: Trains and evaluates a Random Forest model.
- **generate_statistics()**: Generates basic statistics and feature correlations.

## Sample Output

- **Statistics**: Descriptive statistics of cleaned data.
- **Confusion Matrix and Classification Report**: Model evaluation metrics.

## License

This project is licensed under the MIT License.

--- 

This README provides an overview, setup instructions, function descriptions, and a sample output section for users. Let me know if you'd like any further details!- **create_correlation_heatmap()**: Plots heatmap of feature correlations.
- **train_and_evaluate_model()**: Trains and evaluates a Random Forest model.
- **generate_statistics()**: Generates basic statistics and feature correlations.

## Sample Output

- **Statistics**: Descriptive statistics of cleaned data.
- **Confusion Matrix and Classification Report**: Model evaluation metrics.

## License

This project is licensed under the MIT License.

--- 
