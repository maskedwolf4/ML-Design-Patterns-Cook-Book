import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorationPipeline:
    def __init__(self, data):
        self.data = data

    def initial_inspection(self):
        print("First 5 rows:")
        print(self.data.head())
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nMissing Values:")
        print(self.data.isnull().sum())

    def check_unique_values(self, column):
        print(f"Unique values in {column}:")
        print(self.data[column].unique())

    def handle_missing_values(self):
        # Strategy 1: Imputation
        self.data.fillna(self.data.mean(), inplace=True)  # For numerical columns
        self.data.fillna(self.data.mode().iloc[0], inplace=True)  # For categorical columns

        # Strategy 2: Removal
        # self.data.dropna(inplace=True)

    def detect_outliers(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        print(f"Outliers in {column}:")
        print(outliers)

    def explore_numerical_features(self):
        numerical_features = self.data.select_dtypes(include=np.number)
        sns.pairplot(numerical_features)
        plt.show()

    def explore_categorical_features(self):
        categorical_features = self.data.select_dtypes(include='object')
        for feature in categorical_features:
            sns.countplot(x=feature, data=self.data)
            plt.title(f"Countplot for {feature}")
            plt.show()

# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Create a data exploration pipeline object
pipeline = DataExplorationPipeline(data)

# Initial inspection
pipeline.initial_inspection()

# Check unique values in a categorical column
pipeline.check_unique_values("column_name")

# Handle missing values (choose a strategy)
pipeline.handle_missing_values()

# Detect outliers in a numerical column
pipeline.detect_outliers("column_name")

# Explore numerical features
pipeline.explore_numerical_features()

# Explore categorical features
pipeline.explore_categorical_features()