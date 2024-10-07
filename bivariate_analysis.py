# bivariate analysis using factory design pattern

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalysis:
    def __init__(self, data):
        self.data = data

    def analyze(self, x_col, y_col):
        if self.data[x_col].dtype == 'object' and self.data[y_col].dtype == 'object':
            self.analyze_categorical_categorical(x_col, y_col)
        elif (self.data[x_col].dtype == 'object' and self.data[y_col].dtype != 'object') or (self.data[x_col].dtype != 'object' and self.data[y_col].dtype == 'object'):
            self.analyze_categorical_numerical(x_col, y_col)
        else:
            self.analyze_numerical_numerical(x_col, y_col)

    def analyze_categorical_categorical(self, x_col, y_col):
        print(f"Categorical-Categorical Analysis for {x_col} and {y_col}")
        sns.countplot(x=x_col, hue=y_col, data=self.data)
        plt.title(f"Count Plot for {x_col} and {y_col}")
        plt.show()

    def analyze_categorical_numerical(self, x_col, y_col):
        print(f"Categorical-Numerical Analysis for {x_col} and {y_col}")
        sns.boxplot(x=x_col, y=y_col, data=self.data)
        plt.title(f"Box Plot for {x_col} and {y_col}")
        plt.show()

    def analyze_numerical_numerical(self, x_col, y_col):
        print(f"Numerical-Numerical Analysis for {x_col} and {y_col}")
        sns.scatterplot(x=x_col, y=y_col, data=self.data)
        plt.title(f"Scatter Plot for {x_col} and {y_col}")
        plt.show()

        correlation = self.data[[x_col, y_col]].corr().iloc[0, 1]
        print(f"Correlation: {correlation}")

class BivariateAnalysisFactory:
    @staticmethod
    def create_analysis(data):
        return BivariateAnalysis(data)


# How to use it
# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Create a bivariate analysis object using the factory
analysis = BivariateAnalysisFactory.create_analysis(data)

# Analyze specific columns
analysis.analyze("column1", "column2")
analysis.analyze("column3", "column4")