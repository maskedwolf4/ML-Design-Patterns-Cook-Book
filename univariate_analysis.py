# univariate analysis in factory design pattern

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysis:
    def __init__(self, data):
        self.data = data

    def analyze(self, column):
        if self.data[column].dtype == 'object':
            self.analyze_categorical(column)
        else:
            self.analyze_numerical(column)

    def analyze_categorical(self, column):
        print(f"Categorical Analysis for {column}")
        print(self.data[column].describe())
        print(self.data[column].value_counts())
        sns.countplot(x=column, data=self.data)
        plt.title(f"Count Plot for {column}")
        plt.show()

    def analyze_numerical(self, column):
        print(f"Numerical Analysis for {column}")
        print(self.data[column].describe())
        sns.histplot(self.data[column], kde=True)
        plt.title(f"Histogram for {column}")
        plt.show()

class UnivariateAnalysisFactory:
    @staticmethod
    def create_analysis(data):
        return UnivariateAnalysis(data)


# Instructions to use it
# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Create a univariate analysis object using the factory
analysis = UnivariateAnalysisFactory.create_analysis(data)

# Analyze specific columns
analysis.analyze("column1")
analysis.analyze("column2")