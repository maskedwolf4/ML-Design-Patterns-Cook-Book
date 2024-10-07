# Multivate analysis in factory design pattern

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class MultivariateAnalysis:
    def __init__(self, data):
        self.data = data

    def analyze(self, columns):
        if all(self.data[col].dtype == 'object' for col in columns):
            self.analyze_categorical_categorical(columns)
        elif any(self.data[col].dtype == 'object' for col in columns):
            self.analyze_mixed(columns)
        else:
            self.analyze_numerical_numerical(columns)

    def analyze_categorical_categorical(self, columns):
        print(f"Categorical-Categorical Analysis for columns: {columns}")
        sns.pairplot(self.data[columns])
        plt.show()

    def analyze_mixed(self, columns):
        print(f"Mixed Analysis for columns: {columns}")
        sns.pairplot(self.data[columns])
        plt.show()

    def analyze_numerical_numerical(self, columns):
        print(f"Numerical-Numerical Analysis for columns: {columns}")
        sns.pairplot(self.data[columns])
        plt.show()

        correlation_matrix = self.data[columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="viridis")
        plt.title("Correlation Matrix")
        plt.show()

        # Principal Component Analysis (PCA)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.data[columns])
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
        plt.title("PCA Scatter Plot")
        plt.show()

class MultivariateAnalysisFactory:
    @staticmethod
    def create_analysis(data):
        return MultivariateAnalysis(data)

# Example usage
# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Create a multivariate analysis object using the factory
analysis = MultivariateAnalysisFactory.create_analysis(data)

# Analyze specific columns
analysis.analyze(["column1", "column2", "column3"])