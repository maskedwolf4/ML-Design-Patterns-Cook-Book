import pandas as pd
import numpy as np

class DataPreprocessor:
    def preprocess(self, data):
        return data

class ImputerDecorator(DataPreprocessor):
    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def preprocess(self, data):
        data = super().preprocess(data)
        if self.strategy == 'mean':
            data.fillna(data.mean(), inplace=True)
        elif self.strategy == 'median':
            data.fillna(data.median(), inplace=True)
        elif self.strategy == 'mode':
            data.fillna(data.mode().iloc[0], inplace=True)
        return data

class NormalizationDecorator(DataPreprocessor):
    def preprocess(self, data):
        data = super().preprocess(data)
        data = (data - data.min()) / (data.max() - data.min())
        return data

class StandardizationDecorator(DataPreprocessor):
    def preprocess(self, data):
        data = super().preprocess(data)
        data = (data - data.mean()) / data.std()
        return data

class OutlierHandlerDecorator(DataPreprocessor):
    def __init__(self, method='z-score', threshold=3):
        self.method = method
        self.threshold = threshold

    def preprocess(self, data):
        data = super().preprocess(data)
        if self.method == 'z-score':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > self.threshold
            data[outliers] = np.nan
            data = super().preprocess(data)  # Re-apply imputation
        # Add other outlier handling methods like IQR or capping
        return data

class CategoricalEncodingDecorator(DataPreprocessor):
    def __init__(self, encoding_type='one-hot'):
        self.encoding_type = encoding_type

    def preprocess(self, data):
        data = super().preprocess(data)
        if self.encoding_type == 'one-hot':
            data = pd.get_dummies(data, columns=data.select_dtypes(include='object').columns)
        # Add other encoding methods like label encoding or target encoding
        return data

class MappingDecorator(DataPreprocessor):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def preprocess(self, data):
        data = super().preprocess(data)
        for column, mapping in self.mapping_dict.items():
            data[column] = data[column].map(mapping)
        return data

# Create a preprocessing pipeline
preprocessor = DataPreprocessor()
preprocessor = ImputerDecorator(strategy='mean')(preprocessor)
preprocessor = NormalizationDecorator()(preprocessor)
preprocessor = CategoricalEncodingDecorator()(preprocessor)
preprocessor = OutlierHandlerDecorator()(preprocessor)
preprocessor = MappingDecorator()(preprocessor)

# Preprocess the data
preprocessed_data = preprocessor.preprocess(data)
print(preprocessed_data)