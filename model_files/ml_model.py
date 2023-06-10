# Here we will write the functions that we wrote
# Importing a few general use case libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Functions

# preprocess the Origin column in data
def preprocess_origin_cols(df):
    df['Origin'] = df['Origin'].map({1:'India', 2:'USA', 3:'Germany'})
    return df

# creating custom attribute adder class
# column indexes of Acceleration, Horsepower, Cylinders
acc_ix, hpower_ix, cyl_ix = 4, 2, 0

class CustomAttrAdder(BaseEstimator, TransformerMixin):    # inherts BaseEstimator and TransformerMixin class, they work on ndarrays
    def __init__(self, acc_on_power=True):                 # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self                                        # nothing else to do
    def transform(self, X):                                # X is a 2D array
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]      # concatenates arrays
        return np.c_[X, acc_on_cyl]
    

def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
    '''
    numerics = ['float64', 'int64']
    
    num_attrs = data.select_dtypes(include=numerics)
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline


def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    numerical and categorical data
    
    Argument:
        data: original dataframe
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ['Origin']
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    print(list(num_attrs))
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, list(num_attrs)),
        ('cat', OneHotEncoder(), cat_attrs),
        ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


def predict_mpg(config, model):
    '''
    Arguments:
        config: Vehicle configuration in dictionary or dataframe
        model: The trained model used for prediction
    Returns:
        y_pred: mpg predicted values
    '''
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
        
    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df)
    # print(prepared_df)
    y_pred = model.predict(prepared_df)
    return y_pred