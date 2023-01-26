
import pandas as pd
from env import get_connection
import os

import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")



def get_mall():
    '''This function takes in the zillow dataframe specifically 
    and runs a query thorugh SQL to return the dataframe requested'''
    
    if os.path.isfile('mall_customers.csv'):
        
        return pd.read_csv('mall_customers.csv')
    
    else:
        
        url = get_connection('mall_customers')
        query = '''
        SELECT * FROM customers
        '''
        df = pd.read_sql(query, url)
    
        df.to_csv('mall_customers.csv')
    
    return df

def train_val_test(df):
    '''This function takes in a dataframe that has been prepared and splits it into train, validate, and test
    sections at 70/18/12 so it can be run through algorithms and tested for accuracy'''
    
    seed = 42
    
    train, val_test = train_test_split(df, train_size = 0.7, random_state = seed) 
        
    val, test = train_test_split(val_test, train_size = 0.6, random_state = seed)
    
    return train, val, test #these will be returned in the order in which they are sequenced

def scale_splits_mm(train, val, test, 
                    columns_to_scale = ['annual_income', 'spending_score'],
                    return_scaler = False):
    '''
    The purpose of this function is to accept, as input, the 
    train, validate, and test data splits, and returns the scaled versions of each.
    '''

    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    # make the thing
    scaler = MinMaxScaler()
    # fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    val_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(val[columns_to_scale]),
                                                  columns=val[columns_to_scale].columns.values).set_index([val.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, val_scaled, test_scaled
    else:
        return train_scaled, val_scaled, test_scaled
    
    return train_scaled, val_scaled, test_scaled

def mall_dummies(df):
    '''This function takes in the mall df specifically and drops the 
    listed columns, creates dummy variables for the selected columns, and renames
    the dummy variable columns to make them easier to read'''
    
    dummies = pd.get_dummies(df['gender'], drop_first = True)
    
    df = pd.concat([df, dummies], axis = 1)
    
    df.drop(columns = ['gender', 'Unnamed: 0'], inplace = True)
    
    return df



def handle_missing_values(df, prop_required_column, prop_required_row):
    
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    row_threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    
    return df



def null_counts(df):
    
    new_columns = ['name', 'num_rows_missing', 'pct_rows_missing']
    
    new_df = pd.DataFrame(columns=new_columns)
    
    for col in list(df.columns):
        num_missing = df[col].isna().sum()
        pct_missing = num_missing / df.shape[0]
        
        add_df = pd.DataFrame([{'name': col, 'num_rows_missing': num_missing,
                               'pct_rows_missing': pct_missing}])
        
        new_df = pd.concat([new_df, add_df], axis=0)
        
    new_df.set_index('name', inplace=True)
    
    return new_df