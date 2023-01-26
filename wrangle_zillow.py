import pandas as pd
import numpy as np
from env import get_connection
import os

def wrangle_zillow():
    '''
    This function's purpose is to pull a query from sequel ace and checks if
    there is a csv file for it, if not, it creates one
    '''
    if os.path.isfile('zillow.csv'):
        
        df = pd.read_csv('zillow.csv')

        df = prepare_zillow(df)

        df, var_fences = remove_outliers(df)

        return df
        
    
    else:
        
        url = get_connection('zillow')
        
        query = '''
        SELECT bathroomcnt AS bath,
        bedroomcnt AS bed,
        calculatedfinishedsquarefeet AS sqft,
        finishedsquarefeet12 AS fin_sqft,
        fips,
        fullbathcnt AS full_bath,
        lotsizesquarefeet AS lotsize,
        regionidzip AS zipcode,
        roomcnt AS rooms,
        yearbuilt,
        taxvaluedollarcnt 
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
        AND propertylandusetypeid LIKE 261;
        '''
        
        df = pd.read_sql(query, url)
        
        df.to_csv('zillow.csv')

        df, var_fences = remove_outliers(df)
        
        return df

def prepare_zillow(df):
    '''
    This function's purpose is to pull a query from sequel ace and checks if
    there is a csv file for it, if not, it creates one
    Then it prepares and cleans it to prepare it for use to train and model
    '''
        
    #Dropping Nulls as it only consisted of 1% of the original data
    df = df.dropna()
    
    #changing datatypes from floats to ints that had no decimal value
    df.bath, df.taxvaluedollarcnt, df.bed, df.yearbuilt, df.fips = (df.bath.astype(int), 
                                                                    df.taxvaluedollarcnt.astype(int), 
                                                                    df.bed.astype(int), 
                                                                    df.yearbuilt.astype(int),
                                                                    df.fips.astype(int))
    
    #dropping unnecessary column
    df.drop(columns = ('Unnamed: 0'), inplace = True)

    df = df[df['zipcode'] != 399675.0]

    df = df[df.taxvaluedollarcnt <= 8000000]

    df = df[df.sqft <= 10000]
    
    return df
     

def remove_outliers(df, k=1.5):
    '''
    This function is to remove the top 25% and bottom 25% of the data for each column.
    This removes the top and bottom 50% for every column to ensure all outliers are gone.
    '''
    a=[]
    b=[]
    fences=[a, b]
    features= []
    col_list = []
    i=0
    for col in df:
            new_df=np.where(df[col].nunique()>8, True, False)
            if new_df==True:
                if df[col].dtype == 'float' or df[col].dtype == 'int':
                    '''
                    for each feature find the first and third quartile
                    '''
                    q1, q3 = df[col].quantile([.25, .75])
                    '''
                    calculate inter quartile range
                    '''
                    iqr = q3 - q1
                    '''
                    calculate the upper and lower fence
                    '''
                    upper_fence = q3 + (k * iqr)
                    lower_fence = q1 - (k * iqr)
                    '''
                    appending the upper and lower fences to lists
                    '''
                    a.append(upper_fence)
                    b.append(lower_fence)
                    '''
                    appending the feature names to a list
                    '''
                    features.append(col)
                    '''
                    assigning the fences and feature names to a dataframe
                    '''
                    var_fences= pd.DataFrame(fences, columns=features, index=['upper_fence', 'lower_fence'])
                    col_list.append(col)
                else:
                    print(col)
                    print('column is not a float or int')
            else:
                print(f'{col} column ignored')
    '''
    for loop used to remove the data deemed unecessary
    '''
    for col in col_list:
        df = df[(df[col]<= a[i]) & (df[col]>= b[i])]
        i+=1
    return df, var_fences


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