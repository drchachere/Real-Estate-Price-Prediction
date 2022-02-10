import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from kedro.extras.datasets.matplotlib import MatplotlibWriter
from kedro.extras.datasets.text import TextDataSet


def add_features_o(df: pd.DataFrame):
    df['TotalSF'] = df['1stFlrSF']+df['2ndFlrSF']+df['TotalBsmtSF']
    # df['OverallQCP'] = (df['OverallQual']+df['OverallCond'])/20
    df['OverallQCP'] = (df['OverallQual']*df['OverallCond'])/100
    df['TotalBath'] = df['FullBath']+df['BsmtFullBath']

    df_new = df

    #create two quant features for location (relevant to price)...median is better
    hood_names = df_new['Neighborhood'].unique()
    hood_avg_prices = []
    hood_median_prices = []
    for name in hood_names:
        df_temp = df_new[(df_new['Neighborhood']==name)]
        avg_price = int(df_temp['SalePrice'].mean())
        hood_avg_prices.append(avg_price)
        median_price = df_temp['SalePrice'].median()
        hood_median_prices.append(median_price)
    #     print("{} has an average home sale price of {}".format(name,avg_price))

    name_avg_price_dict = dict(zip(hood_names, hood_avg_prices))
    df_new['HoodAvg'] = df_new['Neighborhood'].map(lambda x: name_avg_price_dict[x])
    name_med_price_dict = dict(zip(hood_names, hood_median_prices))
    df_new['HoodMed'] = df_new['Neighborhood'].map(lambda x: name_med_price_dict[x])

    #create quant feature for Functional
    func_dict ={
        'Typ':7,
        'Min1':6,
        'Min2':5,
        'Mod':4,
        'Maj1':3,
        'Maj2':2,
        'Sev':1,
        'Sal':0  
    }
    df_new['FuncScore'] = df_new['Functional'].map(lambda x: func_dict[x])
    
    return df_new, name_avg_price_dict, name_med_price_dict

# adding average and median neighborhood prices to test data (without sales prices)
def add_features_to_test(df: pd.DataFrame, name_avg_price_dict, name_med_price_dict) -> pd.DataFrame:
    df['TotalSF'] = df['1stFlrSF']+df['2ndFlrSF']+df['TotalBsmtSF']
    # df['OverallQCP'] = (df['OverallQual']+df['OverallCond'])/20
    df['OverallQCP'] = (df['OverallQual']*df['OverallCond'])/100
    df['TotalBath'] = df['FullBath']+df['BsmtFullBath']

    df_new = df

    df_new['HoodAvg'] = df_new['Neighborhood'].map(lambda x: name_avg_price_dict[x])
    df_new['HoodMed'] = df_new['Neighborhood'].map(lambda x: name_med_price_dict[x])

    #create quant feature for Functional
    func_dict ={
        'Typ':7,
        'Min1':6,
        'Min2':5,
        'Mod':4,
        'Maj1':3,
        'Maj2':2,
        'Sev':1,
        'Sal':0  
    }
    df_new['FuncScore'] = df_new['Functional'].map(lambda x: func_dict[x] if not pd.isnull(x) else 7)
    
    return df_new

def split_data(cols, df_new: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    X = df_new[cols]
    y = df_new['SalePrice']
    return X, y

#creates scatter plots and saves them to data/04_feature
def plot_scatter(cols, X, y):
    for col in cols:
        plt.scatter(X[col], y, alpha=0.5)
        plt.title("Scatter Plot of {} vs. {}".format(col, y.name))
        single_plot_writer = MatplotlibWriter(filepath="data/04_feature/Scatter_Plot_of_{}_vs._{}.png".format(col, y.name))
        single_plot_writer.save(plt)
        plt.close()
    return

    

