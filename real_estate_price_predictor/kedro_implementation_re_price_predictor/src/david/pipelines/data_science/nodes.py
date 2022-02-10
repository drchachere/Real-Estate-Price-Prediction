import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

#creates scatter plots and saves them to data/04_feature
# def plot_scatter(X, y):
#     plt.scatter(X, y, alpha=0.5)
#     plt.title("Scatter Plot of {} vs. {}".format(X.name, y.name))
#     single_plot_writer = MatplotlibWriter(filepath="data/04_feature/Scatter_Plot_of_{}_vs._{}.png".format(X.name, y.name))
#     plt.close()
#     single_plot_writer.save(plt)

#standardize features of interest
def standardize_foi(cols, df_new: pd.DataFrame) -> pd.DataFrame:
    scaler = preprocessing.MinMaxScaler()
    for feature in cols:
        feature_mat = df_new[feature].values.reshape(-1,1)
        df_new[feature] = scaler.fit_transform(feature_mat)
    for col in cols:
        df_new[col] = df_new[col].map(lambda x: x if not pd.isnull(x) else df_new[col].median())
    return df_new

# train test split function
def tts(cols, df_new: pd.DataFrame) -> [pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    X = df_new[cols]
    #print(X.shape)
    y = df_new['SalePrice']
    #print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    return X_train, X_test, y_train, y_test

def find_model_perf(X_train, y_train, X_test, y_test):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train) #
    model = LinearRegression()
    model.fit(X_poly, y_train) #
    y_results = model.predict(X_poly) #
    rmse = np.sqrt(mean_squared_error(y_train,y_results))
    r2 = r2_score(y_train,y_results)
    # print('Train RMSE: ', rmse)
    # print('Train R2: ', r2)
    line1 = 'Train RMSE: ' + str(rmse) + "\n"
    line2 = 'Train R2: ' + str(r2)+ "\n" + "\n"
    
    X_test_poly = poly.transform(X_test)
    y_test_results = model.predict(X_test_poly)
    rmse = np.sqrt(mean_squared_error(y_test,y_test_results))
    r2 = r2_score(y_test,y_test_results)
    # print('Test RMSE: ', rmse)
    # print('Test R2: ', r2)
    line3 = 'Train RMSE: ' + str(rmse) + "\n"
    line4 = 'Train R2: ' + str(r2)
    string_to_write = line1+line2+line3+line4
    data_set = TextDataSet(filepath="accuracy_report.txt")
    data_set.save(string_to_write)
    reloaded = data_set.load()
    assert string_to_write == reloaded
    
    return