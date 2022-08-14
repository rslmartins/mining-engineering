# Package for framework to operate on Web
import streamlit as st

# Packages for Basic Data Analysis
import pandas as pd
import matplotlib.pyplot as plt

# Packages for Regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Packages for Metrics
from sklearn.metrics import mean_squared_error

# Other Packages
import os
import joblib
import datetime


class Preprocessing:
    def load_raw() -> pd.DataFrame:
        """
        loads features needed for training dataset and target dataset to then perform a left outer join and return this single dataframe
        """
        df_raw = pd.read_csv('md_raw_dataset.csv', sep=";")
        df_raw.rename({'Unnamed: 0': 'index'}, axis=1, inplace=True)

        df_target = pd.read_csv('md_target_dataset.csv', sep=";")
        df_target['groups'] = df_target['groups'].map(lambda x: float(x))

        df_raw = pd.merge(df_target, df_raw, on=['index', 'groups'], how='left')
        return df_raw

    def make_column(df: pd.DataFrame, column1: str, column2: str, new_column: str) -> pd.DataFrame:
        """
        With a given dataframe (df) creates a new column (new_column) with the difference between 2 date columns
        (column1 and column2) in terms of seconds then phase-out column1 and column2
        """
        df[column1] = df[column1].dropna().apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M') if type(x)==str else x)
        df[column2] = df[column2].dropna().apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M') if type(x)==str else x)

        df[new_column] = None

        for id in df.index:
            if df[column1][id] >= df[column2][id]:
                df[new_column][id] = (df[column1][id]-df[column2][id]).seconds

            elif df[column1][id] < df[column2][id]:
                df[new_column][id] = (df[column2][id]-df[column1][id]).seconds
            
        df = df.drop(columns=[column1, column2])

        return df


class Training:
    def perform_regressor(method: str, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
        if method == "GBOOST":
            params = {'n_estimators': 500,
                      'max_depth': 4,
                      'min_samples_split': 5,
                      'learning_rate': 0.01,
                      'loss': 'ls'}
            model = GradientBoostingRegressor(**params)
        elif method == "SVM":
            params = {'C': 1.0,
                      'degree': 9,
                      'kernel': 'rbf',
                      'epsilon': 0.2}
            model = SVR(**params)
        elif method == "MLP":
            params = {'activation': 'relu',
                      'max_iter': 500,
                      'alpha': 0.005,
                      'solver': 'adam'}
            model = MLPRegressor(**params)

        model.fit(X_train, y_train)
        os.makedirs('./model', exist_ok=True)
        joblib.dump(model, './model/{0}.pkl'.format(method))


class Scoring:
    def metrics(method: str, X_test, y_test):
        model = joblib.load('./model/{0}.pkl'.format(method))
        y_predicted = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_predicted, squared=True)
        
        pixel = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(800*pixel, 600*pixel))
        plt.scatter(range(len(y_test)), abs(y_test-y_predicted), label='Absolute error between actual target and predicted target')
        plt.legend(loc="best")
        st.pyplot(fig)
        plt.show()

        st.success('$R^2$ of Gradient Boosting Regressor: '+str(model.score(X_test, y_test)))
        st.success("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
