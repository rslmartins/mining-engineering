import pandas as pd
import abc
import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st 

class function:
	def load_raw() -> pd.DataFrame:
		"""
		loads features needed for training dataset and target dataset to then perform a left outer join and return this single dataframe
		"""
		df_raw = pd.read_csv('md_raw_dataset.csv', sep = ";")
		df_raw.rename({'Unnamed: 0': 'index'}, axis=1, inplace=True)

		df_target = pd.read_csv('md_target_dataset.csv', sep = ";")
		df_target['groups'] = df_target['groups'].map(lambda x: float(x))

		df_raw = pd.merge(df_target, df_raw, on=['index', 'groups'], how='left')
		return df_raw
	    
	def make_column(df: pd.DataFrame, column1: str, column2: str, new_column: str) -> pd.DataFrame:
		"""
		With a given dataframe (df) creates a new column (new_column) with the difference between 2 date columns
		(column1 and column2) in terms of seconds then phase-out column1 and column2 
		"""
		df[column1] = df[column1].dropna().apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H:%M') if type(x)==str else x)
		df[column2] = df[column2].dropna().apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H:%M') if type(x)==str else x)

		df[new_column] = None

		for id in df.index:
			if df[column1][id] >= df[column2][id]: df[new_column][id] = (df[column1][id]-df[column2][id]).seconds
			if df[column1][id] < df[column2][id]: df[new_column][id] = (df[column2][id]-df[column1][id]).seconds
			
		df = df.drop(columns=[column1, column2])

		return df

	def perform_regressor(reg: abc.ABCMeta, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
		reg.fit(X_train, y_train)
		y_predicted = reg.predict(X_test)
		rmse = mean_squared_error(y_test, y_predicted, squared = True)

		pixel = 1/plt.rcParams['figure.dpi']
		fig, ax = plt.subplots(figsize=(800*pixel, 600*pixel))
		plt.scatter(range(len(y_test)),abs(y_test-y_predicted), label = 'Absolute error between target and predicted')
		plt.legend(loc="best")
		vals = ax.get_yticks()
		st.pyplot(fig)
		plt.show()

		st.success('$R^2$ of Gradient Boosting Regressor: '+str(reg.score(X_test, y_test)))
		st.success("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))

