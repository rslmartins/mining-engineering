#Framework to operate on Web
import streamlit as st

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Load Data
from pre_processing import df, df_raw

#Basic Data Analysis Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Package to perform PCA
from sklearn.decomposition import PCA

#Packages to manipulate and perform metrics
from sklearn.model_selection import train_test_split
from sklearn import model_selection

#Packages for Regressor
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
#from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

st.title("Data Science for Mining Engineering")
df_target = df['target']
df = df.drop(columns=['target'])
df_raw = df_raw.drop(columns=['target'])

st.write("Dataframe before pre-processing.")
st.write(df_raw.head())

st.write("Dataframe after pre-processing.")
st.write(df.head())


st.write("Dataframe's Features Correlation.")
fig,ax = plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
st.pyplot(fig)


st.write("Dataframe's Features Box-plot.")
fig,ax = plt.subplots(figsize=(15,15))
sns.boxplot(data=df)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=90, fontsize=16)
plt.show()
st.pyplot(fig)


st.write("Dataframe's Features Box-plot After Normalization.")

df=(df-df.min())/(df.max()-df.min())

fig,ax = plt.subplots(figsize=(15,15))
sns.boxplot(data=df)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=90, fontsize=16)
plt.show()
st.pyplot(fig)

st.write("Dataframe's Features Variation Explained.")
pca = PCA()

X_train_pca = pca.fit_transform(df.fillna(0))
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

fig,ax = plt.subplots(figsize=(15,15))
plt.bar(range(1,len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1,len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
st.pyplot(fig)
plt.show()


st.write("Dataframe's After PCA.")
N=15
columns_vector = ['element '+str(i) for i in range(1,N+1)]
pca = PCA(n_components=N)
df_PCA = pca.fit_transform(df.fillna(0))
df_PCA = pd.DataFrame(data = df_PCA
             , columns = columns_vector)
st.write(df_PCA)


st.write("Performance of Gradient Boost Regressor")
X_train, X_test, y_train, y_test = train_test_split(df_PCA, df_target,test_size=0.33, random_state=42)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)
rmse = mean_squared_error(y_test, y_predicted, squared = True)

pixel = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(figsize=(800*pixel, 600*pixel))
#plt.plot(range(len(y_test)),y_test, color ='red', label = 'Target test database')
plt.scatter(range(len(y_test)),abs(y_test-y_predicted), label = 'Absolute error between target and predicted')
#plt.scatter(range(len(y_test)),y_predicted, label = 'Predicted')
plt.legend(loc="best")
#ax.set_xlabel('Temperature (Kelvin)')
#ax.set_ylabel('Relative Error')
vals = ax.get_yticks()
#ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
st.pyplot(fig)
plt.show()


st.success('$R^2$ of Gradient Boosting Regressor: '+str(reg.score(X_test, y_test)))
st.success("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))


st.write("Support Vector Machine")
params = {'C': 1.0,
          'degree': 9,
          'kernel': 'rbf',
          'epsilon': 0.2}       

reg = SVR(**params)
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)
rmse = mean_squared_error(y_test, y_predicted, squared = True)


pixel = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(figsize=(800*pixel, 600*pixel))
#plt.plot(range(len(y_test)),y_test, color ='red', label = 'Target test database')
plt.scatter(range(len(y_test)),abs(y_test-y_predicted), label = 'Absolute error between target and predicted')
#plt.scatter(range(len(y_test)),y_predicted, label = 'Predicted')
plt.legend(loc="best")
#ax.set_xlabel('Temperature (Kelvin)')
#ax.set_ylabel('Relative Error')
vals = ax.get_yticks()
#ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
st.pyplot(fig)
plt.grid(True)
plt.show()


st.success('$R^2$ of Gradient Boosting Regressor: '+str(reg.score(X_test, y_test)))
st.success("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))


st.write("Multi-Layer Percepetron")
params = {'activation': 'relu',
          'max_iter': 500,
          'alpha': 0.005,
          'solver': 'adam'}   

reg = MLPRegressor(**params)
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)
rmse = mean_squared_error(y_test, y_predicted, squared = True)

pixel = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(figsize=(800*pixel, 600*pixel))
#plt.plot(range(len(y_test)),y_test, color ='red', label = 'Target test database')
plt.scatter(range(len(y_test)),abs(y_test-y_predicted), label = 'Absolute error between target and predicted')
#plt.scatter(range(len(y_test)),y_predicted, label = 'Predicted')
plt.legend(loc="best")
#ax.set_xlabel('Temperature (Kelvin)')
#ax.set_ylabel('Relative Error')
vals = ax.get_yticks()
#ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
st.pyplot(fig)
plt.grid(True)
plt.show()


st.success('$R^2$ of Gradient Boosting Regressor: '+str(reg.score(X_test, y_test)))
st.success("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
