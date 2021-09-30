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

#Packages for Regressor
from utils import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# Remove target column from dataframe before pre-processing (df) and after pre-processing (df_raw)
df_target = df['target']
df = df.drop(columns=['target'])
df_raw = df_raw.drop(columns=['target'])


st.title("Data Science for Mining Engineering")


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


# Perform max-min normalization
df=(df-df.min())/(df.max()-df.min())


st.write("Dataframe's Features Box-plot After Normalization.")
fig,ax = plt.subplots(figsize=(15,15))
sns.boxplot(data=df)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=90, fontsize=16)
plt.show()
st.pyplot(fig)


st.write("Dataframe's Features Variation Explained.")
# Instantiate PCA as instance
pca = PCA()

# Determine transformed features
X_train_pca = pca.fit_transform(df.fillna(0))

# Determine explained variance
exp_var_pca = pca.explained_variance_ratio_

# Cumulative sum of eigenvalues (to create step plot)
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

# Plotting
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
# Number of features to perform PCA
N=15

# Feature's names
columns_vector = ['element '+str(i) for i in range(1,N+1)]

# Instance PCA
pca = PCA(n_components=N)

# Perform the PCA with current DataFrame
df_PCA = pca.fit_transform(df.fillna(0))

# Create dataframe with new features
df_PCA = pd.DataFrame(data = df_PCA, columns = columns_vector)

# Display new dataframe
st.write(df_PCA)


# Split train(67%) and test(33%)
X_train, X_test, y_train, y_test = train_test_split(df_PCA, df_target,test_size=0.33, random_state=42)


st.write("Performance of Gradient Boost Regressor")
params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
function.perform_regressor(GradientBoostingRegressor(**params), X_train, y_train, X_test, y_test)


st.write("Performance of Support Vector Machine")
params = {'C': 1.0,
          'degree': 9,
          'kernel': 'rbf',
          'epsilon': 0.2}       
function.perform_regressor(SVR(**params), X_train, y_train, X_test, y_test)


st.write("Performance of Multi-Layer Percepetron")
params = {'activation': 'relu',
          'max_iter': 500,
          'alpha': 0.005,
          'solver': 'adam'}   
function.perform_regressor(MLPRegressor(**params), X_train, y_train, X_test, y_test)
