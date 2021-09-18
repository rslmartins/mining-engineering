import streamlit as st

import warnings
warnings.filterwarnings("ignore")


from pre_processing import df,df_raw

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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
# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
# Create the visualization plot

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

st.write("Performance of Classifiers Using k-Fold Cross-Validation.")
X_train, X_test, y_train, y_test = train_test_split(df_PCA, df_target, test_size=1/3,stratify = df_target, random_state=42)
seed=42
models = []
scoring='accuracy'
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors Algorithm', KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Support Vector Machine', SVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: $\mu$:%f $\;$  $\sigma$:%f" % (name, cv_results.mean(), cv_results.std())
    st.success(msg)
