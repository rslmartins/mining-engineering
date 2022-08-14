# Framework to operate on Web
import streamlit as st

# Basic Data Analysis Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings

# Package to pass argument
import argparse

# Package to perform PCA
from sklearn.decomposition import PCA

# Packages to manipulate and perform metrics
from sklearn.model_selection import train_test_split

# Packages for the steps
from pre_processing import Preparing as cl
from utils import Training as tr
from utils import Scoring as sc

# Load argument
parser = argparse.ArgumentParser(description="Runner", add_help=True)
parser.add_argument(
    "--step",
    metavar="stepName",
    help="[load, train]",
    type=str,
    required=True,
)
args = parser.parse_args()

# Ignore Warnings
warnings.filterwarnings("ignore")

st.title("Data Science for Mining Engineering")

# Load Data
df, df_raw, df_target = cl.data_prepared()

st.write("Dataframe before pre-processing.")
st.write(df_raw.head())


st.write("Dataframe after pre-processing.")
st.write(df.head())


st.write("Dataframe's Features Correlation.")
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
st.pyplot(fig)


st.write("Dataframe's Features Box-plot.")
fig, ax = plt.subplots(figsize=(15, 15))
sns.boxplot(data=df)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=90, fontsize=16)
plt.show()
st.pyplot(fig)


# Perform max-min normalization
df=(df-df.min())/(df.max()-df.min())


st.write("Dataframe's Features Box-plot After Normalization.")
fig, ax = plt.subplots(figsize=(15, 15))
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
fig, ax = plt.subplots(figsize=(15, 15))
plt.bar(range(1, len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
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
columns_vector = ['element '+str(i) for i in range(1, N+1)]

# Instance PCA
pca = PCA(n_components=N)

# Perform the PCA with current DataFrame
df_PCA = pca.fit_transform(df.fillna(0))

# Create dataframe with new features
df_PCA = pd.DataFrame(data=df_PCA, columns=columns_vector)

# Display new dataframe
st.write(df_PCA)

# Split train(67%) and test(33%)
X_train, X_test, y_train, y_test = train_test_split(df_PCA, df_target, test_size=0.33, random_state=42)

st.write("Performance of Gradient Boost Regressor")
if args.step == "train":
    tr.perform_regressor("GBOOST", X_train, y_train, X_test, y_test)
sc.metrics("GBOOST", X_test, y_test)

st.write("Performance of Support Vector Machine")
if args.step == "train":
    tr.perform_regressor("SVM", X_train, y_train, X_test, y_test)
sc.metrics("SVM", X_test, y_test)

st.write("Performance of Multi-Layer Percepetron")
if args.step == "train":
    tr.perform_regressor("MLP", X_train, y_train, X_test, y_test)
sc.metrics("MLP", X_test, y_test)
