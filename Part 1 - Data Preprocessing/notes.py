import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# DataFrame object
dataset = pd.read_csv("Data.csv")
# Indepedent variables (numpy 2d arrays)
X = dataset.iloc[:, :-1].values
# Dependent variables
y = dataset.iloc[:, -1].values

# Tricks with DataFrame objects

# Keeps all except the last column
# X = dataset.drop([-1], axis=1)

# Get column by name
# y = dataset["Purchased"]

# Print number of missing data in each column (axis=1 for rows)
# print(dataset.isnull().sum(axis=0).values) 

# Dealing with missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# Encoding categorical data (independent variables)
# Transforms categorical data into vectors <1, 0, 0>, <0, 1, 0>, <0, 0, 1>... for each category
# In the tuple:
# 'encoder' is the type of the transformer
# OneHotEncoder is the class of the specific transformer
# [0] is the index of the column to transform
# - Can include strings for DataFrame objects (name of the column in the .csv file)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding binary data (dependent variable)
le = LabelEncoder()
y = le.fit_transform(y)

# print("%s\n\n%s" % (X, y))

# train_test_split from sklearn.model_selection splits the data into training
# and test sets
# test_size is % of data split into testing group
# 80/20 split is recommended
# random_state is the splitting seed; setting a seed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("X Training: %s\n\nX Testing: %s\ny Training: %s\ny Testing: %s" % (X_train, X_test, y_train, y_test))

# Feature Scaling
scaler = StandardScaler()
# Note that transform methods can take a dataframe object but will return a numpy array
X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scaler.transform(X_test[:, 3:]) # Make sure to use the fit from the training set to transform the test set!!!

print(X_test)