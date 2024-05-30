# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
dataset = pd.read_csv("titanic.csv")

# Identify the categorical data
categories = ["Sex", "Embarked", "Pclass"]

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categories)], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(dataset)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
y = le.fit_transform(dataset["Survived"])

# Print the updated matrix of features and the dependent variable vector
print("%s\n%s" % (X, y))
