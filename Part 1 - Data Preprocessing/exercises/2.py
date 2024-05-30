# Importing the necessary libraries
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv("pima-indians-diabetes.csv")

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull()

# Print the number of missing entries in each column
print(missing_data.sum(axis=0).values)

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Fit the imputer on the DataFrame
X = dataset.iloc[:, :-1].values
imputer.fit(X)

# Apply the transform to the DataFrame
X = imputer.transform(X)

#Print your updated matrix of features
print(X)