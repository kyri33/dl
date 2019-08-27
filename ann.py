import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('datasets/Churn_Modelling.csv')
# Seperate irrelevant attributes like row number
# Seperate answers

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
X_display = dataset.iloc[:, 3:13]
print(X_display.head())

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# The country has more than 2 so needs to be one hot encoded
# Specify which index needs to be encoded
ct_pipe = ColumnTransformer([("onehotenc", OneHotEncoder(), [1])], remainder="passthrough")
X = ct_pipe.fit_transform(X)
# Remove the extra one hot column because it is inferred ?

# Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Scale the training data and transform it to the test data
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
X_train = stdscaler.fit_transform(X_train)
X_test = stdscaler.transform(X_test)
print(X_test.shape)
# Now we build the neural network
import keras
from keras.models import Sequential
from keras.layers import Dense
# Network will have 12 input nodes
# 6 nodes in hidden layer
# 1 node in output layer
classifier = Sequential()
# Technically we are adding the second layer.
# First layer automatically gets added
classifier.add(Dense(output_dim = 6,init = 'uniform', 
    activation = 'relu', input_dim = 11))

# Second hidden layer also 6 nodes
# It doesn't need an input dim because it is inferred
# since it is the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', 
    activation = 'relu'))

# Adding output layer (different activation)
classifier.add(Dense(output_dim = 1, init = 'uniform', 
    activation = 'sigmoid'))

# Compiling classifier
# First arg is gradient descent algo we use stochastic gradient descent 'adam'
# The next paramter is loss function,
# Because our result is binary we are using a binary logarithmic loss function
# Use categorical_crossentropy for categories not binary
# Last argument is what metrics to evalueate model with
# All we care about is accuracy
classifier.compile(optimizer = 'adam', 
    loss = 'binary_crossentropy', metrics = ['accuracy'])

