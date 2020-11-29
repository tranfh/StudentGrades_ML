import tensorflow 
import keras
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Label
predict = "G3"

#Define our attributes
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Training Model
"""
best = 0
for _ in range(1000):
    # Split 10% of our data for testing 
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    #creates best fit line
    linear.fit(x_train, y_train)
    #Calculate accuracy of our model
    acc = linear.score(x_test, y_test)
    print("Model Accuracy: ", acc)
    print("Current Best Accuracy: ", best)
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# Load Pickle Model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

# Get Grade Predictions 
correct = 0
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    if int(predictions[x]) == y_test[x]:
        correct += 1

print(correct, "/", len(predictions))

# Plot the data 
p = "G1"
style.use("ggplot")
# G1 is x G3 is y
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()