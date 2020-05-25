# -*- coding: utf-8 -*-

# Importing all libs
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.utils import shuffle
from sklearn import datasets
import re
import time
import matplotlib.pyplot as plt

# --- Functions ----
reg_strength = 10000 # regularization variable

# Tunning the learning rate results:
# 1e-6 = ASGD aprox. 4 sec
# 1e-5 = ASGD aprox. 0.07 sec
# 1e-4 = ASGD aprox. 44 sec
learning_rate = 1e-5
costs = []

def compute_cost(W, X, Y):
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0 # max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    
    # calculate cost 
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    if type(Y_batch) == np.float64 or type(Y_batch) == np.int64 or type(Y_batch) == int:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)
    return dw

def gd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01
    for epoch in range(1, max_epochs):
        X = features
        Y = outputs
        ascent = calculate_cost_gradient(weights, X, Y)
        weights = weights - (learning_rate * ascent)
        
        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            costs.append(cost)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion - percentage of previous cost is grater that the absolute difference between the current and the previous cost
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return (weights, epoch)
            prev_cost = cost
            nth += 1
            
    return (weights, max_epochs)

def sgd(features, outputs, averaged=True):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01
    all_weights = np.zeros(features.shape[1])
    for epoch in range(1, max_epochs): 
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
            if averaged:
                all_weights = np.vstack([all_weights, weights])
        
        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            costs.append(cost)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion - percentage of previous cost is grater that the absolute difference between the current and the previous cost
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                print("ALL WEIGHTS")
                print(all_weights)
                if averaged:
                    return ( (all_weights.sum(axis=0) / (all_weights.shape[0] - 1)), epoch)
                return (weights, epoch)
            prev_cost = cost
            nth += 1
            
    return (weights, max_epochs)


# ---- Fitting the data ----
    
data = pd.read_csv('./breast-cancer-scale.txt', header=None, delimiter=' ')
numpy_data = data.to_numpy()

method_stats = {
    "epochs": 0,
    "elapsed_time": 0,
    "accuracy": 0
}

methods = [{}, {}, {}]

# Remove Nan column breast-cancer normal
# numpy_data = np.delete(numpy_data, 1, 1)

# Remove Nan column breast-cancer scaled
numpy_data = np.delete(numpy_data, 11, 1)

print(numpy_data)

for i in range(0, numpy_data.shape[0]):
  if numpy_data[i][0] == 2:
    numpy_data[i][0] = -1
  else:
    numpy_data[i][0] = 1
  for j in range(1, numpy_data.shape[1]):
    #print(numpy_data[i][j])
    numpy_data[i][j] = float(re.sub('\d+\:', '', numpy_data[i][j]))

# Separate Labels/Features
Y = numpy_data[:, 0]
X = numpy_data[:, 1:]

# Normalize Data breast cancer normal
X_normalized = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X_normalized)

# Normalize Data breast cancer scaled
X = pd.DataFrame(X)

print(X)

# Insert a new column for intercept
X.insert(loc=len(X.columns), column='intercept', value=1)

# Splitting
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

# ---- Simple Gradient Descent ----
# aprox. 5000 epochs

# Train
print("SIMPLE GRADIENT DESCENT")
print("training started...")
start_time = time.time()
W, methods[0]["epochs"] = gd(X_train.to_numpy(), y_train)
print("training finished in {} sec".format(time.time() - start_time))
print("weights are: {}".format(W))

# Test
y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = int (np.sign(np.dot(W, X_test.to_numpy()[i]))) #model
    y_test_predicted = np.append(y_test_predicted, yp)


# ---- Accuracy ----
    
# Convert to int64
y_test = y_test.astype(np.int64)
y_test_predicted = y_test_predicted.astype(np.int64)

methods[0]["accuracy"] = accuracy_score(y_test, y_test_predicted.round())
print("Accuracy: {}".format(methods[0]["accuracy"], normalize=False))
print("Recall: {}".format(recall_score(y_test, y_test_predicted)))
methods[0]["elapsed_time"] = time.time() - start_time

print(classification_report(y_test, y_test_predicted)) 

# ------------------------


# ---- Simple Stochastic Gradient Descent ----
# aprox. 32-64 epochs

# Train
print("SIMPLE STOCHASTIC GRADIENT DESCENT")
print("training started...")
start_time = time.time()
W, methods[1]["epochs"] = sgd(X_train.to_numpy(), y_train, False)
print("training finished in {} sec".format(time.time() - start_time))
print("weights are: {}".format(W))

# Test
y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = int (np.sign(np.dot(W, X_test.to_numpy()[i]))) #model
    y_test_predicted = np.append(y_test_predicted, yp)


# ---- Accuracy ----
    
# Convert to int64
y_test = y_test.astype(np.int64)
y_test_predicted = y_test_predicted.astype(np.int64)

methods[1]["accuracy"] = accuracy_score(y_test, y_test_predicted.round())
print("Accuracy: {}".format(methods[1]["accuracy"], normalize=False))
print("Recall: {}".format(recall_score(y_test, y_test_predicted)))
methods[1]["elapsed_time"] = time.time() - start_time

print(classification_report(y_test, y_test_predicted))

# ------------------------

# ---- Averaged Stochastic Gradient Descent ----
# aprox. 16-32 epochs

# Train
print("AVERAGED STOCHASTIC GRADIENT DESCENT")
print("training started...")
start_time = time.time()
W, methods[2]["epochs"] = sgd(X_train.to_numpy(), y_train, True)
print("training finished in {} sec".format(time.time() - start_time))
print("weights are: {}".format(W))

# Test
y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = int (np.sign(np.dot(W, X_test.to_numpy()[i]))) #model
    y_test_predicted = np.append(y_test_predicted, yp)


# ---- Accuracy ----
    
# Convert to int64
y_test = y_test.astype(np.int64)
y_test_predicted = y_test_predicted.astype(np.int64)

methods[2]["accuracy"] = accuracy_score(y_test, y_test_predicted.round())
print("Accuracy: {}".format(methods[2]["accuracy"], normalize=False))
print("Recall: {}".format(recall_score(y_test, y_test_predicted)))
methods[2]["elapsed_time"] = time.time() - start_time

print(classification_report(y_test, y_test_predicted))

# ------------------------


# ----- Plots ------

print("EPOCHS:\nSG = {}\nSGD = {}\nASGD = {}".format(methods[0]["epochs"], methods[1]["epochs"], methods[2]["epochs"]))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['SG', 'SGD', 'ASGD']
ax.bar(names, [item["epochs"] for item in methods])
plt.show()

print("ACCURACY:\nSG = {}\nSGD = {}\nASGD = {}".format(methods[0]["accuracy"], methods[1]["accuracy"], methods[2]["accuracy"]))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['SG', 'SGD', 'ASGD']
ax.bar(names, [item["accuracy"] for item in methods])
plt.show()

print("ELAPSED TIME:\nSG = {}\nSGD = {}\nASGD = {}".format(methods[0]["elapsed_time"], methods[1]["elapsed_time"], methods[2]["elapsed_time"]))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['SG', 'SGD', 'ASGD']
ax.bar(names, [item["elapsed_time"] for item in methods])
plt.show()
