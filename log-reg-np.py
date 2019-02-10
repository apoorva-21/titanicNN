import numpy as np
import pandas as pd
import numpy as np
import re
import random

TRAIN_FILE = './data/train.csv'
TEST_FILE = './data/test.csv'
VALIDATION_SPLIT = 0.01

df = pd.read_csv(TRAIN_FILE)

#remove unnecessary features::
df.drop(['PassengerId','Name','Cabin', 'Ticket', 'Parch', 'Fare', 'Embarked'], axis=1, inplace=True)

#remove NaN's
df = df.dropna(axis= 0, how = 'any')
y = df['Survived'].values
df.drop(['Survived'], axis = 1, inplace = True)
X = df.values

def process_gender(X):
	genders = list(set(X.T[1]))
	#0=male
	for i in range(X.shape[0]):
		X[i][1] = genders.index(X[i][1])
	return X


def feature_normalize(X):
	#feature_normalized = (feature - mean)/feature_range
	X_normed = np.zeros((X.shape[0], X.shape[1]))
	for i in range(X.shape[1]):
		feature_mean = np.mean(X[:,i])
		feature_range = np.max(X[:,i]) - np.min(X[:,i])
		X_normed[:,i] = (X[:,i] - feature_mean) * 1./ feature_range
	return X_normed

def activate(input):
	global epsilon
	output = 1.0 / (1 + np.exp(-input+epsilon))
	return output

X = process_gender(X)
X = feature_normalize(X)

# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# y = y[indices]

# X = X.T
y = np.reshape(y, (y.shape[0], 1))
X = np.hstack([np.ones((X.shape[0], 1)), X])
print 'First 10 entries of cleaned dataset : '
print X[:10]
num_val = int(VALIDATION_SPLIT * X.shape[0])
X_train = X#[:-1 * num_val]
y_train = y#[:-1 * num_val]

X_val = X[-1 * num_val:]
y_val = y[-1 * num_val:]

n_attr = X_train.shape[1]
n_samples = X_train.shape[0]

ALPHA = 0.01
epsilon = 1e-15
N_EPS = 20
W = np.zeros((n_attr))

# exit()
'''
for k in range(N_EPS):
	for i in range(X_train.shape[1]):
		output = activate(np.dot(W, X_train[:,i]))
		print W
		# loss = np.sum(-1 * y_train * np.log(output + epsilon) - (1 - y_train) * np.log(1 - output + epsilon)) / (n_samples - num_val)
		# print loss
		for j in range(W.shape[1]):
			gradient = (y_train[0,i] - output) * X_train[j, i]
			W[0,j] += ALPHA * gradient
		print '.'
		'''
for k in range(N_EPS):
	for i in range(X_train.shape[0]):
		print X_train[i].shape
		exit()
		h = np.dot(X_train[i], W)
		output = activate(h)
		# loss = np.sum(-1 * y_train[i] * np.log(output+epsilon) - (1 - y_train[i]) * np.log(1 - output + epsilon)) 
		gradient = X_train[i] * (y_train[i] - output)	#np.dot(X_train.T,(y_train - output))
		W = W + ALPHA * gradient

#load in the test set::

X_test = pd.read_csv(TEST_FILE)
y_test = pd.read_csv('./data/gender_submission.csv')[['Survived']]
#remove unnecessary features::
X_test.drop(['PassengerId','Name','Cabin', 'Ticket', 'Parch', 'Fare', 'Embarked'], axis=1, inplace=True)

age = X_test[['Age']].values
to_be_deleted = []
for i in range(len(age)):
	if np.isnan(age[i]):
		to_be_deleted.append(i)
X_test.drop(X_test.index[to_be_deleted], axis=0, inplace=True)
y_test.drop(y_test.index[to_be_deleted], axis=0, inplace=True)

X_test = X_test.values
X_test = process_gender(X_test)
X_test = feature_normalize(X_test)
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

y_test = y_test.values

correct = 0
print 'Accuracy on the test set : '
for i in range(X_test.shape[0]):
	output = activate(np.dot(X_test[i], W))
	if output > 0.5:
		output = 1
	elif output <= 0.5:
		output = 0
	if output == y_test[i]:
		correct += 1
print correct * 1. / X_test.shape[0]
