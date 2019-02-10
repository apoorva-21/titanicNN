import numpy as np
import pandas as pd
import numpy as np
import re
import random

TRAIN_FILE = './data/train.csv'
TEST_FILE = './data/test.csv'
TEST_LABELS_FILE = './data/gender_submission.csv'
VALIDATION_SPLIT = 0.1
ALPHA = 6e-4
N_EPOCHS = 400

def process_gender(X):
	genders = list(set(X.T[1]))
	#0=male
	for i in range(X.shape[0]):
		X[i][1] = genders.index(X[i][1])
	return X

def normalize(matrix):
	matrix = np.array(matrix.T, dtype= np.float32)
	for i in range(matrix.shape[0]): #i.e. for each attribute in the dataset
		matrix[i] = (matrix[i] - np.average(matrix[i]))/(np.max(matrix[i]) - np.min(matrix[i]) + 1)
	return matrix.T

def sigmoid(x):
	expo = np.exp(-x)
	z = 1. / (1. + expo)
	return z

def relu(x):
	z = x.copy()
	z[z < 0] = 0
	return z

df = pd.read_csv(TRAIN_FILE)
df = df.drop(['PassengerId','Name','Cabin', 'Ticket', 'Parch', 'Fare', 'Embarked'], axis = 1)
df = df.dropna()

y = df['Survived'].values
df = df.drop(['Survived'], axis = 1)
X = df.values

X = process_gender(X)
X = normalize(X)

X = X.T
y = np.reshape(y, (1, y.shape[0]))
n_inputs = X.shape[0]

n_HL1 = 3
n_HL2 = 2
n_Out = 1

W_HL1 = np.random.rand(n_HL1, n_inputs) / np.sqrt(n_inputs)
b_HL1 = np.zeros((n_HL1, 1)) / np.sqrt(n_inputs)

W_HL2 = np.random.rand(n_HL2, n_HL1) / np.sqrt(n_HL1)
b_HL2 = np.zeros((n_HL2, 1)) / np.sqrt(n_HL1)

W_Out = np.random.rand(n_Out, n_HL2) / np.sqrt(n_HL2)
b_Out = np.zeros((n_Out, 1)) / np.sqrt(n_HL2)


def forward_prop(X_in):
	global W_HL1, b_HL1, W_HL2, b_HL2,W_Out, b_Out

	z_1 = np.matmul(W_HL1, X_in) + b_HL1
	a_1 = relu(z_1)

	z_2 = np.matmul(W_HL2, a_1) + b_HL2
	a_2 = relu(z_2)

	z_Out = np.matmul(W_Out, a_2) + b_Out
	a_Out = sigmoid(z_Out)

	return a_1, a_2, a_Out

def backward_prop(X_in, Y_true, a_1, a_2, a_Out):

	dz_Out = a_Out - Y_true
	dW_Out = np.matmul(dz_Out, a_2.T)
	# print dW_Out
	# print dW_Out.shape
	db_Out = np.reshape(np.sum(dz_Out, axis = 1), (b_Out.shape[0], b_Out.shape[1]))

	da_2 = np.matmul(W_Out.T,dz_Out)
	dz_2 =  da_2.copy()
	dz_2[a_2 == 0] = 0
	dW_HL2 = np.matmul(dz_2, a_1.T)
	db_HL2 = np.reshape(np.sum(dz_2, axis = 1),(b_HL2.shape[0], b_HL2.shape[1]))
	
	da_1 = np.matmul(W_HL2.T, dz_2)
	# dz_1 =  da_1.copy()
	# dz_1[a_1 == 0] = 0
	dz_1 = da_1 * (a_1) * (1. - a_1)
	dW_HL1 = np.matmul(dz_1, X_in.T)
	db_HL1 = np.reshape(np.sum(dz_1, axis = 1),(b_HL1.shape[0], b_HL1.shape[1]))

	return dW_HL1, db_HL1, dW_HL2, db_HL2, dW_Out, db_Out

print '\n\nInitial Weights : '
print 'W_HL1 : ', W_HL1
print 'b_HL1 : ', b_HL1
print 'W_HL2 : ', W_HL2
print 'b_HL2 : ', b_HL2
print 'W_Out : ', W_Out
print 'b_Out : ', b_Out

print '\n\n'
print '#' * 50
for i in range(N_EPOCHS):
	

	#get activations::
	a_1, a_2,a_Out = forward_prop(X)

	#get gradients::
	dW_HL1, db_HL1,dW_HL2, db_HL2, dW_Out, db_Out = backward_prop(X, y, a_1, a_2,a_Out)

	#update params::
	W_HL1 -= ALPHA * dW_HL1
	b_HL1 -= ALPHA * db_HL1
	W_HL2 -= ALPHA * dW_HL2
	b_HL2 -= ALPHA * db_HL2
	W_Out -= ALPHA * dW_Out
	b_Out -= ALPHA * db_Out

print '\n\nTrained Weights : '
print 'W_HL1 : ', W_HL1
print 'b_HL1 : ', b_HL1
print 'W_HL2 : ', W_HL2
print 'b_HL2 : ', b_HL2
print 'W_Out : ', W_Out
print 'b_Out : ', b_Out

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
X_test = normalize(X_test)
X_test= X_test.T
# X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

y_test = y_test.values

#inference on train set
a_1_train,a_2_train, a_Out_train = forward_prop(X)
# print a_Out_test
# exit()
correct = 0
print 'Accuracy on the train set : '
for i in range(X.shape[1]):
	output = 0
	# print a_Out_train[0,i], y[0,i]
	if a_Out_train[0,i] > 0.5:
		output = 1
	if output == y[0,i]:
		correct += 1

print correct * 1. / X.shape[1]


#inference on test set
a_1_test,a_2_test, a_Out_test = forward_prop(X_test)
# print a_Out_test
# exit()
correct = 0
print 'Accuracy on the test set : '
for i in range(X_test.shape[1]):
	output = 0
	if a_Out_test[0,i] > 0.5:
		output = 1
	if output == y_test[i]:
		correct += 1
	# print a_Out_test[0,i], y_test[i]

print correct * 1. / X_test.shape[1]
