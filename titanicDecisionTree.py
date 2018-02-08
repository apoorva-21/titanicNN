from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import re
import random
from sklearn.tree import export_graphviz

TRAIN_FILE = './data/train.csv'
TEST_FILE = './data/test.csv'
VALIDATION_SPLIT = 0.1

df = pd.read_csv(TRAIN_FILE)
df = df.drop(['PassengerId','Name','Cabin'], axis = 1)
y = df.as_matrix(['Survived'])
df = df.drop(['Survived'], axis = 1)

df = df.dropna(axis= 0, how = 'any')

dirtyFeatures = df.as_matrix()

genders = list(set(dirtyFeatures.T[1]))
#0=male
for i in range(dirtyFeatures.shape[0]):
	dirtyFeatures[i][1] = genders.index(dirtyFeatures[i][1])

embarkLocs = list(set(dirtyFeatures.T[7]))
for i in range(dirtyFeatures.shape[0]):
	dirtyFeatures[i][7] = embarkLocs.index(dirtyFeatures[i][7])

reInt = re.compile(r'^[-+]?([1-9]\d*|0)$')

for i in range(dirtyFeatures.shape[0]):
	ticketNoList = dirtyFeatures[i][5].split(' ')
	ticketNo = ticketNoList[len(ticketNoList) - 1]
	if(reInt.match(ticketNo)):
		dirtyFeatures[i][5] = int(ticketNo)
	else:
		dirtyFeatures[i][5] = 0

X = dirtyFeatures
y = np.ravel(y)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

numVal = int(VALIDATION_SPLIT * X.shape[0])
X_train = X[:-numVal]
y_train = y[:-numVal]
X_test = X[-numVal:]
y_test = y[-numVal:]

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

print np.argmax(clf.feature_importances_)


print clf.score(X_test, y_test)

# dot -Tps irisTree.dot -o irisTree.ps
export_graphviz(clf, out_file = 'titanicTree.dot')