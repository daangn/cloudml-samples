from os import listdir
from os.path import isfile, join
from random import shuffle
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in xrange(0, len(l), n):
		yield l[i:i + n]

def files(path):
  return [f for f in listdir(path) if isfile(join(path, f))]

def dirs(path):
  return [f for f in listdir(path) if not isfile(join(path, f))]

EVAL_RATIO = 0.25
CHUNK_SIZE = 5000

with open('data/emb.csv') as f:
    X = f.readlines()[1:]
y = [x.split(',')[7].rstrip() for x in X]

X = np.array(X)
y = np.array(y)

print('All y counter')
print(Counter(y).most_common())

sss = StratifiedShuffleSplit(n_splits=1, test_size=EVAL_RATIO)
train_index, test_index = next(sss.split(X, y))

trains = zip(X[train_index], y[train_index])
evals = zip(X[test_index], y[test_index])

assert len(set(test_index) - set(train_index)) == len(test_index)

print('train y counter')
print(Counter(y[train_index]).most_common())
print('test y counter')
print(Counter(y[test_index]).most_common())

for i, chunked in enumerate(chunks(trains, CHUNK_SIZE)):
  with open("data/train_set%d.csv" % i, 'w') as f:
    for x, _ in chunked:
      f.write("%s" % x)

for i, chunked in enumerate(chunks(evals, CHUNK_SIZE)):
  with open('data/eval_set%d.csv' % i, 'w') as f:
    for x, _ in chunked:
      f.write("%s" % x)
