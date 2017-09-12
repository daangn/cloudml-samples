from os import listdir
from os.path import isfile, join
from random import shuffle
import argparse

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
    default='data/flower_photos',
    help='category images dir path (default: data/flower_photos)')

args = parser.parse_args()

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

root = args.data_dir
categories = dirs(root)

X = []
y = []
for category in categories:
  category_path = join(root, category)
  category_files = files(category_path)
  category_files = [join(category_path, f)
                    for f in category_files if f.endswith('.jpg')]
  X += category_files
  y += [category] * len(category_files)

X = np.array(X)
y = np.array(y)
sss = StratifiedShuffleSplit(n_splits=1, test_size=EVAL_RATIO)
train_index, test_index = next(sss.split(X, y))

trains = zip(X[train_index], y[train_index])
evals = zip(X[test_index], y[test_index])

for i, chunk_trains in enumerate(chunks(trains, CHUNK_SIZE)):
  with open("data/train_set%d.csv" % i, 'w') as f:
    for path, label in chunk_trains:
      f.write("%s,%s\n" % (path, label))

for i, chunk_evals in enumerate(chunks(evals, CHUNK_SIZE)):
  with open('data/eval_set%d.csv' % i, 'w') as f:
    for path, label in chunk_evals:
      f.write("%s,%s\n" % (path, label))
