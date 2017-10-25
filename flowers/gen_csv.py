from os import listdir
from os.path import isfile, join
from random import shuffle
from collections import Counter
import linecache

import numpy as np
import pandas
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

def get_scale_max_count(counts, threshold=0.9998):
    total_count = len(counts)
    min_value = total_count * threshold
    s = 0
    for n, c in sorted(Counter(counts).most_common(), key=lambda x: x[0]):
        s += c
        if s >= min_value:
            return n

df = pandas.read_csv('data/emb.csv')
print df.columns

for key in ['title_chars_count', 'title_words_count', 'content_chars_count', 'content_words_count', 'price', 'images_count']:
    print ('%s: %d' % (key, get_scale_max_count(df[key].values)))

with open('data/emb.csv') as f:
    X = f.readlines()[1:]
y = [x.split(',')[7].rstrip() for x in X]

X = np.array([x.rstrip() for x in X])
y = np.array(y)

print('All y counter')
print(Counter(y).most_common())

sss = StratifiedShuffleSplit(n_splits=1, test_size=EVAL_RATIO)
train_index, test_index = next(sss.split(X, y))

trains = zip(train_index, X[train_index], y[train_index])
evals = zip(test_index, X[test_index], y[test_index])

assert len(set(test_index) - set(train_index)) == len(test_index)

print('train y counter')
print(Counter(y[train_index]).most_common())
print('test y counter')
print(Counter(y[test_index]).most_common())

for i, chunked in enumerate(chunks(trains, CHUNK_SIZE)):
  with open("data/train_set%d.csv" % i, 'w') as f:
    for i, x, _ in chunked:
      title = linecache.getline('data/titles.emb', i+1).rstrip()
      content = linecache.getline('data/contents.emb', i+1).rstrip()
      f.write("%s,%s,%s\n" % (x, title, content))

for i, chunked in enumerate(chunks(evals, CHUNK_SIZE)):
  with open('data/eval_set%d.csv' % i, 'w') as f:
    for i, x, _ in chunked:
      title = linecache.getline('data/titles.emb', i+1).rstrip()
      content = linecache.getline('data/contents.emb', i+1).rstrip()
      f.write("%s,%s,%s\n" % (x, title, content))
