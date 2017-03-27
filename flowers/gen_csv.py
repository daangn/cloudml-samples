from os import listdir
from os.path import isfile, join
from random import shuffle

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in xrange(0, len(l), n):
		yield l[i:i + n]

def files(path):
  return [f for f in listdir(path) if isfile(join(path, f))]

def dirs(path):
  return [f for f in listdir(path) if not isfile(join(path, f))]

EVAL_RATIO = 0.2
CHUNK_SIZE = 1500

root = 'data/flower_photos'
categories = dirs(root)

evals = []
trains = []

for category in categories:
  category_path = join(root, category)
  category_files = files(category_path)
  category_files = [f for f in category_files if f.endswith('.jpg')]
  shuffle(category_files)
  i = int(len(category_files) * EVAL_RATIO)
  evals += [(join(category_path, f), category) for f in category_files[:i]]
  trains += [(join(category_path, f), category) for f in category_files[i:]]

shuffle(trains)
shuffle(evals)

for i, chunk_trains in enumerate(chunks(trains, CHUNK_SIZE)):
  with open("data/train_set%d.csv" % i, 'w') as f:
    for path, label in chunk_trains:
      f.write("%s,%s\n" % (path, label))

for i, chunk_evals in enumerate(chunks(evals, CHUNK_SIZE)):
  with open('data/eval_set%d.csv' % i, 'w') as f:
    for path, label in chunk_evals:
      f.write("%s,%s\n" % (path, label))
