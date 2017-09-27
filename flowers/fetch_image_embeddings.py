import os
import sys
import glob
import logging
from subprocess import call
from multiprocessing import Pool

emb_path = 'data/image_embeddings'

def remove(id):
    shard = id / 10000
    filepath = "%s/%d/%d.emb" % (emb_path, shard, id)
    return os.remove(filepath)

def down(id):
    shard = id / 10000
    to_path = "%s/%d" % (emb_path, shard)
    if not os.path.exists(to_path):
        try:
            os.mkdir(to_path)
        except OSError as e:
            if e.errno != 17: # File exists
                raise e
    to_path = "%s/%d.emb" % (to_path, id)
    url = 'http://ml.daangn.com/articles/image_embeddings/%d/%d.emb' % (shard, id)
    logging.info('down: %s', url)
    return call(['curl', '-f', '-o', to_path, url])

def main():
    with open('data/emb.csv') as f:
        rows = [line.split(',') for line in f.readlines()[1:]]
        new_ids = set([int(row[0]) for row in rows if row[4] != '0'])

    pathes = glob.glob('%s/*/*.emb' % emb_path)
    local_ids = set([int(x.split('/')[-1][:-4]) for x in pathes])

    print "local ids count: %d" % len(local_ids)
    print "new ids count: %d" % len(new_ids)

    add_ids = new_ids - local_ids
    remove_ids = local_ids - new_ids

    p = Pool()

    print 'remove'
    results = p.map(remove, remove_ids)
    print 'removed count: %d' % len(results)

    print 'down'
    results = p.map(down, add_ids)
    print 'downloaded count: %d' % len(results)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
