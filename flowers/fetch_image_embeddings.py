import os
import glob
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
        os.mkdir(to_path)
    to_path = "%s/%d.emb" % (to_path, id)
    url = 'http://ml.daangn.com/articles/image_embeddings/%d/%d.emb' % (shard, id)
    return call(['curl', '-f', '-o', to_path, url])

def main():
    with open('data/emb.csv') as f:
        new_ids = set([int(line.split(',')[0]) for line in f.readlines()[1:]])

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
    main()
