import os
import sys
import glob
import logging
from subprocess import call
from multiprocessing import Pool

from trainer.emb import get_shard, id_to_path

IMAGES_COUNT_COL = 3
emb_path = 'data/image_embeddings'

def remove(id):
    filepath = "%s/%s" % (emb_path, id_to_path(id))
    return os.remove(filepath)

def down(id):
    shard = get_shard(id)
    to_path = "%s/%d" % (emb_path, shard)
    if not os.path.exists(to_path):
        try:
            os.mkdir(to_path)
        except OSError as e:
            if e.errno != 17: # File exists
                raise e
    to_path = "%s/%d.emb" % (to_path, id)
    url = 'http://ml.daangn.com/articles/image_embeddings/%s' % id_to_path(id)
    logging.info('down: %s', url)
    return call(['curl', '-f', '--connect-timeout', '2', '-o', to_path, url])

def main():
    with open('data/emb.csv') as f:
        rows = [line.split(',') for line in f.readlines()[1:]]
        new_ids = set([int(row[0]) for row in rows if row[IMAGES_COUNT_COL] != '0'])

    pathes = glob.glob('%s/*/*.emb' % emb_path)
    local_ids = set([int(x.split('/')[-1][:-4]) for x in pathes])

    print "local ids count: %d" % len(local_ids)
    print "new ids count: %d" % len(new_ids)

    add_ids = new_ids - local_ids
    remove_ids = local_ids - new_ids

    p = Pool()

    print 'removing count: %d' % len(remove_ids)
    results = p.map(remove, remove_ids)
    print 'removed'

    print 'downloading %d files' % len(add_ids)
    if len(add_ids) < 1000:
        results = p.map(down, add_ids)
        print 'downloaded count: %d' % len(results)
    else:
        with open('data/down_emb_files.txt', 'w') as f:
            for id in add_ids:
                filepath = id_to_path(id)
                f.write("%s\n" % filepath)
            print '%d files list saved to %s' % (len(add_ids), f.name)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
