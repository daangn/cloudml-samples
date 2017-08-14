# Copyright 2016 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example dataflow pipeline for preparing image training data.

The tool requires two main input files:

'input' - URI to csv file, using format:
gs://image_uri1,labela,labelb,labelc
gs://image_uri2,labela,labeld
...

'input_dict' - URI to a text file listing all labels (one label per line):
labela
labelb
labelc

The output data is in format accepted by Cloud ML framework.

This tool produces outputs as follows.
It creates one training example per each line of the created csv file.
When processing CSV file:
- all labels that are not present in input_dict are skipped

To execute this pipeline locally using default options, run this script
with no arguments. To execute on cloud pass single argument --cloud.

To execute this pipeline on the cloud using the Dataflow service and non-default
options:
python -E preprocess.py \
--input_path=PATH_TO_INPUT_CSV_FILE \
--output_path=YOUR_OUTPUT_PATH \
--cloud

For other flags, see PrepareImagesOptions() bellow.

To run this pipeline locally run the above command without --cloud.

TODO(b/31434218)
"""

# TODO(mikehcheng): Beam convention for stage names is CapitalCase as opposed to
# English sentences (eg ReadAndConvertToJpeg as opposed to
# "Read and convert to JPEG"). Fix all samples that don't conform to the
# convention.

# TODO(mikehcheng): Standardize the casing of the various counters (metrics)
# used within this file. So far we have been using underscore_case for metrics.


import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
import tarfile
import random
from importlib import import_module
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen # for python2

import apache_beam as beam
from apache_beam.metrics import Metrics
# pylint: disable=g-import-not-at-top
# TODO(yxshi): Remove after Dataflow 0.4.5 SDK is released.
try:
  try:
    from apache_beam.options.pipeline_options import PipelineOptions
  except ImportError:
    from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
  from apache_beam.utils.options import PipelineOptions
from PIL import Image
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
handler.setFormatter(formatter)
root = logging.getLogger()
level = logging.DEBUG
root.setLevel(level)
root.addHandler(handler)

slim = tf.contrib.slim

error_count = Metrics.counter('main', 'errorCount')
missing_label_count = Metrics.counter('main', 'missingLabelCount')
csv_rows_count = Metrics.counter('main', 'csvRowsCount')
labels_count = Metrics.counter('main', 'labelsCount')
labels_without_ids = Metrics.counter('main', 'labelsWithoutIds')
existing_file = Metrics.counter('main', 'existingFile')
non_existing_file = Metrics.counter('main', 'nonExistingFile')
skipped_empty_line = Metrics.counter('main', 'skippedEmptyLine')
embedding_good = Metrics.counter('main', 'embedding_good')
embedding_bad = Metrics.counter('main', 'embedding_bad')
incompatible_image = Metrics.counter('main', 'incompatible_image')
invalid_uri = Metrics.counter('main', 'invalid_file_name')
unlabeled_image = Metrics.counter('main', 'unlabeled_image')
unknown_label = Metrics.counter('main', 'unknown_label')


class Default(object):
  """Default values of variables."""
  FORMAT = 'jpeg'

  # Make sure to update the default checkpoint file if using another
  # inception graph or when a newer checkpoint file is available. See
  # https://research.googleblog.com/2016/08/improving-inception-and-image.html
  IMAGE_GRAPH_CHECKPOINT_URI = (
      'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt')
      #'tmp/cache/faiss_image_index/models/inception_v3.ckpt')


class FetchImageDoFn(beam.DoFn):
    def process(self, row):
        try:
          row = row.element
        except AttributeError:
          pass

        if not row:
          skipped_empty_line.inc()
          return

        csv_rows_count.inc()
        id, url, created_at_ts = row
        try:
            image_bytes = urlopen(url, timeout=5).read()
        except Exception as e:
            logging.error("Error id %s, url %s: %s", id, url, str(e))
            invalid_uri.inc()
            return

        logging.debug("image id(%s) fetched, url: %s", id, url)
        yield id, image_bytes


class ReadImageAndConvertToJpegDoFn(beam.DoFn):
  """Read files from GCS and convert images to JPEG format.

  We do this even for JPEG images to remove variations such as different number
  of channels.
  """

  def process(self, element):
    try:
      id, image_bytes = element.element
    except AttributeError:
      id, image_bytes = element

    try:
      img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error processing image %s: %s', uri, str(e))
      error_count.inc()
      return

    # Convert to desired format and output.
    output = io.BytesIO()
    img.save(output, Default.FORMAT)
    image_bytes = output.getvalue()
    yield id, image_bytes


class EmbeddingsGraph(object):
  """Builds a graph and uses it to extract embeddings from images.
  """

  # These constants are set by Inception v3's expectations.
  WIDTH = 299
  HEIGHT = 299
  CHANNELS = 3

  def __init__(self, tf_session, version='inception_v3'):
    self.version = version
    self.tf_session = tf_session
    # input_jpeg is the tensor that contains raw image bytes.
    # It is used to feed image bytes and obtain embeddings.
    self.input_jpeg, self.embedding = self.build_graph()

    init_op = tf.global_variables_initializer()
    self.tf_session.run(init_op)

    if version == 'inception_v4':
      ckpt_path = 'gs://towneers-ml/models/inception_v4.ckpt'
    else:
      ckpt_path = 'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt'
    self.restore_from_checkpoint(ckpt_path)

  def dim(self):
    return self.embedding.shape[-1].value

  def build_graph(self):
    """Forms the core by building a wrapper around the inception graph.

      Here we add the necessary input & output tensors, to decode jpegs,
      serialize embeddings, restore from checkpoint etc.

      To use other Inception models modify this file. Note that to use other
      models beside Inception, you should make sure input_shape matches
      their input. Resizing or other modifications may be necessary as well.
      See tensorflow/contrib/slim/python/slim/nets/inception_v3.py for
      details about InceptionV3.

    Returns:
      input_jpeg: A tensor containing raw image bytes as the input layer.
      embedding: The embeddings tensor, that will be materialized later.
    """

    input_jpeg = tf.placeholder(tf.string, shape=None)
    image = tf.image.decode_jpeg(input_jpeg, channels=self.CHANNELS)

    # Note resize expects a batch_size, but we are feeding a single image.
    # So we have to expand then squeeze.  Resize returns float32 in the
    # range [0, uint8_max]
    image = tf.expand_dims(image, 0)

    # convert_image_dtype also scales [0, uint8_max] -> [0 ,1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_bilinear(
        image, [self.HEIGHT, self.WIDTH], align_corners=False)

    # Then rescale range to [-1, 1) for Inception.
    image = tf.subtract(image, 0.5)
    inception_input = tf.multiply(image, 2.0)

    embedding = self.get_pre_logits(inception_input)
    return input_jpeg, embedding

  def get_pre_logits(self, inception_input):
    # import model
    if self.version == 'inception_v3':
        from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
    else:
        inception = import_module('nets.inception_v4')

    # Build Inception layers, which expect a tensor of type float from [-1, 1)
    # and shape [batch_size, height, width, channels].
    arg_scope = getattr(inception, '%s_arg_scope' % self.version)
    model = getattr(inception, self.version)
    with tf.contrib.slim.arg_scope(arg_scope()):
      _, end_points = model(inception_input, is_training=False)

    pre_logits_name = (self.version == 'inception_v3' and 'PreLogits' or 'PreLogitsFlatten')
    return end_points[pre_logits_name]

  def restore_from_checkpoint(self, checkpoint_path):
    """To restore inception model variables from the checkpoint file.

       Some variables might be missing in the checkpoint file, so it only
       loads the ones that are avialable, assuming the rest would be
       initialized later.
    Args:
      checkpoint_path: Path to the checkpoint file for the Inception graph.
    """
    # Get all variables to restore. Exclude Logits and AuxLogits because they
    # depend on the input data and we do not need to intialize them from
    # checkpoint.
    scope = ''.join([x.capitalize() for x in self.version.split('_')])
    all_vars = tf.contrib.slim.get_variables_to_restore(
            exclude=['%s/AuxLogits' % scope, '%s/Logits' % scope, 'global_step'])

    saver = tf.train.Saver(all_vars)
    saver.restore(self.tf_session, checkpoint_path)

  def calculate_embedding(self, batch_image_bytes):
    """Get the embeddings for a given JPEG image.

    Args:
      batch_image_bytes: As if returned from [ff.read() for ff in file_list].

    Returns:
      The Inception embeddings (bottleneck layer output)
    """
    return self.tf_session.run(
        self.embedding, feed_dict={self.input_jpeg: batch_image_bytes})


class ExtractEmbeddingDoFn(beam.DoFn):
  """Embeds image bytes and labels, stores them in tensorflow.Example.

  (uri, label_ids, image_bytes) -> (tensorflow.Example).

  Output proto contains 'label', 'image_uri' and 'embedding'.
  The 'embedding' is calculated by feeding image into input layer of image
  neural network and reading output of the bottleneck layer of the network.

  Attributes:
    image_graph_uri: an uri to gcs bucket where serialized image graph is
                     stored.
  """

  def __init__(self):
    self.tf_session = None
    self.graph = None
    self.preprocess_graph = None

  def start_bundle(self, context=None):
    # There is one tensorflow session per instance of TFExampleFromImageDoFn.
    # The same instance of session is re-used between bundles.
    # Session is closed by the destructor of Session object, which is called
    # when instance of TFExampleFromImageDoFn() is destructed.
    if not self.graph:
      self.graph = tf.Graph()
      self.tf_session = tf.InteractiveSession(graph=self.graph)
      with self.graph.as_default():
        self.preprocess_graph = EmbeddingsGraph(self.tf_session, 'inception_v4')
        logging.debug('graph loaded')

  def process(self, element):

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    try:
      element = element.element
    except AttributeError:
      pass
    id, image_bytes = element

    try:
      embedding = self.preprocess_graph.calculate_embedding(image_bytes)
    except errors.InvalidArgumentError as e:
      incompatible_image.inc()
      logging.warning('Could not encode an image id from %s: %s', id, str(e))
      return

    if embedding.any():
      embedding_good.inc()
      logging.debug("embedding extracted id: %s", id)
      yield id, embedding
    else:
      embedding_bad.inc()
      logging.error("embedding extracted error id: %s", id)
      return
    return


class SaveEmbeddingFile(beam.DoFn):
  def __init__(self, path):
      self.path = path
      self.tar = None

  def start_bundle(self):
      if not os.path.exists('embeddings'):
          os.makedirs('embeddings')
      filepath = 'embeddings/embeddings.%s.tgz' % random.randint(0, 100000)
      self.tar = tarfile.open(filepath, "w:gz")
      self.tar_filepath = filepath
      logging.debug("start bundle: %s, filepath: %s", self, filepath)

  def finish_bundle(self):
      files_count = len(self.tar.getmembers())
      self.tar.close()
      if files_count < 1:
          logging.debug("finish bundle: %s, filepath: %s",
                  self, self.tar_filepath)
          return
      to_path = os.path.join(self.path, self.tar_filepath)
      to_dir = os.path.dirname(to_path)
      if not file_io.file_exists(to_dir):
          file_io.recursive_create_dir(to_dir)
      file_io.copy(self.tar_filepath, to_path)
      logging.debug("finish bundle: %s, update filepath: %s to path: %s",
              self, self.tar_filepath, to_path)

  def process(self, element):
    try:
      element = element.element
    except AttributeError:
      pass
    id, embedding = element

    filepath = os.path.join('embeddings/%s.emb' % id)
    file_io.write_string_to_file(filepath, embedding.tostring())
    self.tar.add(filepath)
    logging.debug("add %s file to %s tar file", filepath, self.tar_filepath)
    yield filepath


def configure_pipeline(p, opt):
  """Specify PCollection and transformations in pipeline."""
  print('project: %s' % opt.project)
  read_input_source = beam.io.ReadFromText(opt.input_path,
          skip_header_lines=1)
  output_path = os.path.join(opt.output_path, opt.job_name)
  _ = (p
       | 'Read input' >> read_input_source
       | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
       | 'Fetch image' >> beam.ParDo(FetchImageDoFn())
       | 'Read and convert to JPEG'
       >> beam.ParDo(ReadImageAndConvertToJpegDoFn())
       | 'Extract embedding' >> beam.ParDo(ExtractEmbeddingDoFn())
       | 'Save embedding files' >> beam.ParDo(SaveEmbeddingFile(output_path))
       | 'Save results' >> beam.io.WriteToText(
           os.path.join(output_path, 'files.txt'))
       )

def run(in_args=None):
  """Runs the pre-processing pipeline."""

  pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
  with beam.Pipeline(options=pipeline_options) as p:
    configure_pipeline(p, in_args)


def default_args(argv):
  """Provides default values for Workflow flags."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--input_path',
      required=True,
      help='Input specified as uri to CSV file. Each line of csv file '
      'contains colon-separated GCS uri to an image and labels.')
  parser.add_argument(
      '--output_path',
      required=True,
      help='Output directory to write results to.')
  parser.add_argument(
      '--project',
      type=str,
      help='The cloud project name to be used for running this pipeline')
  parser.add_argument(
      '--job_name',
      type=str,
      default='job-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
      help='A unique job identifier.')
  parser.add_argument(
      '--extra_package', type=str, help='extra_package')
  #parser.add_argument(
  #    '--num_workers', default=20, type=int, help='The number of workers.')
  #parser.add_argument(
  #    '--max_num_workers', default=2, type=int, help='The number of workers.')
  parser.add_argument('--cloud', default=False, action='store_true')
  parser.add_argument(
      '--runner',
      help='See Dataflow runners, may be blocking'
      ' or not, on cloud or not, etc.')

  parsed_args, _ = parser.parse_known_args(argv)

  if parsed_args.cloud:
    # Flags which need to be set for cloud runs.
    default_values = {
        'project': get_cloud_project(),
        'temp_location':
            os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
        'runner': 'DataflowRunner',
        'save_main_session': True,
        'extra_package': 'tmp/slim-0.1.tar.gz',
    }
  else:
    # Flags which need to be set for local runs.
    default_values = {
        'runner': 'DirectRunner',
    }

  for kk, vv in default_values.iteritems():
    if kk not in parsed_args or not vars(parsed_args)[kk]:
      vars(parsed_args)[kk] = vv

  return parsed_args


def get_cloud_project():
  cmd = [
      'gcloud', '-q', 'config', 'list', 'project',
      '--format=value(core.project)'
  ]
  with open(os.devnull, 'w') as dev_null:
    try:
      res = subprocess.check_output(cmd, stderr=dev_null).strip()
      if not res:
        raise Exception('--cloud specified but no Google Cloud Platform '
                        'project found.\n'
                        'Please specify your project name with the --project '
                        'flag or set a default project: '
                        'gcloud config set project YOUR_PROJECT_NAME')
      return res
    except OSError as e:
      if e.errno == errno.ENOENT:
        raise Exception('gcloud is not installed. The Google Cloud SDK is '
                        'necessary to communicate with the Cloud ML service. '
                        'Please install and set up gcloud.')
      raise


def main(argv):
  arg_dict = default_args(argv)
  run(arg_dict)


if __name__ == '__main__':
  main(sys.argv[1:])
