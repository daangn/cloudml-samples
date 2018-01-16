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
--input_dict=PATH_TO_INPUT_DIC_TXT_FILE \
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
import numpy as np

from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

try:
  from model import BOTTLENECK_TENSOR_SIZE, TEXT_EMBEDDING_SIZE
  from model import get_extra_embeddings, GraphReferences
except ImportError:
  from trainer.model import BOTTLENECK_TENSOR_SIZE, TEXT_EMBEDDING_SIZE
  from trainer.model import get_extra_embeddings, GraphReferences


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


class ExtractLabelIdsDoFn(beam.DoFn):
  """Extracts (uri, label_ids) tuples from CSV rows.
  """

  def start_bundle(self, context=None):
    self.label_to_id_map = {}

  # The try except is for compatiblity across multiple versions of the sdk
  def process(self, row, all_labels):
    try:
      row = row.element
    except AttributeError:
      pass
    if not self.label_to_id_map:
      for i, label in enumerate(all_labels):
        label = label.strip()
        if label:
          self.label_to_id_map[label] = i

    # Row format is: image_uri(,label_ids)*
    if not row:
      skipped_empty_line.inc()
      return

    csv_rows_count.inc()
    uri = row[0]
    if not uri:
      invalid_uri.inc()
      return

    # In a real-world system, you may want to provide a default id for labels
    # that were not in the dictionary.  In this sample, we simply skip it.
    # This code already supports multi-label problems if you want to use it.
    label_ids = []
    for label in row[7:]:
      try:
        label_ids.append(self.label_to_id_map[label.strip()])
      except KeyError:
        unknown_label.inc()

    labels_count.inc(len(label_ids))

    if not label_ids:
      unlabeled_image.inc()
    yield row, label_ids


class ReadImageAndConvertToJpegDoFn(beam.DoFn):
  """Read files from GCS and convert images to JPEG format.

  We do this even for JPEG images to remove variations such as different number
  of channels.
  """
  SHAPE = [1, 1, 1, BOTTLENECK_TENSOR_SIZE]

  def process(self, element):
    try:
      row, label_ids = element.element
    except AttributeError:
      row, label_ids = element

    id = int(row[0])
    shard = id / 10000
    emb_filepath = "data/image_embeddings/%d/%d.emb" % (shard, id)
    if not file_io.file_exists(emb_filepath):
        emb_filepath = 'data/empty.emb'

    try:
      embedding = np.frombuffer(
          file_io.read_file_to_string(emb_filepath),
          dtype=np.float32)
      embedding = embedding.reshape(self.SHAPE)
      yield row, label_ids, embedding
    except ValueError as e:
      logging.error('Could not load an embedding file from %s: %s', emb_filepath, str(e))
      error_count.inc()


class ExtractTextDataDoFn(beam.DoFn):
  def __init__(self):
    self.sess = None
    self.tensors = None
    self.extra_embeddings = None

  def start_bundle(self, context=None):
    if not self.sess:
      self.sess = tf.Session()
      self.tensors = GraphReferences()
      self.extra_embeddings = get_extra_embeddings(self.tensors)

  def process(self, element):
    try:
      item, label_ids, embedding = element.element
    except AttributeError:
      item, label_ids, embedding = element

    key = item[0]
    try:
      if item[1] == '':
        text_embedding = [0.0] * TEXT_EMBEDDING_SIZE
      else:
        text_embedding = [float(x) for x in item[1].rstrip().split(' ')]
    except ValueError as e:
      logging.error("%s", item)
      raise e

    category_id = item[2]
    price = item[3]
    images_count = item[4]
    created_at_ts = item[5]
    offerable = item[6]

    extra_embedding = self.sess.run(self.extra_embeddings, feed_dict={
          self.tensors.input_price: [price],
          self.tensors.input_images_count: [images_count],
          self.tensors.input_offerable: [offerable],
          self.tensors.input_created_at_ts: [created_at_ts],
          self.tensors.input_category_id: [category_id],
          })[0]

    yield item, label_ids, embedding, {
          'text_embedding': text_embedding,
          'extra_embedding': list(extra_embedding),
          }


class TFExampleFromImageDoFn(beam.DoFn):
  def start_bundle(self, context=None):
    self.data_map = {}

  def process(self, element):

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int_feature(value):
      return tf.train.Feature(float_list=tf.train.Int64List(value=value))

    try:
      element = element.element
    except AttributeError:
      pass
    row, label_ids, embedding, data = element

    id = row[0]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_uri': _bytes_feature([id]),
        'embedding': _float_feature(embedding.ravel().tolist()),
        'text_embedding': _float_feature(data['text_embedding']),
        'extra_embedding': _float_feature(data['extra_embedding']),
    }))

    if label_ids:
      label_ids.sort()
      example.features.feature['label'].int64_list.value.extend(label_ids)

    yield example


def configure_pipeline(p, opt):
  """Specify PCollection and transformations in pipeline."""
  read_input_source = beam.io.ReadFromText(
      opt.input_path, strip_trailing_newlines=True)
  read_label_source = beam.io.ReadFromText(
      opt.input_dict, strip_trailing_newlines=True)
  labels = (p | 'Read dictionary' >> read_label_source)
  _ = (p
       | 'Read input' >> read_input_source
       | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
       | 'Extract label ids' >> beam.ParDo(ExtractLabelIdsDoFn(),
                                           beam.pvalue.AsIter(labels))
       | 'Read and convert to JPEG'
       >> beam.ParDo(ReadImageAndConvertToJpegDoFn())
       | 'Extract text data' >> beam.ParDo(ExtractTextDataDoFn())
       | 'Embed and make TFExample' >> beam.ParDo(TFExampleFromImageDoFn())
       # TODO(b/35133536): Get rid of this Map and instead use
       # coder=beam.coders.ProtoCoder(tf.train.Example) in WriteToTFRecord
       # below.
       | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
       | 'Save to disk'
       >> beam.io.WriteToTFRecord(opt.output_path,
                                  file_name_suffix='.tfrecord.gz'))


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
      '--input_dict',
      dest='input_dict',
      required=True,
      help='Input dictionary. Specified as text file uri. '
      'Each line of the file stores one label.')
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
      default='flowers-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
      help='A unique job identifier.')
#  parser.add_argument(
#      '--num_workers', default=2, type=int, help='The number of workers.')
  parser.add_argument('--cloud', default=False, action='store_true')
  parser.add_argument(
      '--runner',
      help='See Dataflow runners, may be blocking'
      ' or not, on cloud or not, etc.')
  parser.add_argument(
      '--extra_package', default='./dist/trainer-0.1.1.tar.gz', type=str,
      help='Path to the extra package path.')

  parsed_args, _ = parser.parse_known_args(argv)

  if parsed_args.cloud:
    # Flags which need to be set for cloud runs.
    default_values = {
        'project':
            get_cloud_project(),
        'temp_location':
            os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
        'runner':
            'DataflowRunner',
        'save_main_session':
            True,
    }
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  else:
    # Flags which need to be set for local runs.
    default_values = {
        'runner': 'DirectRunner',
    }
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


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
