# Copyright 2016 Google Inc. All Rights Reserved.
#
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
"""Flowers classification model.
"""

import argparse
import logging

import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

from rnn import stack_bidirectional_dynamic_rnn, simple_rnn, multi_rnn, tacotron
from modules import encoder_cbhg

import util
from util import override_if_not_in_args

slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
IMAGE_URI_COLUMN = 'image_uri'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

TOTAL_CATEGORIES_COUNT = 33
MAX_PRICE = 460000000.0
MAX_IMAGES_COUNT = 11.0
DAY_TIME = 60.0 * 60 * 24

BOTTLENECK_TENSOR_SIZE = 1536
TEXT_EMBEDDING_SIZE = 10
FEATURES_COUNT = 14
EXTRA_EMBEDDING_SIZE = FEATURES_COUNT + TOTAL_CATEGORIES_COUNT

RNN_UNIT_SIZE = 16
WORD_DIM = 32
MAX_TITLE_LENGTH = 32
MAX_CONTENT_LENGTH = 512
TITLE_EMBEDDING_SIZE = WORD_DIM * MAX_TITLE_LENGTH
CONTENT_EMBEDDING_SIZE = WORD_DIM * MAX_CONTENT_LENGTH


class GraphMod():
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3


def build_signature(inputs, outputs):
  """Build the signature.

  Not using predic_signature_def in saved_model because it is replacing the
  tensor name, b/35900497.

  Args:
    inputs: a dictionary of tensor name to tensor
    outputs: a dictionary of tensor name to tensor
  Returns:
    The signature, a SignatureDef proto.
  """
  signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  # Label count needs to correspond to nubmer of labels in dictionary used
  # during preprocessing.
  parser.add_argument('--label_count', type=int, default=5)
  parser.add_argument('--dropout', type=float, default=0.5)
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '100', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  return Model(args.label_count, args.dropout), task_args


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []
    self.input_image = None
    self.input_text = None
    self.input_category_id = None
    self.input_price = None
    self.input_images_count = None
    self.input_created_at_ts = None
    self.input_offerable = None
    self.input_title_length = None
    self.input_title_embedding = None
    self.input_content_length = None
    self.input_content_embedding = None
    self.input_title_chars_count = None
    self.input_title_words_count = None
    self.input_content_chars_count = None
    self.input_content_words_count = None


def get_extra_embeddings(tensors):
    tensors.input_category_id = tf.placeholder(tf.int32, shape=[None])
    tensors.input_price = tf.placeholder(tf.float32, shape=[None])
    tensors.input_images_count = tf.placeholder(tf.float32, shape=[None])
    tensors.input_created_at_ts = tf.placeholder(tf.float64, shape=[None])
    tensors.input_offerable = tf.placeholder(tf.float32, shape=[None])
    tensors.input_title_chars_count = tf.placeholder(tf.float32, shape=[None])
    tensors.input_title_words_count = tf.placeholder(tf.float32, shape=[None])
    tensors.input_content_chars_count = tf.placeholder(tf.float32, shape=[None])
    tensors.input_content_words_count = tf.placeholder(tf.float32, shape=[None])

    category_id = tensors.input_category_id
    price = tensors.input_price
    images_count = tensors.input_images_count
    created_at_ts = tensors.input_created_at_ts
    offerable = tensors.input_offerable
    title_chars_count = tensors.input_title_chars_count
    title_words_count = tensors.input_title_words_count
    content_chars_count = tensors.input_content_chars_count
    content_words_count = tensors.input_content_words_count

    category = tf.one_hot(category_id - 1, TOTAL_CATEGORIES_COUNT)
    price_norm = tf.minimum(price / MAX_PRICE, 1.0)
    is_free = tf.cast(tf.equal(price, 0), tf.float32)
    price_over_10 = tf.cast(tf.greater_equal(price, 100000), tf.float32)
    price_over_30 = tf.cast(tf.greater_equal(price, 300000), tf.float32)
    price_over_50 = tf.cast(tf.greater_equal(price, 500000), tf.float32)
    high_price = tf.minimum(price / MAX_PRICE / 1000, 1.0)
    images_count_norm = tf.minimum(images_count / MAX_IMAGES_COUNT, 1.0)
    created_hour = created_at_ts % DAY_TIME * 1.0 / DAY_TIME
    created_hour = tf.cast(created_hour, tf.float32)
    day = tf.cast(created_at_ts / DAY_TIME % 7 / 7.0, tf.float32)
    title_chars_count_norm = tf.minimum(title_chars_count / 91.0, 1.0)
    title_words_count_norm = tf.minimum(title_words_count / 15.0, 1.0)
    content_chars_count_norm = tf.minimum(content_chars_count / 2711.0, 1.0)
    content_words_count_norm = tf.minimum(content_words_count / 572.0, 1.0)

    extra_embeddings = tf.concat([price_norm, is_free, price_over_10, price_over_30,
      price_over_50, high_price, images_count_norm, offerable, created_hour, day,
      title_chars_count_norm, title_words_count_norm, content_chars_count_norm,
      content_words_count_norm], 0)
    extra_embeddings = tf.reshape(extra_embeddings, [-1, FEATURES_COUNT])
    extra_embeddings = tf.concat([extra_embeddings, category], 1)
    return extra_embeddings

class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, label_count, dropout):
    self.label_count = label_count
    self.dropout = dropout

  def add_final_training_ops(self,
                             embeddings,
                             all_labels_count,
                             hidden_layer_size=BOTTLENECK_TENSOR_SIZE / 4,
                             dropout_keep_prob=None):
    """Adds a new softmax and fully-connected layer for training.

     The set up for the softmax and fully-connected layers is based on:
     https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

     This function can be customized to add arbitrary layers for
     application-specific requirements.
    Args:
      embeddings: The embedding (bottleneck) tensor.
      all_labels_count: The number of all labels including the default label.
      hidden_layer_size: The size of the hidden_layer. Roughtly, 1/4 of the
                         bottleneck tensor size.
      dropout_keep_prob: the percentage of activation values that are retained.
    Returns:
      softmax: The softmax or tensor. It stores the final scores.
      logits: The logits tensor.
    """
    with tf.name_scope('input'):
      with tf.name_scope('Wx_plus_b'):
        hidden = layers.fully_connected(embeddings, hidden_layer_size)
        # We need a dropout when the size of the dataset is rather small.
        if dropout_keep_prob:
          hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        hidden = layers.fully_connected(hidden, hidden_layer_size / 2)
        if dropout_keep_prob:
          hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        logits = layers.fully_connected(
            hidden, all_labels_count, activation_fn=None)

    softmax = tf.nn.softmax(logits, name='softmax')
    return softmax, logits

  def build_inception_graph(self):
    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    def decode(image_str_tensor):
        embeddings = tf.decode_raw(image_str_tensor, tf.float32)
        return embeddings

    inception_embeddings = tf.map_fn(
        decode, image_str_tensor, back_prop=False, dtype=tf.float32)
    inception_embeddings = tf.reshape(inception_embeddings, [-1, BOTTLENECK_TENSOR_SIZE])
    return image_str_tensor, inception_embeddings

  def build_graph(self, data_paths, batch_size, graph_mod):
    """Builds generic graph for training or eval."""
    tensors = GraphReferences()
    is_training = graph_mod == GraphMod.TRAIN
    if data_paths:
      tensors.keys, tensors.examples = util.read_examples(
          data_paths,
          batch_size,
          shuffle=is_training,
          num_epochs=None if is_training else 2)
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

    if graph_mod == GraphMod.PREDICT:
      inception_input, inception_embeddings = self.build_inception_graph()
      embeddings = inception_embeddings
      text_embeddings = tf.placeholder(tf.float32, shape=[None, TEXT_EMBEDDING_SIZE])
      tensors.input_image = inception_input
      tensors.input_text = text_embeddings

      extra_embeddings = get_extra_embeddings(tensors)
    else:
      # For training and evaluation we assume data is preprocessed, so the
      # inputs are tf-examples.
      # Generate placeholders for examples.
      with tf.name_scope('inputs'):
        feature_map = {
            'image_uri':
                tf.FixedLenFeature(
                    shape=[], dtype=tf.string, default_value=['']),
            # Some images may have no labels. For those, we assume a default
            # label. So the number of labels is label_count+1 for the default
            # label.
            'label':
                tf.FixedLenFeature(
                    shape=[1], dtype=tf.int64,
                    default_value=[self.label_count]),
            'embedding':
                tf.FixedLenFeature(
                    shape=[BOTTLENECK_TENSOR_SIZE], dtype=tf.float32),
            'text_embedding':
                tf.FixedLenFeature(
                    shape=[TEXT_EMBEDDING_SIZE], dtype=tf.float32),
            'extra_embedding':
                tf.FixedLenFeature(
                    shape=[EXTRA_EMBEDDING_SIZE], dtype=tf.float32),
            'title_embedding':
                tf.FixedLenFeature(
                    shape=[TITLE_EMBEDDING_SIZE], dtype=tf.float32),
            'title_length':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'content_embedding':
                tf.FixedLenFeature(
                    shape=[CONTENT_EMBEDDING_SIZE], dtype=tf.float32),
            'content_length':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        labels = tf.squeeze(parsed['label'])
        uris = tf.squeeze(parsed['image_uri'])
        embeddings = parsed['embedding']
        text_embeddings = parsed['text_embedding']
        extra_embeddings = parsed['extra_embedding']
        title_lengths = parsed['title_length']
        title_embeddings = parsed['title_embedding']
        content_lengths = parsed['content_length']
        content_embeddings = parsed['content_embedding']

    # We assume a default label, so the total number of labels is equal to
    # label_count+1.
    all_labels_count = self.label_count + 1
    with tf.name_scope('final_ops'):
      dropout_keep_prob = self.dropout if is_training else None

      title_embeddings = tf.reshape(title_embeddings, [-1, MAX_TITLE_LENGTH, WORD_DIM])
      title_lengths = tf.reshape(title_lengths, [-1])
      layer_sizes = [32,32]
      title_outputs = multi_rnn(title_embeddings, layer_sizes, title_lengths,
              dropout_keep_prob=dropout_keep_prob, attn_length=0,
              base_cell=tf.contrib.rnn.BasicLSTMCell)

      content_embeddings = tf.reshape(content_embeddings, [-1, MAX_CONTENT_LENGTH, WORD_DIM])
      content_lengths = tf.reshape(content_lengths, [-1])
      layer_sizes = [32,64]
      content_outputs = stack_bidirectional_dynamic_rnn(content_embeddings, layer_sizes, content_lengths,
              dropout_keep_prob=dropout_keep_prob, attn_length=0,
              base_cell=tf.contrib.rnn.LSTMBlockCell)

      title_lengths = tf.maximum(title_lengths, 1)
      content_lengths = tf.maximum(content_lengths, 1)
      title_mean_embeddings = tf.reduce_sum(title_embeddings, 1) / \
              tf.reshape(tf.cast(title_lengths, tf.float32), [-1, 1])
      content_mean_embeddings = tf.reduce_sum(content_embeddings, 1) / \
              tf.reshape(tf.cast(content_lengths, tf.float32), [-1, 1])

      text_outputs = tf.concat([title_outputs, content_outputs], 1)
      text_embeddings = tf.concat([title_mean_embeddings, content_mean_embeddings], 1)
      text_embeddings = tf.concat([text_embeddings, text_outputs], 1)

      text_embeddings = layers.fully_connected(text_embeddings, 64)
      if dropout_keep_prob:
          text_embeddings = tf.nn.dropout(text_embeddings, dropout_keep_prob)

      embeddings = layers.fully_connected(embeddings, 192)
      if dropout_keep_prob:
          embeddings = tf.nn.dropout(embeddings, dropout_keep_prob)

      extra_embeddings = layers.fully_connected(extra_embeddings, EXTRA_EMBEDDING_SIZE / 2,
              normalizer_fn=tf.contrib.layers.batch_norm)
      if dropout_keep_prob:
          extra_embeddings = tf.nn.dropout(extra_embeddings, dropout_keep_prob)

      embeddings = tf.concat([embeddings, extra_embeddings, text_embeddings],
          1, name='article_embeddings')

      softmax, logits = self.add_final_training_ops(
          embeddings,
          all_labels_count,
          hidden_layer_size=BOTTLENECK_TENSOR_SIZE / 8,
          dropout_keep_prob=dropout_keep_prob)

    # Prediction is the index of the label with the highest score. We are
    # interested only in the top score.
    prediction = tf.argmax(softmax, 1)
    tensors.predictions = [prediction, softmax]

    if graph_mod == GraphMod.PREDICT:
      return tensors

    with tf.name_scope('evaluate'):
      loss_value = loss(logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add means across all batches.
    loss_updates, loss_op = util.loss(loss_value)
    accuracy_updates, accuracy_op = util.accuracy(logits, labels)
    recall_op, recall_updates = tf.metrics.recall_at_k(labels, logits, 1)
    precision_op, precision_updates = tf.metrics.sparse_precision_at_k(labels, logits, 1)

    if not is_training:
      tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)
      tf.summary.scalar('recall', recall_op)
      tf.summary.scalar('precision', precision_op)
      tf.summary.histogram('histogram_loss', loss_op)

    tensors.metric_updates = loss_updates + accuracy_updates + [recall_updates, precision_updates]
    tensors.metric_values = [loss_op, accuracy_op, recall_op, precision_op]
    return tensors

  def build_train_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)

  def restore_from_checkpoint(self, session, trained_checkpoint_file):
    """To restore model variables from the checkpoint file.

       The graph is assumed to consist of an inception model and other
       layers including a softmax and a fully connected layer. The former is
       pre-trained and the latter is trained using the pre-processed data. So
       we restore this from two checkpoint files.
    Args:
      session: The session to be used for restoring from checkpoint.
      trained_checkpoint_file: path to the trained checkpoint for the other
                               layers.
    """
    if not trained_checkpoint_file:
      return

    # Restore the rest of the variables from the trained checkpoint.
    trained_saver = tf.train.Saver()
    trained_saver.restore(session, trained_checkpoint_file)

  def build_prediction_graph(self):
    """Builds prediction graph and registers appropriate endpoints."""

    tensors = self.build_graph(None, 1, GraphMod.PREDICT)

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder,
        'image_embedding_bytes': tensors.input_image,
        'text_embedding': tensors.input_text,
        'category_id': tensors.input_category_id,
        'price': tensors.input_price,
        'images_count': tensors.input_images_count,
        'created_at_ts': tensors.input_created_at_ts,
        'offerable': tensors.input_offerable,
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys,
        'prediction': tensors.predictions[0],
        'scores': tensors.predictions[1]
    }

    return inputs, outputs

  def export(self, last_checkpoint, output_dir):
    """Builds a prediction graph and xports the model.

    Args:
      last_checkpoint: Path to the latest checkpoint file from training.
      output_dir: Path to the folder to be used to output the model.
    """
    logging.info('Exporting prediction graph to %s', output_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      # Build and save prediction meta graph and trained variable values.
      inputs, outputs = self.build_prediction_graph()
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      self.restore_from_checkpoint(sess, last_checkpoint)
      signature_def = build_signature(inputs=inputs, outputs=outputs)
      signature_def_map = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
      }
      builder = saved_model_builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess,
          tags=[tag_constants.SERVING],
          signature_def_map=signature_def_map)
      builder.save()

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
      loss_str = '%.3f' % metric_values[0]
      accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s, %s' % (loss_str, accuracy_str)


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(epsilon=0.001)
    train_op = optimizer.minimize(loss_op, global_step)
    return train_op, global_step
