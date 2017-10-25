import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.contrib.rnn.python.ops import rnn as contrib_rnn
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper

from attention import attention
from modules import encoder_cbhg, prenet
from rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper
from sequence_example_lib import flatten_maybe_padded_sequences

def make_rnn_cells(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
  """Makes a RNN cell from the given hyperparameters.
  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units)
    if attn_length and not cells:
      # Add attention wrapper to first layer.
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, attn_length, state_is_tuple=True)
    else:
      cell = tf.contrib.rnn.ResidualWrapper(cell)
    if dropout_keep_prob is not None and dropout_keep_prob < 1.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)
  return cells

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
    cells = make_rnn_cells(rnn_layer_sizes, dropout_keep_prob, attn_length, base_cell)
    return tf.contrib.rnn.MultiRNNCell(cells)

def stack_bidirectional_dynamic_rnn(inputs, layer_sizes, sequence_length,
        attn_length=0, dropout_keep_prob=1.0, base_cell=lstm_ops.LSTMBlockCell):
    cells_fw = make_rnn_cells(layer_sizes, dropout_keep_prob=dropout_keep_prob,
          attn_length=attn_length, base_cell=base_cell)
    cells_bw = make_rnn_cells(layer_sizes, dropout_keep_prob=dropout_keep_prob,
          attn_length=attn_length, base_cell=base_cell)

    outputs, output_state_fw, output_state_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
      cells_fw, cells_bw, inputs,
      sequence_length=sequence_length,
      dtype=tf.float32)
    output_state_fw = output_state_fw[-1]
    output_state_bw = output_state_bw[-1]
    return tf.concat([output_state_fw.h, output_state_bw.h], 1) # eval: 81, 500
    #last_outputs = tf.concat([output_state_fw[-1].c, output_state_fw[-1].h, output_state_fw[-1].c, output_state_bw[-1].h], 1) # eval: 83, 1200
    #last_outputs = outputs[:, 0] # eval: 83, 1000

    #range1 = tf.range(sequence_length.shape[0], dtype=tf.int64)
    #nd = tf.stack([range1, sequence_length - 1], 1)
    #last_outputs = tf.gather_nd(outputs, nd) # eval: 76, 600

def simple_rnn(inputs, num_units, sequence_length, dropout_keep_prob=1.0, attn_length=0):
    #cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True)
    if dropout_keep_prob is not None:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=dropout_keep_prob)
    if attn_length:
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, attn_length, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs,
            sequence_length=sequence_length, dtype=tf.float32)
    #print state
    if attn_length:
        return tf.concat([state[0].c, state[0].h], 1)
        #return tf.concat([state[0].h, state[1]], 1)
    return tf.concat([state.c, state.h], 1)

def multi_rnn(inputs, layers_size, sequence_length, dropout_keep_prob=1.0,
        attn_length=0, base_cell=tf.contrib.rnn.BasicLSTMCell):
    cells = make_rnn_cells(layers_size, dropout_keep_prob=dropout_keep_prob,
            attn_length=attn_length, base_cell=base_cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, inputs,
            sequence_length=sequence_length, dtype=tf.float32)
    if attn_length:
        return tf.reduce_sum(outputs, 1)
        return tf.reduce_sum(outputs, 1) / \
              tf.reshape(tf.cast(sequence_length, tf.float32), [-1, 1])
        return tf.concat([states[0][0].h, states[0][1]], 1) # 82
    return tf.concat([states[-1].c, states[-1].h], 1)
    return tf.concat([states[0].c, states[0].h], 1)

def tacotron(inputs, layers_size, sequence_length, dropout_keep_prob=1.0,
        attn_length=0, base_cell=tf.contrib.rnn.BasicLSTMCell):
    is_training = True
    prenet_outputs = prenet(inputs, is_training)
    outputs, states = encoder_cbhg(prenet_outputs, sequence_length, is_training)
    return tf.concat(states, 1)

def tmp():
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01)
  def lstm_cell():
      hidden_size = RNN_UNIT_SIZE
      input_size = CONTENT_DIM
      cell = tf.contrib.rnn.LSTMCell(hidden_size, input_size, initializer=initializer, state_is_tuple=True)
      return cell

  if True:
      attn_length = 16
      cells = [lstm_cell() for _ in range(2)]
      cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
      cell = tf.contrib.rnn.AttentionCellWrapper(
          cell, attn_length, state_is_tuple=True)
      outputs, states = tf.nn.dynamic_rnn(cell, content_embeddings,
              sequence_length=content_lengths, dtype=tf.float32)
      #last_outputs = states[0][-1].h
      last_outputs = tf.concat([states[0][-1].h, states[-1]], 1)
  elif True:
      content_embeddings = tf.unstack(content_embeddings, 200, 1)
      cell = lstm_ops.LSTMBlockFusedCell(RNN_UNIT_SIZE)
      content_lengths = tf.cast(content_lengths, tf.int32)
      outputs, state = cell(content_embeddings, sequence_length=content_lengths,
              dtype=tf.float32)
      last_outputs = state.h
  elif True:
      layer_sizes = [RNN_UNIT_SIZE, RNN_UNIT_SIZE]
      cell = make_rnn_cell(layer_sizes, dropout_keep_prob=dropout_keep_prob,
              base_cell=lstm_ops.LSTMBlockCell,
              attn_length=16)
      outputs, final_state = tf.nn.dynamic_rnn(cell, content_embeddings,
              sequence_length=content_lengths, swap_memory=True, dtype=tf.float32)
      last_outputs = final_state[-1].h
      #last_outputs = tf.concat([final_state[-1].h, final_state[0][1]], 1)
  elif True:
      cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
      outputs, states = tf.nn.dynamic_rnn(cell, content_embeddings, sequence_length=content_lengths, dtype=tf.float32)
      last_outputs = states[-1].h
  elif True:
      num_hidden = RNN_UNIT_SIZE
      cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
      cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
      outputs, states  = tf.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, content_embeddings,
          sequence_length=content_lengths,
          dtype=tf.float32)
      output_fw, output_bw = outputs
      output_state_fw, output_state_bw = states
      #last_outputs = tf.concat([output_fw[:, 0], output_state_bw.h], 1)
      last_outputs = tf.concat([output_state_fw.h, output_state_bw.h], 1)
  elif True:
      num_hidden = RNN_UNIT_SIZE
      lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
      lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
      content_embeddings = tf.unstack(content_embeddings, 200, 1)
      outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
              lstm_fw_cell, lstm_bw_cell, content_embeddings,
              sequence_length=content_lengths,
              dtype=tf.float32)
      last_outputs = outputs[-1]
