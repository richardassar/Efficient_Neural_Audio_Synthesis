import math
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import tensorflow as tf

def kaiming_initializer(seed=None, dtype=dtypes.float32):
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')

  def _initializer(shape, dtype=dtype, partition_info=None):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])      
    else:
      fan_in = 1.0
      
    for dim in shape[:-2]:
      fan_in *= float(dim)
          
    n = fan_in
    limit = math.sqrt(3.0) * (math.sqrt(2.0 / 6.0) / math.sqrt(n))    
    
    return random_ops.random_uniform(shape, -limit, limit, dtype, seed=seed)
  
  return _initializer


def bias_initializer(n, seed=None, dtype=dtypes.float32):
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')

  def _initializer(shape, dtype=dtype, partition_info=None):    
    limit = 1 / math.sqrt(n)
    return random_ops.random_uniform(shape, -limit, limit, dtype, seed=seed)
  
  return _initializer

class WaveRNN(object):
    def __init__(self, hidden_size=896, use_softsign=False):
        self.use_softsign = use_softsign

        self.hidden_size = hidden_size
        self.split_size = self.hidden_size / 2

        self.quantisation_levels = 2**8

        with tf.variable_scope('WaveRNN'):
            self.R = tf.layers.Dense(3 * self.hidden_size, use_bias=False, name="R", kernel_initializer=kaiming_initializer())

            self.O1 = tf.layers.Dense(self.split_size, name="O1", kernel_initializer=kaiming_initializer(), bias_initializer=bias_initializer(self.split_size))
            self.O2 = tf.layers.Dense(self.quantisation_levels, name="O2", kernel_initializer=kaiming_initializer(), bias_initializer=bias_initializer(self.split_size))
            self.O3 = tf.layers.Dense(self.split_size, name="O3", kernel_initializer=kaiming_initializer(), bias_initializer=bias_initializer(self.split_size))
            self.O4 = tf.layers.Dense(self.quantisation_levels, name="O4", kernel_initializer=kaiming_initializer(), bias_initializer=bias_initializer(self.split_size))

            self.I_coarse = tf.layers.Dense(3 * self.split_size, use_bias=False, name="I_coarse", kernel_initializer=kaiming_initializer())
            self.I_fine = tf.layers.Dense(3 * self.split_size, use_bias=False, name="I_fine", kernel_initializer=kaiming_initializer())

            self.bias_u = tf.get_variable('bias_u', (self.hidden_size), initializer=tf.zeros_initializer())
            self.bias_r = tf.get_variable('bias_r', (self.hidden_size), initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable('bias_e', (self.hidden_size), initializer=tf.zeros_initializer())
            
            self.bias_coarse_u, self.bias_fine_u = tf.split(self.bias_u, num_or_size_splits=2, axis=-1)
            self.bias_coarse_r, self.bias_fine_r = tf.split(self.bias_r, num_or_size_splits=2, axis=-1)
            self.bias_coarse_e, self.bias_fine_e = tf.split(self.bias_e, num_or_size_splits=2, axis=-1)

    def __call__(self, current_y, next_coarse, prev_hidden):
        R_hidden = self.R(prev_hidden)
        R_u, R_r, R_e = tf.split(R_hidden, num_or_size_splits=3, axis=-1)

        coarse_input_proj = self.I_coarse(current_y)
        I_coarse_u, I_coarse_r, I_coarse_e = tf.split(coarse_input_proj, num_or_size_splits=3, axis=-1)

        fine_input = tf.concat([current_y, next_coarse], axis=-1)
        fine_input_proj = self.I_fine(fine_input)

        I_fine_u, I_fine_r, I_fine_e = tf.split(fine_input_proj, num_or_size_splits=3, axis=-1)

        I_u = tf.concat([I_coarse_u, I_fine_u], axis=-1)
        I_r = tf.concat([I_coarse_r, I_fine_r], axis=-1)
        I_e = tf.concat([I_coarse_e, I_fine_e], axis=-1)

        u = R_u + I_u + self.bias_u
        r = R_r + I_r + self.bias_r
        
        if self.use_softsign:
            u = (tf.nn.softsign(u) + 1) / 2
            r = (tf.nn.softsign(r) + 1) / 2            
        else:
            u = tf.nn.sigmoid(u)
            r = tf.nn.sigmoid(r)            

        e = r * R_e + I_e + self.bias_e

        if self.use_softsign:
            e = tf.nn.softsign(e)
        else:
            e = tf.nn.tanh(e)

        hidden = u * prev_hidden + (1. - u) * e

        hidden_coarse, hidden_fine = tf.split(hidden, num_or_size_splits=2, axis=-1)

        out_coarse = self.O2(tf.nn.relu(self.O1(hidden_coarse)))
        out_fine = self.O4(tf.nn.relu(self.O3(hidden_fine)))        

        return out_coarse, out_fine, hidden

    def generate(self, prev_coarse, prev_fine, prev_hidden):
        hidden_coarse, hidden_fine = tf.split(prev_hidden, num_or_size_splits=2, axis=-1)    

        prev_coarse = tf.cast(prev_coarse, dtype=tf.float32) / 255 * 2 - 1
        prev_fine = tf.cast(prev_fine, dtype=tf.float32) / 255 * 2 - 1
        prev_outputs = tf.concat([prev_coarse, prev_fine], axis=-1)
        
        coarse_input_proj = self.I_coarse(prev_outputs)
        I_coarse_u, I_coarse_r, I_coarse_e = tf.split(coarse_input_proj, num_or_size_splits=3, axis=-1)

        R_hidden = self.R(prev_hidden)

        R_coarse_u, R_fine_u, \
        R_coarse_r, R_fine_r, \
        R_coarse_e, R_fine_e = tf.split(R_hidden, num_or_size_splits=6, axis=-1)

        u = R_coarse_u + I_coarse_u + self.bias_coarse_u
        r = R_coarse_r + I_coarse_r + self.bias_coarse_r

        if self.use_softsign:
            u = (tf.nn.softsign(u) + 1) / 2
            r = (tf.nn.softsign(r) + 1) / 2
        else:
            u = tf.nn.sigmoid(u)
            r = tf.nn.sigmoid(r)

        e = r * R_coarse_e + I_coarse_e + self.bias_coarse_e

        if self.use_softsign:
            e = tf.nn.softsign(e)
        else:
            e = tf.nn.tanh(e)
            
        hidden_coarse = u * hidden_coarse + (1. - u) * e

        out_coarse = self.O2(tf.nn.relu(self.O1(hidden_coarse)))                    
        out_coarse = tf.multinomial(out_coarse, num_samples=1)        
        out_coarse = tf.cast(out_coarse, dtype=tf.int32)
        
        out_coarse_pred = tf.cast(out_coarse, dtype=tf.float32) / 255 * 2 - 1
        
        fine_input = tf.concat([prev_outputs, out_coarse_pred], axis=-1)

        fine_input_proj = self.I_fine(fine_input)
        I_fine_u, I_fine_r, I_fine_e = tf.split(fine_input_proj, num_or_size_splits=3, axis=-1)
        
        u = R_fine_u + I_fine_u + self.bias_fine_u
        r = R_fine_r + I_fine_r + self.bias_fine_r

        if self.use_softsign:
            u = (tf.nn.softsign(u) + 1) / 2
            r = (tf.nn.softsign(r) + 1) / 2
        else:
            u = tf.nn.sigmoid(u)
            r = tf.nn.sigmoid(r)

        e = r * R_fine_e + I_fine_e + self.bias_fine_e

        if self.use_softsign:
            e = tf.nn.softsign(e)
        else:
            e = tf.nn.tanh(e)
        
        hidden_fine = u * hidden_fine + (1. - u) * e
        
        out_fine = self.O4(tf.nn.relu(self.O3(hidden_fine)))
        out_fine = tf.multinomial(out_fine, num_samples=1)
        out_fine = tf.cast(out_fine, dtype=tf.int32)
        
        next_hidden = tf.concat([hidden_coarse, hidden_fine], axis=-1)
    
        return out_coarse, out_fine, next_hidden