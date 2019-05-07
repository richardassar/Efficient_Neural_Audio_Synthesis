import os
import math
import time
import tensorflow as tf
from model import WaveRNN
from utils import combine_signal

#
hidden_size = 896
sample_rate = 24000
tbptt_size = 960
data_path = "./test_dataset_5"
q_levels = 256
batch_size = 1

#
out_size = sample_rate * 1

#
out_ta_coarse = tf.TensorArray(dtype=tf.int32, size=out_size, clear_after_read=False, element_shape=(batch_size, 1))
out_ta_fine = tf.TensorArray(dtype=tf.int32, size=out_size, clear_after_read=False, element_shape=(batch_size, 1))

out_ta_coarse = out_ta_coarse.write(0, tf.fill((batch_size, 1), 127))
out_ta_fine = out_ta_fine.write(0, tf.fill((batch_size, 1), 127))

#
wavernn = WaveRNN(hidden_size=hidden_size)

#
hidden_state = tf.zeros((1, hidden_size))
i0 = tf.constant(1)
def body(i, out_ta_coarse, out_ta_fine, hidden_state):
    inp_coarse = out_ta_coarse.read(i-1)
    inp_fine = out_ta_fine.read(i-1)

    out_coarse, out_fine, next_hidden = wavernn.generate(inp_coarse, inp_fine, hidden_state)

    out_ta_coarse = out_ta_coarse.write(i, out_coarse)
    out_ta_fine = out_ta_fine.write(i, out_fine)
    
    return [i + 1, out_ta_coarse, out_ta_fine, next_hidden]

i0, out_ta_coarse, out_ta_fine, next_hidden = tf.while_loop(lambda i, out_ta_coarse, out_ta_fine, hidden_state: i < out_size, body, loop_vars=[i0, out_ta_coarse, out_ta_fine, hidden_state], swap_memory=True)

#
out_coarse = out_ta_coarse.stack()
out_fine = out_ta_fine.stack()

out = combine_signal(out_coarse, out_fine)

out = tf.squeeze(out, axis=-2)

aud = tf.cast(out, dtype=tf.float32)
aud = aud / 2**15

encoded_audio_data = tf.contrib.ffmpeg.encode_audio(aud, file_format="wav", samples_per_second=sample_rate)
write_file_op = tf.write_file("sample.wav", encoded_audio_data)

#
saver = tf.train.Saver()

#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   

    #
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints_wavernn/wavernn'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path) 

    #
    sess.run(write_file_op)