import os
import time
import tensorflow as tf
import numpy as np
from model import WaveRNN
from utils import split_signal

#
hidden_size = 896
sample_rate = 24000
batch_size = 128
tbptt_size = 960
data_path = "./dataset/data.wav"
q_levels = 256

#
dataset = tf.data.Dataset.list_files(data_path)
dataset = dataset.map(lambda fname: tf.contrib.ffmpeg.decode_audio(tf.read_file(fname), file_format='wav', samples_per_second=sample_rate, channel_count=1))

def normalize(x):
    with tf.name_scope('normalize'):
        neg_peak = tf.abs(tf.reduce_min(x))
        pos_peak = tf.reduce_max(x)
        peak = tf.maximum(neg_peak, pos_peak)
        return x / tf.clip_by_value(peak, np.finfo(np.float32).eps, np.finfo(np.float32).max) 

dataset = dataset.map(normalize)

def quantize(x):
    x = (x + 1) / 2
    x = -0x8000 + 0xFFFF * x
    x = tf.cast(x, dtype=tf.int32)    
    return x

dataset = dataset.map(quantize)
dataset = dataset.cache()

def random_slice(x):
    start = tf.random_uniform([], 0, tf.shape(x)[0] - (tbptt_size + 1) + 1, tf.int32)
    x = x[start:start + tbptt_size + 1]
    return x

dataset = dataset.map(random_slice)
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

#
aud_data_c, aud_data_f = split_signal(next_element)
aud_data = tf.concat([aud_data_c, aud_data_f], axis=-1)

tgt = aud_data[:,1:,:]

aud_data = tf.cast(aud_data, dtype=tf.float32)
aud_data = (aud_data / 255) * 2 - 1

aud_data = tf.transpose(aud_data, perm=[1,0,2])

# 
out_ta_coarse = tf.TensorArray(dtype=tf.float32, size=tbptt_size, clear_after_read=False, element_shape=(batch_size, q_levels))
out_ta_fine = tf.TensorArray(dtype=tf.float32, size=tbptt_size, clear_after_read=False, element_shape=(batch_size, q_levels))

#
wavernn = WaveRNN(hidden_size=hidden_size)

#
hidden_state = tf.zeros((batch_size, hidden_size))
i0 = tf.constant(0)
def body(i, out_ta_coarse, out_ta_fine, hidden_state):
    current_aud = aud_data[i,:,:]
    next_coarse = aud_data[i+1,:,:][:,:1]
    
    out_coarse, out_fine, next_hidden_state = wavernn(current_aud, next_coarse, hidden_state)
     
    out_ta_coarse = out_ta_coarse.write(i, out_coarse)    
    out_ta_fine = out_ta_fine.write(i, out_fine)    

    return [i + 1, out_ta_coarse, out_ta_fine, next_hidden_state]

i0, out_ta_coarse, out_ta_fine, next_hidden_state = tf.while_loop(lambda i, out_ta_coarse, out_ta_fine, hidden_state: i < tbptt_size, body, loop_vars=[i0, out_ta_coarse, out_ta_fine, hidden_state], swap_memory=True)

#
out_coarse = out_ta_coarse.stack()
out_fine = out_ta_fine.stack()

out_coarse = tf.transpose(out_coarse, perm=[1,0,2])
out_fine = tf.transpose(out_fine, perm=[1,0,2])

#
coarse_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_coarse, labels=tgt[:,:,0])
fine_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_fine, labels=tgt[:,:,1])

coarse_loss_avg = tf.reduce_mean(coarse_loss)
fine_loss_avg = tf.reduce_mean(fine_loss)
tf.summary.scalar('coarse_loss', coarse_loss_avg)
tf.summary.scalar('fine_loss', fine_loss_avg)

total_loss = tf.concat([coarse_loss, fine_loss], axis=-1)

loss = tf.reduce_mean(total_loss)
tf.summary.scalar('loss', loss)

#
optimizer = tf.train.AdamOptimizer(0.001)

gvs = optimizer.compute_gradients(loss)
#clipped_grads_and_vars = [(None if grad is None else tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]

global_step = tf.train.get_or_create_global_step()

train = optimizer.apply_gradients(gvs, global_step=global_step)

#
saver = tf.train.Saver()
merged_summary = tf.summary.merge_all()

#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   

    #
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter(logdir='logdir/wavernn', graph=graph)
    writer.flush()

    #
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints_wavernn/wavernn'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path) 

    #
    while(True):        
        start = time.time()
        _, _global_step, _loss, _coarse_loss, _fine_loss, _summary = sess.run([train, global_step, loss, coarse_loss_avg, fine_loss_avg, merged_summary])                                                    
        end = time.time()

        print "Iter %d: loss = %f coarse_loss = %f fine_loss = %f time = %f" % (_global_step, _loss, _coarse_loss, _fine_loss, end - start)

        writer.add_summary(_summary, global_step=_global_step)
        
        if _global_step % 1000 == 999:
            print("Saving checkpoint...")
            saver.save(sess, 'checkpoints_wavernn/wavernn', global_step=_global_step)
            print("Done!")