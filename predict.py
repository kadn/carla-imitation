import tensorflow as tf
import numpy as np

from network import make_network
from data_provider import DataProvider
from tensorflow.core.protobuf import saver_pb2

import time
import os

from IPython import embed

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    network = make_network()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
    saver.restore(sess, './data/step-10500.ckpt')

    val_provider = DataProvider('val.tfrecords', sess)

    one_batch = val_provider.get_minibatch()

    for i in range(120):
        one_image = one_batch.images[i,...][None]
        one_speed = one_batch.data[0][i][None]
        a = time.time()
        target_control, = sess.run(network['outputs'],
	           feed_dict={network['inputs'][0]: one_image,
	                      network['inputs'][1]: one_speed})
        b = time.time()
        print("Inference consumes %.5f seconds" % (b-a))
        print(target_control[0])
