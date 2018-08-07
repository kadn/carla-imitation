from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np
import cv2

slim = tf.contrib.slim

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from tensorflow.core.protobuf import saver_pb2
from agents.imitation.network import make_network
from PIL import Image


class ImitationLearning(Agent):

    def __init__(self, city_name, avoid_stopping, memory_fraction=0.25, image_cut=[115, 510]):

        Agent.__init__(self)
        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping
        self.network = make_network()
        self._sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self._sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
        saver.restore(self._sess, './agents/imitation/mymodel/epoch-374.ckpt')
        print('hellohellohellohellohellohello')

        self._image_cut = image_cut

    def run_step(self, measurements, sensor_data, directions, target):

        control = self._compute_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed, directions)

        return control

    def _compute_action(self, rgb_image, speed, direction=None):

        # rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)

        steer, acc, brake = self._control_function(image_input, speed, self._sess)

        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0
        if acc <= 0:
            acc = 0
        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, sess):


        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        # speed = 0
        speed = np.array([[speed]])
        # print(image_input.shape ,image_input.dtype)
        # cv2.imwrite('/home/kadn/Desktop/test.jpg', image_input[0].astype('uint8'))
        # img = cv2.imread("/home/kadn/Desktop/test.jpg")
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # img = img.reshape((1, self._image_size[0], self._image_size[1], self._image_size[2]))
        target_control = sess.run(self.network['outputs'][0],
                                  feed_dict={self.network['inputs'][0]:image_input,
                                             self.network['inputs'][1]: speed})

        predicted_steers = (target_control[0][0])

        predicted_acc = (target_control[0][1])

        predicted_brake = (target_control[0][2])

        return predicted_steers, predicted_acc, predicted_brake
