import numpy as np
import tensorflow as tf

def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):

    def __init__(self):
        """ We put a few counters to see how many times we called each function """
        self._count_conv        = 0
        self._count_pool        = 0
        self._count_bn          = 0
        self._count_dropouts    = 0
        self._count_activations = 0
        self._count_fc          = 0
        self._count_lstm        = 0
        self._count_soft_max    = 0
        self._conv_kernels      = []
        self._conv_strides      = []
        self._weights           = {}
        self._features          = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,
                                            updates_collections=None, scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x, prob=1):
        print ("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, prob,
                               name='dropout' + str(self._count_dropouts))
        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME', dropout_prob=None):
        print (" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            if dropout_prob is not None:
                x = self.dropout(x, dropout_prob)

            return self.activation(x)

    def fc_block(self, x, output_size, dropout_prob=None):
        print (" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            if dropout_prob is not None:
                x = self.dropout(x, dropout_prob)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features


def make_network():
    inp_img = tf.placeholder(tf.float32, shape=[None, 88, 200, 3], name='input_image')
    inp_speed = tf.placeholder(tf.float32, shape=[None, 1], name='input_speed')

    target_control = tf.placeholder(tf.float32, shape=[None, 3], name='target_control')
    #target_command = tf.placeholder(tf.float32, shape=[None, 4], name='target_command')

    network_manager = Network()

    xc = network_manager.conv_block(inp_img, 5, 2, 32, padding_in='VALID')
    print (xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
    print (xc)

    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
    print (xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    print (xc)

    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    print (xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    print (xc)

    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    print (xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    print (xc)

    x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print (x)

    x = network_manager.fc_block(x, 512, dropout_prob=1)
    print (x)
    x = network_manager.fc_block(x, 512, dropout_prob=1)

    with tf.name_scope("Speed"):
        speed = network_manager.fc_block(inp_speed, 128, dropout_prob=1)
        speed = network_manager.fc_block(speed, 128, dropout_prob=1)

    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512, dropout_prob=1)

    control_out = network_manager.fc_block(j, 256, dropout_prob=1)
    control_out = network_manager.fc_block(control_out, 256)
    control_out = network_manager.fc(control_out, 3)
    loss = tf.reduce_sum(tf.square(tf.subtract(control_out, target_control)))

    '''
    branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], \
                     ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]

    branches = []
    losses = []
    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            branch_output = network_manager.fc_block(j, 256, dropout_prob=1)
            branch_output = network_manager.fc_block(branch_output, 256)
            branches.append(network_manager.fc(branch_output, len(branch_config[i])))
            losses.append(tf.square(tf.subtract(branches[i], target_control)))

        print (branch_output)

    losses = tf.convert_to_tensor(losses)
    losses = tf.reduce_mean(tf.transpose(losses, [1, 2, 0]), axis=1) * target_command;
    loss = tf.reduce_sum(losses)
    '''

    return {'loss': loss,
            'inputs': [inp_img, inp_speed],
            'labels': [target_control],
            'outputs': [control_out]}

