import tensorflow as tf
import numpy as np
import os
import re
import glob
import argparse
from PIL import Image
from time import clock
import tables

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--log_dir", type=str, default="./log/",
    help="(optional) the directory to store logs")
ap.add_argument("-f", "--fake_data", type=int, default=0,
    help="(optional) use fake data for sanity check or not")
ap.add_argument("-s", "--save_model", type=int, default=-1,
    help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load_model", type=int, default=-1,
    help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--model_path", type=str,
    help="(optional) path to weights file")
ap.add_argument("-t", "--test_mode", type=int, default=-1,
    help="(optional) whether you are testing a prediction of a single image")
ap.add_argument("-i", "--image_path", type=str,
    help="(optional) path to the image if you are using test mode" )
args = vars(ap.parse_args())


batch_size = 20
image_height = 320
image_width = 320
num_channels = 3
patch_size = 3
encoder = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
decoder = (512, 512, 512, 512, 512, 256)
final_depth = (64, 48)
affine_size = (1024, 512)
keep_prob = 0.5
label_num = 11
num_epoch = 1
num_iters = 900

def main():
    
    def load_batch():
        if(args["fake_data"]):
            x = np.random.randn(batch_size, image_height, image_width, num_channels)
            y = np.ones((batch_size, ))
            g = np.random.randn(batch_size, 1, 1, conv_final_depth[1])
        return x, y, g

    def reformat(data, labels):
        data = data.reshape(
            (-1, image_height, image_width, num_channels)).astype(np.float32)
        labels = (np.arange(label_num) == labels[:,None]).astype(np.uint32).reshape((-1, label_num))
        return data, labels

    sess = tf.InteractiveSession()
    
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [batch_size, image_height, image_width, num_channels], name='x-input')
        y_ = tf.placeholder(tf.float32, [batch_size, label_num], name='y-input')
   
    def weight_init(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    # use KaimingHe initialization for the conv weights. 
    def conv_weight_init(shape):
        initial = tf.truncated_normal(shape, stddev=0.1) / np.sqrt((shape[0]*shape[1]*shape[2]*shape[3])/2)
        return tf.variable(initial)
        
    # zero init bias
    def bias_init(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    # for tensorboard use... 
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var) 
   
    def affine_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_init([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_init([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def conv_layer(input_tensor, input_filters, output_filters, layer_name, act=tf.nn.relu, patch_size=patch_size):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = conv_weight_init([patch_size, patch_size, input_filters, output_filters])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_init(output_filters)
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding='SAME') + biases
                tf.summary.histogram('pre_activations', preactivate)
            normalized = batchnorm_for_conv(preactivate)
            activations = act(normalized, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def dropout_layer(input_tensor, layer_name, keep_prob=keep_prob):
        with tf.name_scope(layer_name):
            output = tf.nn.dropout(input_tensor, keep_prob)
            return output

    def maxpool_layer(input_tensor, layer_name):
        with tf.name_scope(layer_name):
            output = tf.nn.max_pool(input_tensor, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            return output 
    
    def batchnorm_for_conv(layer):
        shape = layer.get_shape().as_list()
        flatten = tf.reshape(layer, [shape[0], shape[1]*shape[2]*shape[3]])
        D = flatten.get_shape()[-1]
        scale = tf.Variable(tf.ones([D]))
        beta = tf.Variable(tf.zeros([D]))
        pop_mean = tf.Variable(tf.zeros([D]), trainable=False)
        pop_var = tf.Variable(tf.ones([D]), trainable=False)
        epsilon = 1e-3
        decay = 0.999

        if(args["test_mode"] <= 0):
            batch_mean, batch_var = tf.nn.moments(flatten,[0])
            train_mean = tf.assign(pop_mean,
                pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                norm = tf.nn.batch_normalization(flatten,
                    batch_mean, batch_var, beta, scale, epsilon)
                return tf.reshape(norm, [shape[0], shape[1], shape[2], shape[3]])
        else:
            norm = tf.nn.batch_normalization(flatten,
                pop_mean, pop_var, beta, scale, epsilon)
            return tf.reshape(norm, [shape[0], shape[1], shape[2], shape[3]])

    ''' model definition '''
    # encoder part
    encoder0 = conv_layer(x, num_channels, encoder[0], 'encoder0')
    encoder1 = conv_layer(encoder0, encoder[0], encoder[1], 'encoder1')
    pool1 = maxpool_layer(encoder1, 'pool1')

    encoder2 = conv_layer(pool1,encoder[1], encoder[2], 'encoder2')
    encoder3 = conv_layer(encoder2, encoder[2], encoder[3], 'encoder3')
    pool2 = maxpool_layer(encoder3, 'pool2')

    encoder4 = conv_layer(pool2, encoder[3], encoder[4], 'encoder4') 
    encoder5 = conv_layer(encoder4, encoder[4], encoder[5], 'encoder5')
    encoder6 = conv_layer(encoder5, encoder[5], encoder[6], 'encoder6')
    pool3 = maxpool_layer(encoder6, 'pool3')
    drop1 = dropout_layer(pool3, 'drop1')

    encoder7 = conv_layer(drop1, encoder[6], encoder[7], 'encoder7') 
    encoder8 = conv_layer(encoder7, encoder[7], encoder[8], 'encoder8')
    encoder9 = conv_layer(encoder8, encoder[8], encoder[9], 'encoder9')
    pool4 = maxpool_layer(encoder9, 'pool4')
    drop2 = dropout_layer(pool4, 'drop2')

    encoder10 = conv_layer(drop2, encoder[9], encoder[10], 'encoder10')
    encoder11 = conv_layer(encoder10, encoder[10], encoder[11], 'encoder11')
    encoder12 = conv_layer(encoder11, encoder[11], encoder[12], 'encoder12')
    pool5 = maxpool_layer(encoder12, 'pool5'])
    drop3 = dropout_layer(pool5, 'drop3')
    
    #decoder part
    decoder0 = conv_layer(drop3, encoder[12], decoder[0], 'decoder0')
    decoder1 = conv_layer(decoder0, decoder[0], decoder[1], 'decoder1')
    decoder2 = conv_layer(decoder1, decoder[1], decoder[2], 'decoder2')
    drop4 = dropout_layer(decoder2, 'drop4')

    decoder3 = conv_layer(drop4, decoder[2], decoder[3], 'decoder3')
    decoder4 = conv_layer(decoder3, decoder[3], decoder[4], 'decoder4')
    decoder5 = conv_layer(decoder4, decoder[4], decoder[5], 'decoder5')
    
    final1 = conv_layer()
    


    # Do not apply softmax activation yet, see below.
    y = affine_layer(dropped, 500, 11, 'layer2', act=tf.identity) 

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(
            cross_entropy)
    
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args["log_dir"] + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(args["log_dir"] + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train or args["fake_data"]:
            xs, ys = load_batch()
            xs, ys = reformat(xs, ys)
        return {x: xs, y_: ys}

    for i in range(num_epoch):
        for j in range(num_iters):
            if j % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%d' % j)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
main()
