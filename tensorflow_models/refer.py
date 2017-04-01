import tensorflow as tf
import numpy as np
import os
import re
import glob
import argparse
from PIL import Image
from time import clock
import tables
from img2txt import img2directMap

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
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

data_path = "./trainData_32.hdf5"
label_path = "./trainLabel.hdf5"

# declare some hyperparameters
batch_size = 1000
image_height = 32
image_width = 32
num_channels = 8
patch_size = 3
depth = (50, 100, 150, 200, 250, 300, 350, 400)
num_hidden = (900, 200)
keep_prob = (1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 1.0)
num_labels = 3755

# reformat data to the intended dim
def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, num_channels, image_height, image_width)).transpose(0,2,3,1).astype(np.float32)
    dataset = dataset / 255
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.uint32).reshape((-1, num_labels))
    return dataset, labels

# shuffle the data and label accordingly
def shuffle(dataset, labels):
    perm = np.random.permutation(len(dataset))
    dataset = dataset[perm]
    labels = labels[perm]
    return dataset, labels

# calculate training or testing accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])

# Data prepare
if(args["test_mode"] <= 0):
    print("[INFO] using training mode")
    print("[INFO] loading features...")
    data_file = tables.open_file(data_path, mode='r')
    print("[INFO] loading labels...")
    label_file = tables.open_file(label_path, mode='r')
    testData = data_file.root.trainData[919000:919973]
    testLabels = label_file.root.trainLabel[919000:919973]
    testData, testLabels = reformat(testData, testLabels)
#if(args["test_mode"] <= 0):
#    print("[INFO] using training mode")
#    print("[INFO] loading features...")
#    features = open(data_path)
#    totalData = features.readline().strip('\t').split('\t')
#    totalData = np.asarray(totalData, dtype='float32')
#    print("[INFO] finished loading features from %s" %data_path)
#    print("[INFO] loading labels...")
#    labels = open(label_path)
#    totalLabels = labels.readline().strip('\t').split('\t')
#    totalLabels = np.asarray(totalLabels, dtype='int32')
#    print("[INFO] finished loading labels from %s" %label_path)
#    totalData, totalLabels = reformat(totalData, totalLabels)
#    # restrict data to [0, 1]
#    totalData = totalData / 255
#    train_size =(int)(0.9 * totalData.shape[0])
#    train_index = np.random.choice(totalData.shape[0], train_size, replace=False)
#    test_index = np.asarray(list(set(np.arange(totalData.shape[0])) - set(train_index)))
#    trainData = totalData[train_index]
#    testData = totalData[test_index]
#    trainLabels = totalLabels[train_index]
#    testLabels = totalLabels[test_index]
#
#    print('[INFO] Training set', trainData.shape, trainLabels.shape)
#    print('[INFO] Test set', testData.shape, testLabels.shape)
#
else:
    print("[INFO] using single image test mode!")
    print("[INFO] loading features...")
    #data_file = tables.open_file(data_path, mode='r')
    #testData = data_file.root.trainData[202]
    #testData = testData.reshape(
    #                (-1, num_channels, image_height, image_width)).transpose(0,2,3,1).astype(np.float32)
    #testData = testData / 255
    #label_file = tables.open_file(label_path, mode='r')
    #testLabel = label_file.root.trainLabel[202]
    #print('the tested image is a %d' %testLabel)
    #img_names = os.listdir(args["image_path"])
    img_names = glob.glob(args["image_path"] + '*.jpg')
    print('[INFO] number of test images: %d' %len(img_names))
    is_jpg = re.compile(r'.+?\.jpg')
    testData = np.zeros((len(img_names), num_channels, image_height, image_width)).astype(np.float32)
    testLabels = np.zeros((len(img_names),))
    i = 0
    for name in img_names:
        if(is_jpg.match(name)):
            data = np.array(Image.open(name), dtype='float32')
            data = img2directMap(data)
            testData[i] = data
            class_name = int(name.split('/')[-1].split('.')[0].split('_')[0]) - 1
            testLabels[i] = class_name
            i += 1
    testData, testLabels = reformat(testData, testLabels)

# constructing stage
graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_height, image_width, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_dataset = tf.constant(testData)

    # Eight CONV layer variables, in truncated normal distribution.
    conv_weights = []
    conv_biases = []
    conv_weights.append(tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth[0]], stddev=1) / np.sqrt(patch_size * patch_size * num_channels / 2)))
    conv_biases.append(tf.Variable(tf.zeros(depth[0])))
    for i in range(1, 8):
        conv_weights.append(tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth[i-1], depth[i]], stddev=1) / np.sqrt(patch_size * patch_size * depth[i-1] / 2)))
        conv_biases.append(tf.Variable(tf.zeros(depth[i])))

    # Three FC layer variables
    fc_weights = []
    fc_biases = []
    fc_weights.append(tf.Variable(tf.truncated_normal(
      [image_height // 16 * image_width // 16 * depth[7], num_hidden[0]], stddev=1) / np.sqrt((image_height // 16) * (image_width // 16) * depth[7] / 2)))
    fc_biases.append(tf.Variable(tf.zeros(num_hidden[0])))
    
    fc_weights.append(tf.Variable(tf.truncated_normal(
      [num_hidden[0], num_hidden[1]], stddev=1) / np.sqrt(num_hidden[0] / 2)))
    fc_biases.append(tf.Variable(tf.zeros(num_hidden[1])))

    softmax_weights = tf.Variable(tf.truncated_normal(
      [num_hidden[1], num_labels], stddev=1) /np.sqrt(num_hidden[1] / 2))
    softmax_biases = tf.Variable(tf.zeros(num_labels))    
    
    def batchnorm_for_affine(layer):
        D = layer.get_shape()[-1]
        scale = tf.Variable(tf.ones([D]))
        beta = tf.Variable(tf.zeros([D]))
        pop_mean = tf.Variable(tf.zeros([D]), trainable=False)
        pop_var = tf.Variable(tf.ones([D]), trainable=False)
        epsilon = 1e-3
        decay = 0.999
        
        if(args["test_mode"] <= 0):
            batch_mean, batch_var = tf.nn.moments(layer,[0])
            train_mean = tf.assign(pop_mean,
                pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(layer,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(layer,
                pop_mean, pop_var, beta, scale, epsilon)

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
    # Model.
    def model(data):
        # conv layers
        conv1 = tf.nn.conv2d(data, conv_weights[0], [1, 1, 1, 1], padding='SAME') + conv_biases[0]
        conv1 = batchnorm_for_conv(conv1)
        hidden1 = tf.nn.relu(conv1)
        hidden1 = tf.nn.dropout(hidden1, keep_prob[0])

        conv2 = tf.nn.conv2d(hidden1, conv_weights[1], [1, 1, 1, 1], padding='SAME') + conv_biases[1] 
        conv2 = batchnorm_for_conv(conv2)
        hidden2 = tf.nn.relu(conv2)
        hidden2 = tf.nn.dropout(hidden2, keep_prob[1])

        pool1 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        

        conv3 = tf.nn.conv2d(pool1, conv_weights[2], [1, 1, 1, 1], padding='SAME') + conv_biases[2]
        conv3 = batchnorm_for_conv(conv3)
        hidden3 = tf.nn.relu(conv3)
        hidden3 = tf.nn.dropout(hidden3, keep_prob[2])

        conv4 = tf.nn.conv2d(hidden3, conv_weights[3], [1, 1, 1, 1], padding='SAME') + conv_biases[3]
        conv4 = batchnorm_for_conv(conv4)
        hidden4 = tf.nn.relu(conv4)
        hidden4 = tf.nn.dropout(hidden4, keep_prob[3])

        pool2 = tf.nn.max_pool(hidden4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        

        conv5 = tf.nn.conv2d(pool2, conv_weights[4], [1, 1, 1, 1], padding='SAME') + conv_biases[4]
        conv5 = batchnorm_for_conv(conv5)
        hidden5 = tf.nn.relu(conv5)
        hidden5 = tf.nn.dropout(hidden5, keep_prob[4])

        conv6 = tf.nn.conv2d(hidden5, conv_weights[5], [1, 1, 1, 1], padding='SAME') + conv_biases[5]
        conv6 = batchnorm_for_conv(conv6)
        hidden6 = tf.nn.relu(conv6)
        hidden6 = tf.nn.dropout(hidden6, keep_prob[5])

        pool3 = tf.nn.max_pool(hidden6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        

        conv7 = tf.nn.conv2d(pool3, conv_weights[6], [1, 1, 1, 1], padding='SAME') + conv_biases[6]
        conv7 = batchnorm_for_conv(conv7)
        hidden7 = tf.nn.relu(conv7)
        hidden7 = tf.nn.dropout(hidden7, keep_prob[6])

        conv8 = tf.nn.conv2d(hidden7, conv_weights[7], [1, 1, 1, 1], padding='SAME') + conv_biases[7]
        conv8 = batchnorm_for_conv(conv8)
        hidden8 = tf.nn.relu(conv8)
        hidden8 = tf.nn.dropout(hidden8, keep_prob[7])

        pool4 = tf.nn.max_pool(hidden8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

        # fc layers
        shape = pool4.get_shape().as_list()
        reshape = tf.reshape(pool4, [shape[0], shape[1]*shape[2]*shape[3]])
        
        hidden9 = tf.matmul(reshape, fc_weights[0]) + fc_biases[0]
        hidden9 = batchnorm_for_affine(hidden9)
        hidden9 = tf.nn.relu(hidden9)
        hidden9 = tf.nn.dropout(hidden9, keep_prob[8])

        hidden10 = tf.matmul(hidden9, fc_weights[1]) + fc_biases[1]
        hidden10 = batchnorm_for_affine(hidden10)
        hidden10 = tf.nn.relu(hidden10)
        hidden10 = tf.nn.dropout(hidden10, keep_prob[9])

        # output layer
        result = tf.matmul(hidden10, softmax_weights) + softmax_biases

        return result
        
    if(args["test_mode"] <= 0):
        # Training loss and pred computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 3e-4
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
        
        momentum = 0.9
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # check gradients
        #grads_and_vars = optimizer.compute_gradients(loss)
        #grads_compute = optimizer.apply_gradients(grads_and_vars)

    else:
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
        saver = tf.train.Saver()

# running stage
num_epochs = 10
num_iters = 919

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
with tf.Session(graph=graph, config=tfconfig) as session:
    begin = clock()
    if(args["load_model"] > 0):
        print('[INFO] restoring model from file...')
        saver.restore(session, args['model_path'])
        print('[INFO] model restored.')
    else:
        print('[INFO] initializing model from scratch...')
        tf.initialize_all_variables().run()
        print('[INFO] model Initialized.')
    if(args["test_mode"] <= 0):
        for epoch in range(num_epochs):
            #trainData, trainLabels = shuffle(trainData, trainLabels)
            trainIndex = np.random.permutation(919000)
            offset = 0
            for iteration in range(num_iters):
                # stochastic gradient descent
                #batch_index = np.random.choice(trainLabels.shape[0], batch_size)
                #batch_data = trainData[batch_index]
                #batch_labels = trainLabels[batch_index]
                # batch gradient descent
                offset = (iteration * batch_size)
                if(offset + batch_size > 919000):
                    offset = 0
                batch_data = np.zeros((batch_size, 8192))
                batch_labels = np.zeros((batch_size, 1))
                for i in range(batch_size):
                    batch_data[i] = data_file.root.trainData[trainIndex[offset+i]]
                    batch_labels[i] = label_file.root.trainLabel[trainIndex[offset+i]]
                #batch_data = data_file.root.trainData[trainIndex[offset:(offset + batch_size)]]
                #batch_labels = label_file.root.trainLabel[trainIndex[offset:(offset + batch_size)]]
                batch_data, batch_labels = reformat(batch_data, batch_labels)
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                #if(iteration == 100):
                    #np.set_printoptions(threshold=np.nan)
                    #print('[TEST] Softmax weights:')
                    #print(tf.Print(softmax_weights))
                    #print('[TEST] Real labels:')
                    #print(batch_labels[0])
                if (iteration % 100 == 0):
                    np.set_printoptions(threshold=np.nan)
                    #session.run(grads_compute)
                    #for gv in grads_and_vars:
                    #    print(str(see.run(gv[0])) + ' - ' + gv[1].name)
                    #var_grad = tf.gradients(loss, [softmax_weights])[0]
                    #var_grad_val = session.run(var_grad)
                    #print(var_grad_val[:20, 1500:2000])
                    #print('[TEST] Softmax weights:')
                    #weights = session.run(softmax_weights)
                    #print(weights[20, 1500:2000])
                    #print('[TEST] Batch predictions:')
                    #print(predictions[0])
                    print('[INFO] Minibatch loss at epoch %d iteration %d: %f' % (epoch, iteration, l))
                    print('[INFO] Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    print('[INFO] Test accuracy: %.1f%%' % accuracy(test_prediction.eval(session=session), testLabels))
                    if(args["save_model"] > 0):
                        print('[INFO] saving model to file...')
                        save_path = saver.save(session, args["model_path"])
                        print("[INFO] Model saved in file: %s" % save_path)
                    else:
                        print('[INFO] you chose not to save model')
    else:
        print('[INFO] the predicted labels are: %s' %np.argsort(test_prediction.eval(session=session), axis=1)[:, -3:][::-1])
        print('[INFO] the actual labels are: %s' %np.argmax(testLabels, axis=1))
        #print('[INFO] the test accuracy is: %.1f%%' %accuracy(test_prediction.eval(session=session), testLabels) )

end = clock()
print('[INFO] total time used: %f' %(end - begin))
