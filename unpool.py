import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape):
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat([t2, t1], 3)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
 
x = tf.placeholder(tf.float32, shape=(1, None, None, None))

pool_1, argmax_1 = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

pool_2, argmax_2 = tf.nn.max_pool_with_argmax(pool_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

pool_3, argmax_3 = tf.nn.max_pool_with_argmax(pool_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

pool_4, argmax_4 = tf.nn.max_pool_with_argmax(pool_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

pool_5, argmax_5 = tf.nn.max_pool_with_argmax(pool_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

unpool_5 = unpool_layer2x2(pool_5, argmax_5, tf.shape(pool_4))

unpool_4 = unpool_layer2x2(unpool_5, argmax_4, tf.shape(pool_3))

unpool_3 = unpool_layer2x2(unpool_4, argmax_3, tf.shape(pool_2))

unpool_2 = unpool_layer2x2(unpool_3, argmax_2, tf.shape(pool_1))

unpool_1 = unpool_layer2x2(unpool_2, argmax_1, tf.shape(x))

session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

image = np.float32(cv2.imread('./14.jpg'))

tests = [
    pool_1.eval(feed_dict={x: [image]}),
    pool_2.eval(feed_dict={x: [image]}),
    pool_3.eval(feed_dict={x: [image]}),
    pool_4.eval(feed_dict={x: [image]}),
    pool_5.eval(feed_dict={x: [image]}),
    unpool_5.eval(feed_dict={x: [image]}),
    unpool_4.eval(feed_dict={x: [image]}),
    unpool_3.eval(feed_dict={x: [image]}),
    unpool_2.eval(feed_dict={x: [image]}),
    unpool_1.eval(feed_dict={x: [image]})
]

session.close()

for test in tests:
    print(test.shape)

for test in tests:
    plt.figure()
    
    plt.imshow(np.uint8(test[0]), interpolation='none')
    plt.show()
