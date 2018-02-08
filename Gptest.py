import tensorflow as tf
import numpy as np
import random
import datetime

Grad_clip = True
tf.logging.set_verbosity(tf.logging.INFO)

input_ph = tf.placeholder(dtype=tf.float32, shape=[5,32,32,1])
output = tf.placeholder(dtype=tf.float32, shape=[5,5])


conv1 = tf.layers.conv2d(inputs=input_ph, filters=32, kernel_size=[5,5], padding='SAME', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=dense, units=5)


loss =tf.nn.softmax_cross_entropy_with_logits_v2(labels=output, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

if Grad_clip is True:
    grads = tf.gradients(loss, tf.trainable_variables())
    omnomed_grads, _ = tf.clip_by_global_norm(grads, 0.5)
    train_op = optimizer.apply_gradients(zip(omnomed_grads,  tf.trainable_variables()))
else:
    train_op = optimizer.minimize(loss)


config = tf.ConfigProto(log_device_placement=True)
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    iter_start_time = datetime.datetime.now()
    for i in range(10000):
        session.run([train_op], feed_dict={input_ph:np.random.random_sample([5, 32, 32, 1]),
                                          output:np.eye(5)[np.random.random_integers(0,4, size=5)]})
    print("Grad_Clip:", Grad_clip, "|| Total time for 10000 iterations:", datetime.datetime.now() - iter_start_time)