import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import load_data
import cv2

tf.reset_default_graph()

batch_size = 32


def visualize_generated(data_np):
    print("Visualizing")
    vis = data_np[0]
    print(vis.shape)
    vis = 255 * (vis + 1) / 2
    print(vis)
    # cv2.imshow("im", vis)
    # cv2.waitKey(0)
    cv2.imwrite("generated.png", vis)


input_batch = tf.placeholder(tf.float32, shape=[batch_size, 200, 200, 1])
noise = tf.placeholder(tf.float32, shape=[batch_size, 10])  # noise dims


def leaky_relu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


def generator_model():  # architecture of generator
    with tf.variable_scope("generator"):
        net = tf.layers.dense(noise, 1024)
        net = tf.layers.dense(net, 12 * 12 * 64)
        net = tf.reshape(net, [-1, 12, 12, 64])
        net = tf.layers.conv2d_transpose(net, 64, kernel_size=[4, 4], strides=[2, 2])
        net = tf.layers.conv2d_transpose(net, 32, kernel_size=[4, 4], strides=[2, 2])
        net = tf.layers.conv2d_transpose(net, 32, kernel_size=[4, 4], strides=[2, 2])
        net = tf.layers.conv2d_transpose(net, 16, kernel_size=[4, 4], strides=[2, 2])
        net = tf.layers.conv2d(net, 1, 4, activation=tf.tanh)
        return net


def discriminator_model(im, reuse=None):  # architecture of discriminator
    keep_prob = 0.6
    activation = leaky_relu
    with tf.variable_scope("discriminator", reuse=reuse):
        net = tf.layers.conv2d(im, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        net = tf.layers.dropout(net, keep_prob)
        net = tf.layers.conv2d(net, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        net = tf.layers.dropout(net, keep_prob)
        net = tf.layers.conv2d(net, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        net = tf.layers.dropout(net, keep_prob)
        net = tf.keras.layers.GlobalAveragePooling2D('channels_last')(net)
        net = tf.layers.dense(net, 128, activation=activation)
        net = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)
        return net


generator = generator_model()
discriminator_real = discriminator_model(input_batch)
discriminator_fake = discriminator_model(generator, reuse=True)

generator_variables = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
discriminator_variables = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

generator_regularizer = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6),
                                                               generator_variables)
discriminator_regularizer = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6),
                                                                   discriminator_variables)

generator_loss = tf.reduce_mean(
    binary_cross_entropy(tf.ones_like(discriminator_fake), discriminator_fake)) + generator_regularizer
loss_d_real = binary_cross_entropy(tf.ones_like(discriminator_real), discriminator_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(discriminator_fake), discriminator_fake)
discriminator_loss = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake)) + discriminator_regularizer

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.005).minimize(discriminator_loss,
                                                                            var_list=discriminator_variables)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.005).minimize(generator_loss,
                                                                            var_list=generator_variables)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_data = load_data.load()

for epoch_i in range(100):
    print("\n Epoch ", epoch_i)
    for index in range(len(train_data) // batch_size):
        print("\t Batch ", index, "/", len(train_data) // batch_size)
        data = train_data[index:index + batch_size]
        noise_vals = np.random.uniform(-1, 1, noise.get_shape())
        print(noise_vals.shape)
        d_loss_, _ = sess.run([discriminator_loss, optimizer_d], feed_dict={noise: noise_vals, input_batch: data})
        generated_im_, g_loss_, _ = sess.run([generator, generator_loss,optimizer_g], feed_dict={noise: noise_vals, input_batch: data})
        visualize_generated(generated_im_)
        print("discriminator loss: ", d_loss_, "; Generated loss: ", g_loss_)
