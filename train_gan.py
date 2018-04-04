import matplotlib.pyplot as plt
import tensorflow as tf
import visualize_images
import preprocess_image
import numpy as np
import load_data
import cv2

tf.reset_default_graph()

batch_size = 32


def visualize_generated(vis):
    print("Visualizing")
    for i in range(len(vis)):
        vis[i] = ((vis[i] + 1) / 2) * 255.0
        cv2.imwrite("./generated/generated%d.png" % i, vis[i])


input_batch = tf.placeholder(tf.float32, shape=[batch_size, 189, 189, 1])
noise = tf.placeholder(tf.float32, shape=[batch_size, 12*12*3])  # noise dims
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)


def leaky_relu(x):
    return tf.maximum(x, tf.multiply(x, 0.02))  # lrelu percentage


def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


def generator_model():  # architecture of generator - remove bias?
    activation = leaky_relu
    with tf.variable_scope("generator"):
        print("Generator Shapes")
        net = tf.reshape(noise, [-1, 12, 12, 3])
        print(net.get_shape())
        net = tf.layers.conv2d_transpose(net, 64, kernel_size=5, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
        net = tf.layers.conv2d_transpose(net, 64, kernel_size=5, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
        net = tf.layers.conv2d_transpose(net, 64, kernel_size=5, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
        net = tf.layers.conv2d_transpose(net, 32, kernel_size=5, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        # net = tf.contrib.layers.batch_norm(net, is_training=is_training)
        net = tf.layers.conv2d(net, 1, 4, activation=tf.nn.tanh)
        print(net.get_shape())
        return net


def discriminator_model(im, reuse=None):  # architecture of discriminator
    activation = leaky_relu
    with tf.variable_scope("discriminator", reuse=reuse):
        print("Discriminator Shapes")
        print(im.get_shape())
        net = tf.layers.conv2d(im, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.layers.dropout(net, keep_prob)
        net = tf.layers.conv2d(net, kernel_size=5, filters=32, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.layers.dropout(net, keep_prob)
        net = tf.layers.conv2d(net, kernel_size=5, filters=16, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.layers.dropout(net, keep_prob)
        net = tf.layers.conv2d(net, kernel_size=5, filters=16, strides=2, padding='same', activation=activation)
        print(net.get_shape())
        net = tf.layers.dropout(net, keep_prob)
        net = tf.contrib.layers.flatten(net)
        print(net.get_shape())
        net = tf.layers.dense(net, 128, activation=activation)
        print(net.get_shape())
        net = tf.layers.dropout(net, keep_prob)
        net = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)
        print(net.get_shape())
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
    binary_cross_entropy(tf.ones_like(discriminator_fake),
                                            discriminator_fake)) + generator_regularizer
loss_d_real = binary_cross_entropy(tf.ones_like(discriminator_real),
                                                      discriminator_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(discriminator_fake),
                                                      discriminator_fake)
discriminator_loss = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake)) + discriminator_regularizer

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(discriminator_loss,
                                                                        var_list=discriminator_variables)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(generator_loss,
                                                                        var_list=generator_variables)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_data = load_data.load()

train_d, train_g = True, True

for epoch_i in range(1000):
    train_data = preprocess_image.preprocess_image(train_data)
    print("\n Epoch ", epoch_i)
    for index in range(len(train_data) // batch_size):
        print("\t Batch ", index, "/", len(train_data) // batch_size)
        print("train g? ", train_g, "; train d? ", train_d)
        data = train_data[index:(index + batch_size)]
        noise_vals = np.random.uniform(0, 1, noise.get_shape())
        if train_d:
            d_real_preds, d_fake_preds, d_loss_, _ = sess.run(
                [discriminator_real, discriminator_fake, discriminator_loss, optimizer_d],
                feed_dict={noise: noise_vals, input_batch: data, keep_prob: 0.7, is_training: True})
            print("discriminator loss: ", d_loss_)
        if train_g:
            d_fake_preds, generated_im_, g_loss_, _ = sess.run(
                [discriminator_fake, generator, generator_loss, optimizer_g],
                feed_dict={noise: noise_vals, input_batch: data, keep_prob: 0.7, is_training: True})
            print("discriminator loss: ", d_loss_, "; Generated loss: ", g_loss_)

        generated_im_ = sess.run([generator],
                                 feed_dict={noise: noise_vals, input_batch: data, keep_prob: 1.0, is_training: False})
        generated_im_ = np.squeeze(generated_im_, 0)
        visualize_generated(generated_im_)

        if g_loss_ * 1.5 < d_loss_:
            train_g = False
        else:
            train_g = True
        if d_loss_ * 2 < g_loss_:
            train_d = False
        else:
            train_d = True
