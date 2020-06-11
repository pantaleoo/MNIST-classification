from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 参数设置
n_input = 784
n_classes = 10
epoch = 20000
learning_rate = 1e-4
batch_size = 50
display_step = 100

# 路径
LOG_DIR = 'cnn_log'
model_path = "model/CNN/cnn_model.ckpt"


def get_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="w_conv")


def get_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b_conv")


def convolution2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name="h_conv")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name="h_pool")

with tf.name_scope("input"):
    x_ = tf.placeholder(tf.float32, [None, n_input], name="input_x")
    y_ = tf.placeholder(tf.float32, [None, n_classes], name="input_y")
x_image = tf.reshape(x_, [-1, 28, 28, 1])

with tf.name_scope("layer1"):
    w_conv1 = get_weight_variable([5, 5, 1, 32])
    tf.compat.v1.summary.histogram('layer1/weights', w_conv1)
    b_conv1 = get_bias_variable([32])
    tf.compat.v1.summary.histogram('layer1/bias', b_conv1)
    h_conv1 = tf.nn.relu(convolution2d(x_image, w_conv1) + b_conv1, name="relu")
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("layer2"):
    w_conv2 = get_weight_variable([5, 5, 32, 64])
    tf.compat.v1.summary.histogram('layer2/weights', w_conv2)
    b_conv2 = get_bias_variable([64])
    tf.compat.v1.summary.histogram('layer2/bias', b_conv2)
    h_conv2 = tf.nn.relu(convolution2d(h_pool1, w_conv2) + b_conv2, name="relu")
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("layer3"):
    w_fc1 = get_weight_variable([7 * 7 * 64, 1024])
    tf.compat.v1.summary.histogram('layer3/weights', w_fc1)
    b_fc1 = get_bias_variable([1024])
    tf.compat.v1.summary.histogram('layer3/bias', b_fc1)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name="relu")

with tf.name_scope("layer4_softmax"):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="dropout")
    w_fc2 = get_weight_variable([1024, 10])
    tf.compat.v1.summary.histogram('layer4/weights', w_fc2)
    b_fc2 = get_bias_variable([10])
    tf.compat.v1.summary.histogram('layer4/bias', b_fc2)
    pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="softmax")

cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(pred), reduction_indices=[1]))
tf.compat.v1.summary.scalar('Loss', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.name_scope("predict"):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1), name="correct_prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


sess = tf.InteractiveSession()

merged = tf.compat.v1.summary.merge_all()
writer = tf.compat.v1.summary.FileWriter(LOG_DIR, sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
tf.add_to_collection("prediction", pred)
tf.add_to_collection("accuracy", accuracy)

with tf.name_scope("train"):
    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)
        if i % display_step == 0:
            rs, train_accuracy = sess.run([merged, accuracy], feed_dict={
                x_: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d/%d, training accuracy %g" % (i, epoch, train_accuracy))
            writer.add_summary(rs, i)
        optimizer.run(feed_dict={x_: batch[0], y_: batch[1], keep_prob: 0.5})
saver.save(sess, save_path=model_path)


writer.close()
print("test accuracy %g" % accuracy.eval(feed_dict={
    x_: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
