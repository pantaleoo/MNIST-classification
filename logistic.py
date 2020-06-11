import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

# 训练参数
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 5

# 设置x, y占位符，w, b为参数
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 逻辑回归
logistic = tf.nn.softmax(tf.matmul(x, w) + b)
# 代价函数, 使用交叉熵损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(logistic), reduction_indices=1))
# 使用梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 预测结果
prediction = tf.equal(tf.argmax(logistic, 1), tf.argmax(y, 1))
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
# 初始化变量
init = tf.global_variables_initializer()

# 开启会话
sess = tf.Session()
sess.run(init)

# 绘图所需x, y
x_epoch = np.zeros(shape=[int(training_epochs)])
x2_epoch = np.zeros(shape=[int(training_epochs)])
y_cost = np.zeros(shape=[int(training_epochs)])
y_train_acc = np.zeros(shape=[int(training_epochs)])
y_test_acc = np.zeros(shape=[int(training_epochs)])
m = 0

# 开始训练
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds_train = {x: batch_xs, y: batch_ys}
        sess.run(optimizer, feed_dict=feeds_train)
        avg_cost += sess.run(cost, feed_dict=feeds_train) / num_batch
    # 绘图所需值
    feeds_test = {x: mnist.test.images, y: mnist.test.labels}
    train_acc = sess.run(accuracy, feed_dict=feeds_train)
    test_acc = sess.run(accuracy, feed_dict=feeds_test)
    x_epoch[m] = epoch
    y_cost[m] = avg_cost
    y_train_acc[m] = train_acc
    y_test_acc[m] = test_acc
    m += 1
    # 训练过程输出
    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f train accuracy: %.3f test accuracy: %.3f"
                % (epoch, training_epochs, avg_cost, train_acc, test_acc))

# 开始绘图, 横轴为epoch, 纵轴为average cost, train accuracy, test_accuracy.
# 保存至graph/logistic.jpg
l1 = plt.plot(x_epoch, y_cost, 'r-', label='avg_cost')
l2 = plt.plot(x_epoch, y_train_acc, 'g-', label='train_accuracy')
l3 = plt.plot(x_epoch, y_test_acc, 'b-', label='test_accuracy')
plt.title('logistic regression')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend()
plt.savefig('graph/logistic.jpg')
plt.show()
print("success")