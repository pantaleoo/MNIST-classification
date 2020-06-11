from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载用于生成PROJECTOR日志的帮助函数
from tensorflow.contrib.tensorboard.plugins import projector
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

TRAINING_STEPS = 10000
LOG_DIR = 'tensorboard_log'
TENSOR_NAME = "MNIST_FINAL_LOGITS"
SPRITE_FILE = 'mnist_sprite.jpg'
META_FIEL = "mnist_meta.tsv"


def visualisation(final_result):
    # 定义变量保存输出层向量的取值
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.compat.v1.summary.FileWriter(LOG_DIR)
    # 生成日志文件
    config = projector.ProjectorConfig()
    # 增加需要可视化的bedding结果
    embedding = config.embeddings.add()
    # 指定embedding对应的Tensorflow变量名称
    embedding.tensor_name = y.name
    # 指定embedding结果对应的原始数据信息
    embedding.metadata_path = META_FIEL
    # 指定sprite 图像
    embedding.sprite.image_path = SPRITE_FILE
    # 截取原始图片。
    embedding.sprite.single_image_dim.extend([28, 28])
    # 写入日志文件。
    projector.visualize_embeddings(summary_writer, config)
    # 生成会话，初始化新声明的变量并将需要的日志信息写入文件。
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()


def get_pred():
    with tf.compat.v1.Session() as sess:
        new_saver = tf.compat.v1.train.import_meta_graph("model/CNN/cnn_model.ckpt.meta")
        new_saver.restore(sess, "model/CNN/cnn_model.ckpt")
        pred = tf.compat.v1.get_collection("prediction")[0]
        acc = tf.compat.v1.get_collection("accuracy")[0]

        graph = tf.compat.v1.get_default_graph()
        x = graph.get_operation_by_name("input/input_x").outputs[0]
        y = graph.get_operation_by_name("input/input_y").outputs[0]
        keep_prob = graph.get_operation_by_name("layer4_softmax/keep_prob").outputs[0]

        train_images = mnist.train.images
        train_labels = mnist.train.labels
        test_images = mnist.test.images
        test_labels = mnist.test.labels
        labels = np.argwhere(test_labels == 1)[:, 1]

        accuracy, prediction = sess.run([acc, pred],
                                        feed_dict={x: test_images, y: test_labels,
                                                   keep_prob: 0.5})
        print("test set accuracy: " + str(accuracy))
        result = prediction
        score = tf.equal(tf.argmax(prediction, 1), tf.argmax(test_labels, 1))
        prediction = np.argwhere(prediction == 1)[:, 1]
        fig = plt.figure(figsize=(12, 10))
        n = 0
        for i in range(10000):
            if score[i].eval().astype(int) != 1:
                print(score[i].eval())
                images = np.reshape(test_images[i], [28, 28])
                ax = fig.add_subplot(5, 6, n + 1, xticks=[], yticks=[])
                ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
                ax.text(0, 7, str(prediction[i]), size='30')
                ax.text(0, 25, str(labels[i]), size='30')
                n += 1
            if n == 30:
                break
        plt.savefig('graph/cnn_prediction_error.jpg')
        plt.show()
        result = np.asarray(result)
    return result

def main(argv=None):
    final_result = get_pred()
    print(final_result.shape)
    visualisation(final_result)

if __name__ == '__main__':
        main()