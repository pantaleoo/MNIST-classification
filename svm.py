import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import _pickle as pickle
import data_read

# 模型保存路径
model_save_path = "model/SVM/model.pkl"

# 加载MNIST数据集
filename_train_images = 'MNIST_data/train-images.idx3-ubyte'
filename_train_labels = 'MNIST_data/train-labels.idx1-ubyte'
filename_test_images = 'MNIST_data/t10k-images.idx3-ubyte'
filename_test_labels = 'MNIST_data/t10k-labels.idx1-ubyte'
train_images = data_read.load_images(filename_train_images)
train_labels = data_read.load_labels(filename_train_labels)
test_images = data_read.load_images(filename_test_images)
test_labels = data_read.load_labels(filename_test_labels)


# 显示预测错误数字图像, 保存至graph/svm_prediction_error.jpg
def show_error_pred(score, pred):
    fig = plt.figure(figsize=(12, 10))
    n = 0
    for i in range(10000):
        if score[i].astype(int) != 1:
            images = np.reshape(test_images[i], [28, 28])
            ax = fig.add_subplot(5, 6, n + 1, xticks=[], yticks=[])
            ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(pred[i]), size='30')
            ax.text(0, 25, str(test_labels[i]), size='30')
            n += 1
        if n == 30:
            break
    plt.savefig('graph/svm_prediction_error.jpg')
    plt.show()


if __name__ == '__main__':
    print("加载模型......")
    svm_model = svm.LinearSVC()
    print("开始训练......")
    svm_model.fit(train_images, train_labels)
    print("开始预测......")
    pred = svm_model.predict(test_images)
    print('准确率:', np.sum(pred == test_labels) / pred.size)
    print("保存模型......")
    with open(model_save_path, 'wb') as file:
        pickle.dump(svm_model, file)
    print("显示预测错误图像......")
    score = np.equal(pred, test_labels)
    show_error_pred(score, pred)