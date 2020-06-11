import numpy as np
import pickle
import data_read
import matplotlib.pyplot as plt


# 模型加载路径
model_path = "model/BAYERS/model.bayers"


filename_train_images = 'MNIST_data/train-images.idx3-ubyte'
filename_train_labels = 'MNIST_data/train-labels.idx1-ubyte'
filename_test_images = 'MNIST_data/t10k-images.idx3-ubyte'
filename_test_labels = 'MNIST_data/t10k-labels.idx1-ubyte'
train_images = data_read.load_images(filename_train_images)
train_labels = data_read.load_labels(filename_train_labels)
test_images = data_read.load_images(filename_test_images)
test_labels = data_read.load_labels(filename_test_labels)


def to_binary(img):
    bin_img = img
    for j in range(bin_img.shape[0]):
        if bin_img[j] > 0.5:
            bin_img[j] = 1
        else:
            bin_img[j] = 0
    return bin_img


# 计算概率
def calculate_probability(img, label):
    probability = int(prior_probability[label])
    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])
    return probability


def predict(testset, prior_probability, conditional_probability):
    predict = []
    for img in testset:
        # 图像二值化
        img = to_binary(img)
        max_label = 0
        max_probability = calculate_probability(img, 0)
        for j in range(1, 10):
            probability = calculate_probability(img, j)
            if max_probability < probability:
                max_label = j
                max_probability = probability
        predict.append(max_label)
    return np.array(predict)


if __name__ == '__main__':
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    prior_probability = data.get('prior_probability')
    conditional_probability = data.get('conditional_probability')
    print("Start predicting")
    test_predict = predict(test_images, prior_probability, conditional_probability)
    score = np.equal(test_predict, test_labels)
    accuracy = np.mean(score.astype(int))

    # 显示预测错误数字图像
    fig = plt.figure(figsize=(12, 10))
    n = 0
    for i in range(10000):
        if score[i].astype(int) != 1:
            images = np.reshape(test_images[i], [28, 28])
            ax = fig.add_subplot(5, 6, n + 1, xticks=[], yticks=[])
            ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(test_predict[i]), size='30')
            ax.text(0, 25, str(test_labels[i]), size='30')
            n += 1
        if n == 30:
            break
    plt.savefig('graph/bayers_prediction_error.jpg')
    plt.show()
    print("The accuracy score is " + str(accuracy))