import pickle
import data_read
import numpy as np


class_num = 10
feature_length = 784
model_save_path = "model/BAYERS/model.bayers"


# 加载MNIST数据集
filename_train_images = 'MNIST_data/train-images.idx3-ubyte'
filename_train_labels = 'MNIST_data/train-labels.idx1-ubyte'
filename_test_images = 'MNIST_data/t10k-images.idx3-ubyte'
filename_test_labels = 'MNIST_data/t10k-labels.idx1-ubyte'
train_images = data_read.load_images(filename_train_images)
train_labels = data_read.load_labels(filename_train_labels)
test_images = data_read.load_images(filename_test_images)
test_labels = data_read.load_labels(filename_test_labels)


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# 图像转为二值图
def to_binary(img):
    bin_img = img
    for i in range(bin_img.shape[0]):
        if bin_img[i] > 0.5:
            bin_img[i] = 1
        else:
            bin_img[i] = 0
    return bin_img


# 训练
def train(images, labels):
    priori_probability = np.zeros(class_num)  # 先验概率
    condition_probability = np.zeros([class_num, feature_length, 2])
    # 计算先验概率和条件概率
    for i in range(images.shape[0]):
        img = to_binary(images[i, :])
        label = labels[i]
        priori_probability[label] += 1
        for j in range(feature_length):
            condition_probability[label][j][img[j]] += 1

    # 将概率归到[1.10001]
    for i in range(class_num):
        for j in range(feature_length):
            pix_0 = condition_probability[i][j][0]
            pix_1 = condition_probability[i][j][1]
            # 计算条件概率
            probability_0 = (float(pix_0) / float(pix_0 + pix_1)) * 1000000 + 1
            probability_1 = (float(pix_1) / float(pix_0 + pix_1)) * 1000000 + 1
            condition_probability[i][j][0] = probability_0
            condition_probability[i][j][1] = probability_1

    return priori_probability, condition_probability


# 计算概率
def calculate_probability(img, label):
    probability = int(prior_probability[label])
    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])
    return probability


# 预测准确率
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


# 保存模型
def model_save():
    with open(model_save_path, "wb") as file:
        data = {'accuracy_score': accuracy, 'prior_probability': prior_probability,
                'conditional_probability': conditional_probability}
        pickle.dump(data, file)


if __name__ == '__main__':
    print("开始训练......")
    prior_probability, conditional_probability = train(train_images, train_labels)
    print("开始预测......")
    test_predict = predict(test_images, prior_probability, conditional_probability)
    score = np.equal(test_predict, test_labels)
    accuracy = np.mean(score.astype(int))
    print("准确率：" + str(accuracy))
    print("开始保存模型......")
    model_save()
    print("完成")
