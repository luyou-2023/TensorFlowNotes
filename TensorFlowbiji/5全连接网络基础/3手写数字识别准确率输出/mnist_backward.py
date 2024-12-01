import tensorflow as tf
import ssl
import os
import numpy as np

# 定义超参数
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = tf.keras.regularizers.l2(0.0001)  # 使用 l2 正则化
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'

# 定义网络结构参数
INPUT_NODE = 784  # 输入节点数目
LAYER1_NODE = 500  # 第一层节点数目
OUTPUT_NODE = 10  # 输出节点数目
ssl._create_default_https_context = ssl._create_unverified_context

# 使用 keras.Sequential 代替手动构建网络
class MnistModel(tf.keras.Model):
    def __init__(self, regularizer=None):
        super(MnistModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(LAYER1_NODE, activation='relu', kernel_regularizer=regularizer)
        self.dense2 = tf.keras.layers.Dense(OUTPUT_NODE, activation=None, kernel_regularizer=regularizer)

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)

def get_weight(shape, regularizer):
    # 使用 tf.random.normal 替代 tf.truncated_normal
    w = tf.Variable(tf.random.normal(shape, stddev=0.1))  # 正态分布初始化
    if regularizer is not None:
        weight_loss = regularizer(w)  # 正则化
        return w, weight_loss  # 返回权重和正则化损失
    return w, None  # 如果没有正则化，返回权重和None

def forward(x, regularizer):
    # 第一层全连接
    w1, weight_loss1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = tf.Variable(tf.zeros([LAYER1_NODE]))  # 偏置初始化
    z1 = tf.matmul(x, w1) + b1
    a1 = tf.nn.relu(z1)  # 激活函数

    # 第二层全连接
    w2, weight_loss2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = tf.Variable(tf.zeros([OUTPUT_NODE]))  # 偏置初始化
    z2 = tf.matmul(a1, w2) + b2

    total_loss = 0.0
    if weight_loss1 is not None:
        total_loss += weight_loss1  # 加上第一层的正则化损失
    if weight_loss2 is not None:
        total_loss += weight_loss2  # 加上第二层的正则化损失

    return z2, total_loss  # 返回logits和总损失

def backward(mnist_data):
    # 解包数据集
    (train_images, train_labels), (test_images, test_labels) = mnist_data

    # 将图像数据平铺成一维向量，784 是 28*28
    x = tf.keras.Input(shape=(784,))
    y_ = tf.keras.Input(shape=(10,))

    # 网络的前向传播
    y, total_loss = forward(x, REGULARIZER)

    # 计算交叉熵损失
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + total_loss  # 加上正则化损失

    # 学习率衰减
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_images.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    # 使用 GradientTape 计算梯度
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # 建立模型
    model = MnistModel(regularizer=REGULARIZER)

    for i in range(STEPS):
        batch_index = np.random.choice(train_images.shape[0], BATCH_SIZE)
        xs, ys = train_images[batch_index], train_labels[batch_index]
        xs = xs.reshape([-1, 784])  # 将图像展平为一维向量

        xs = tf.Variable(xs, dtype=tf.float32)
        ys = tf.Variable(ys, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(xs)
            y = model(xs)  # 通过模型前向传播计算输出
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(ys, 1))
            cem = tf.reduce_mean(ce)
            loss_value = cem  # 没有需要手动加正则化损失，已经在模型中处理

        gradients = tape.gradient(loss_value, model.trainable_variables)  # 获取模型的梯度

        # 使用梯度下降优化
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % 1000 == 0:
            print(f"After {i} training step(s), loss on training batch is {loss_value.numpy()}")

    # 保存模型
    model.save(os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

def main():
    # 加载 MNIST 数据集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # 将数据归一化处理
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 将标签转化为 one-hot 编码
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # 正确传递数据
    mnist_data = ((train_images, train_labels), (test_images, test_labels))

    # 开始训练
    backward(mnist_data)

if __name__ == '__main__':
    main()
