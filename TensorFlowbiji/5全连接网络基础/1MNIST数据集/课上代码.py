import tensorflow as tf
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

# 使用 tf.keras.datasets 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# MNIST 数据集加载完成后，打印训练集和测试集的样本数
print(f"Training samples: {train_images.shape[0]}")  # 训练集样本数
print(f"Test samples: {test_images.shape[0]}")  # 测试集样本数

# 查看训练集中的标签（第一个标签）
print(f"Train label (first example): {train_labels[0]}")  # 查看训练集中第一个标签

# 查看训练集中第一个图像的像素值（784个像素点）
print(f"Train image (first example): \n{train_images[0]}")  # 784个像素点

# 取一小批数据，准备输入到神经网络
BATCH_SIZE = 200
train_images_batch = train_images[:BATCH_SIZE]
train_labels_batch = train_labels[:BATCH_SIZE]

# 打印批量数据的形状
print(f"train_images_batch shape: {train_images_batch.shape}")
print(f"train_labels_batch shape: {train_labels_batch.shape}")
