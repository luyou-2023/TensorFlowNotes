import tensorflow as tf

# 定义输入和参数
# 使用 TensorFlow 2.x 的随机数生成器替代 tf.random_normal
x = tf.constant([[0.7, 0.5]])  # 分别对应物体的重量和体积
w1 = tf.random.normal(shape=(2, 3), stddev=1.0, seed=1)  # 第1层的权重
w2 = tf.random.normal(shape=(3, 1), stddev=1.0, seed=1)  # 第2层的权重

# 定义前向传播过程
a = tf.matmul(x, w1)  # 计算第1层的输出
y = tf.matmul(a, w2)  # 计算第2层的输出

# 打印结果
print("y:", y)
