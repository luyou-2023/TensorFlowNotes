# coding:utf-8

# 0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8  # 每次向神经网络喂入多少组数据，不宜过大
seed = 23455    # 为了教学方便，设置随机种子，确保每个人随机生成的数据是一致的，便于调试

# 基于seed产生随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rng.rand(32, 2)
# 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0
# 作为输入数据集的标签(正确答案)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n", X)
print("Y:\n", Y)

# 1定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.constant(X, dtype=tf.float32)  # 定义输入数据
y_true = tf.constant(Y, dtype=tf.float32)  # 定义标签数据

# 定义可训练变量（权重）
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
def forward_propagation(x):
    a = tf.matmul(x, w1)
    y_pred = tf.matmul(a, w2)
    return y_pred

# 2定义损失函数和优化器
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # 均方误差

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)  # 使用随机梯度下降优化器

# 3训练模型
STEPS = 3000
for i in range(STEPS):
    start = (i * BATCH_SIZE) % 32
    end = start + BATCH_SIZE
    x_batch = x[start:end]
    y_batch = y_true[start:end]

    # 梯度下降优化
    with tf.GradientTape() as tape:
        y_pred = forward_propagation(x_batch)
        loss = compute_loss(y_pred, y_batch)
    grads = tape.gradient(loss, [w1, w2])
    optimizer.apply_gradients(zip(grads, [w1, w2]))

    # 每隔500步输出一次损失
    if i % 500 == 0:
        total_loss = compute_loss(forward_propagation(x), y_true)
        print(f"After {i} training step(s), loss on all data is {total_loss.numpy()}")

# 训练结束后输出权重
print("\n训练结束后权重值：")
print("w1:\n", w1.numpy())
print("w2:\n", w2.numpy())
