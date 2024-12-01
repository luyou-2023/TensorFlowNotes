import tensorflow as tf
import numpy as np

# 超参数
BATCH_SIZE = 8
SEED = 23455

# 随机数生成器
rng = np.random.RandomState(SEED)
X = rng.rand(32, 2)
Y = np.array([[int(x1 + x2 < 1)] for (x1, x2) in X], dtype=np.float32)

# 转换为 TensorFlow 张量
X = tf.constant(X, dtype=tf.float32)
Y = tf.constant(Y, dtype=tf.float32)

# 定义模型参数
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
def forward_propagation(x):
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    return y

# 定义损失函数
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # 均方误差

# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

# 训练模型
STEPS = 3000
for step in range(STEPS):
    start = (step * BATCH_SIZE) % 32
    end = start + BATCH_SIZE
    X_batch = X[start:end]
    Y_batch = Y[start:end]

    # 梯度计算和优化
    with tf.GradientTape() as tape:
        Y_pred = forward_propagation(X_batch)
        loss = compute_loss(Y_pred, Y_batch)
    grads = tape.gradient(loss, [w1, w2])
    optimizer.apply_gradients(zip(grads, [w1, w2]))

    # 每 500 步打印一次损失
    if step % 500 == 0:
        total_loss = compute_loss(forward_propagation(X), Y)
        print(f"After {step} training steps, loss on all data is {total_loss.numpy()}")

# 输出训练后参数的值
print("\n训练结束后参数：")
print("w1:\n", w1.numpy())
print("w2:\n", w2.numpy())
