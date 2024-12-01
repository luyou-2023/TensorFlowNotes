import tensorflow as tf

# 定义待优化参数 w，并初始化为 5
w = tf.Variable(5.0, dtype=tf.float32)

# 定义损失函数 loss = (w + 1)^2
def compute_loss():
    return tf.square(w + 1)

# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

# 训练 40 轮
for i in range(40):
    # 自动计算梯度并优化
    with tf.GradientTape() as tape:
        loss = compute_loss()
    grads = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(grads, [w]))

    # 打印当前 w 和 loss 的值
    print(f"After {i} steps: w is {w.numpy():.4f}, loss is {loss.numpy():.4f}.")
