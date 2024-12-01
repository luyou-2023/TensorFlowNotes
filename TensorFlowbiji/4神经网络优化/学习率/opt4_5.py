import tensorflow as tf

# 超参数
LEARNING_RATE_BASE = 0.1    # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1      # 每多少步衰减一次学习率

# 初始化待优化参数 w，初值为 10
w = tf.Variable(10.0, dtype=tf.float32)

# 定义指数衰减学习率
global_step = tf.Variable(0, trainable=False)  # 计数器
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE_BASE,
    decay_steps=LEARNING_RATE_STEP,
    decay_rate=LEARNING_RATE_DECAY,
    staircase=True
)

# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# 定义损失函数
def compute_loss():
    return tf.square(w + 1)

# 训练 40 轮
for step in range(40):
    with tf.GradientTape() as tape:
        loss = compute_loss()
    grads = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(grads, [w]))

    # 打印当前状态
    print(f"After {step + 1} steps: global_step is {step + 1}, "
          f"w is {w.numpy():.4f}, learning rate is {learning_rate(step):.4f}, loss is {loss.numpy():.4f}.")
