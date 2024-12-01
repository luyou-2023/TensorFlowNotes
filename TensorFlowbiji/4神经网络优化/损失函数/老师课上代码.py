import tensorflow as tf
import numpy as np

# 参数定义
BATCH_SIZE = 8
SEED = 23455
STEPS = 80000

# 数据生成
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)  # 输入
Y_ = np.array([[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X])  # 输出

# 定义模型
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w1)

model = SimpleModel()

# 损失函数和优化器
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

# 训练过程
for step in range(STEPS):
    start = (step * BATCH_SIZE) % 32
    end = start + BATCH_SIZE
    X_batch = X[start:end]
    Y_batch = Y_[start:end]

    with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = loss_fn(Y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 每隔 500 步打印权重和损失
    if step % 500 == 0:
        print(f"Step {step}, w1: \n{model.w1.numpy()}, loss: {loss.numpy()}")

# 最终权重
print("Final w1: \n", model.w1.numpy())
