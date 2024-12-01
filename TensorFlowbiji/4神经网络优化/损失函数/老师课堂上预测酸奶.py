import tensorflow as tf
import numpy as np

# 参数定义
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9
STEPS = 20000

# 数据生成
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y = np.array([[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X])

# 构建模型
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w1)

model = SimpleModel()

# 定义损失函数
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.where(tf.greater(y_pred, y_true), (y_pred - y_true) * COST, (y_true - y_pred) * PROFIT)
    )

# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

# 训练过程
for step in range(STEPS):
    start = (step * BATCH_SIZE) % 32
    end = start + BATCH_SIZE
    X_batch, Y_batch = X[start:end], Y[start:end]

    with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = custom_loss(Y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if step % 500 == 0:
        print(f"After {step} steps, w1 is:\n{model.w1.numpy()}")

# 输出最终权重
print("Final w1 is:\n", model.w1.numpy())
