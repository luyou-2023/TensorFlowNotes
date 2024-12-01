import tensorflow as tf

# 定义输入数据和参数
# 使用张量而非占位符，直接定义输入
x = tf.constant([[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]], dtype=tf.float32)  # 输入数据
w1 = tf.Variable(tf.random.normal(shape=(2, 3), stddev=1, seed=1))  # 第1层权重
w2 = tf.Variable(tf.random.normal(shape=(3, 1), stddev=1, seed=1))  # 第2层权重

# 定义前向传播过程
a = tf.matmul(x, w1)  # 第1层输出
y = tf.matmul(a, w2)  # 第2层输出

# 初始化变量并打印结果
# TensorFlow 2.x 默认启用 Eager Execution，直接运行计算
tf.keras.backend.set_floatx('float32')  # 确保默认数据类型为 float32

# 初始化变量
tf.keras.initializers.GlorotUniform()  # 权重初始化（如需自定义初始化）

# 打印结果
print("y is:\n", y.numpy())  # 打打印。 default is block-numerix-run = context [multi-thread ] O__'' ‘’
