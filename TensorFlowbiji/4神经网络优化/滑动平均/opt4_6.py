import tensorflow as tf

# 1. 定义变量及滑动平均类
w1 = tf.Variable(0.0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)

MOVING_AVERAGE_DECAY = 0.99
# 使用 tf.train.ExponentialMovingAverage 替代并用 tf.keras.optimizers 更新滑动平均
ema = tf.keras.layers.experimental.preprocessing.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)

# 2. 使用 Eager Execution 查看不同迭代中变量的变化
# 直接计算值，不需要通过 Session
def print_values():
    print(f"w1: {w1.numpy()} , ema.average(w1): {ema(w1).numpy()}")

# 初始化变量并进行测试
print_values()

# 更新 w1 的值为 1
w1.assign(1.0)
ema.apply([w1])  # 更新滑动平均
print_values()

# 设置 global_step 并更新 w1 的值为 10
global_step.assign(100)
w1.assign(10.0)
ema.apply([w1])  # 更新滑动平均
print_values()

# 多次运行，观察滑动平均值变化
for i in range(7):
    ema.apply([w1])
    print(f"第{i+1}次运行: ")
    print_values()

# 3. 测试快速追随
MOVING_AVERAGE_DECAY = 0.1
ema_fast = tf.keras.layers.experimental.preprocessing.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)

# 初始化并进行测试
w1.assign(1.0)
ema_fast.apply([w1])
print_values()

w1.assign(10.0)
ema_fast.apply([w1])
print_values()

for i in range(7):
    ema_fast.apply([w1])
    print(f"第{i+1}次运行: ")
    print_values()
