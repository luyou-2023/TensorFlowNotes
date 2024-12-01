import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
labels = ['cat', 'dog', 'fish', 'bird', 'horse']

# 在 TensorFlow 2.0 中，默认启用了 Eager Execution，所以不需要显式地使用 Session
# 读取图片路径
img_path = input('Input the path and image name:')  # 在 Python 3.x 中使用 input 而不是 raw_input
img_ready = utils.load_image(img_path)

# 绘制Top-5预测结果的条形图
fig = plt.figure("Top-5 Prediction Results")

# 使用 TensorFlow 2.x 的方式加载模型并预测
# 加载VGG16模型
vgg = vgg16.Vgg16()
image = tf.convert_to_tensor(img_ready, dtype=tf.float32)  # 转换为 TensorFlow 张量
image = tf.expand_dims(image, axis=0)  # 增加 batch 维度 [1, 224, 224, 3]

# 计算模型的概率输出
probability = vgg.forward(image)

# 获取 Top 5 的预测结果
top5 = np.argsort(probability[0])[-1:-6:-1]
print("Top 5:", top5)

values = []
bar_label = []

# 打印并记录 Top 5 的预测标签及概率值
for n, i in enumerate(top5):
    print(f"n: {n}, i: {i}")
    values.append(probability[0][i])
    bar_label.append(labels[i])
    print(f"{i}: {labels[i]} ----- {utils.percent(probability[0][i])}")

# 绘制条形图
ax = fig.add_subplot(111)
ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
ax.set_ylabel('Probability')
ax.set_title('Top-5')

# 在条形图上显示数值
for a, b in zip(range(len(values)), values):
    ax.text(a, b + 0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)

# 显示图表
plt.show()
