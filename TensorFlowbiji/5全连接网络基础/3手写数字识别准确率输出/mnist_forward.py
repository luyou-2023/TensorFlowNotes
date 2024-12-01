import tensorflow as tf

# 定义网络结构的超参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight(shape, regularizer=None):
    # 使用 tf.random.normal 替代 tf.truncated_normal
    w = tf.Variable(tf.random.normal(shape, stddev=0.1))  # 正态分布初始化
    if regularizer is not None:
        # 使用 tf.keras.regularizers 来添加正则化
        regularizer_loss = regularizer(w)
        return w, regularizer_loss
    return w, None


def get_bias(shape):
    # 偏置初始化
    return tf.Variable(tf.zeros(shape))


def forward(x, regularizer=None):
    # 第一层全连接
    w1, weight_loss1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    # 第二层全连接
    w2, weight_loss2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    # 返回 logits 和正则化损失
    total_loss = 0.0
    if weight_loss1 is not None:
        total_loss += weight_loss1
    if weight_loss2 is not None:
        total_loss += weight_loss2

    return y, total_loss

# 测试模型
def test_model():
    # 模拟输入数据
    x = tf.random.normal([32, INPUT_NODE])  # 假设批量大小为 32
    regularizer = tf.keras.regularizers.l2(0.01)  # L2 正则化
    y, loss = forward(x, regularizer)

    print(f"Logits: {y}")
    print(f"Total Loss (including regularization): {loss}")

test_model()
