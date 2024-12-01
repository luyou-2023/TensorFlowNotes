import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5

# 使用 Keras 来实现模型的加载和测试
def test(mnist_data):
    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(mnist_forward.INPUT_NODE,)),
        tf.keras.layers.Dense(mnist_forward.LAYER1_MODE, activation='relu'),
        tf.keras.layers.Dense(mnist_forward.OUTPUT_NODE)
    ])

    # 定义损失函数和准确度
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    # 使用 ExponentialMovingAverage 实现滑动平均
    ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)

    # 保存模型检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, mnist_backward.MODEL_SAVE_PATH, max_to_keep=1)

    while True:
        # 加载检查点
        ckpt_path = manager.latest_checkpoint
        if ckpt_path:
            checkpoint.restore(ckpt_path)
            global_step = int(ckpt_path.split('-')[-1])
            test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

            # 计算测试准确率
            test_accuracy = 0
            num_batches = len(test_images) // 200
            for i in range(num_batches):
                start = i * 200
                end = start + 200
                batch_images = test_images[start:end]
                batch_labels = test_labels[start:end]
                batch_images = batch_images.reshape([-1, mnist_forward.INPUT_NODE])  # 平铺图像

                # 计算准确率
                logits = model(batch_images, training=False)
                accuracy_metric.update_state(batch_labels, logits)

            test_accuracy = accuracy_metric.result().numpy()
            print(f"After {global_step} training step(s), test accuracy = {test_accuracy:.4f}")

            # 清空 metric 状态
            accuracy_metric.reset_states()
        else:
            print("No checkpoint file found")
            return
        time.sleep(TEST_INTERVAL_SECS)

def main():
    # 使用 TensorFlow Keras 加载 MNIST 数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 归一化
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # 转换为 one-hot 编码
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # 创建一个数据集对象
    mnist_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(200)

    # 进行测试
    test(mnist_data)

if __name__ == '__main__':
    main()
