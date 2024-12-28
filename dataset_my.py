import tensorflow as tf
import numpy as np
import glob


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


# 定义数据增强函数
def augment_image(image, annotation):
    # 随机水平翻转
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        annotation = tf.image.flip_left_right(annotation)

    # 随机旋转（90度的倍数）
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    annotation = tf.image.rot90(annotation, k=k)

    # 随机亮度调整
    image = tf.image.random_brightness(image, max_delta=0.2)

    # 随机对比度调整
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # 随机裁剪并缩放回原尺寸
    image_shape = tf.shape(image)
    cropped_image = tf.image.random_crop(image, size=[image_shape[0] // 2, image_shape[1] // 2, 3])
    image = tf.image.resize(cropped_image, [image_shape[0], image_shape[1]])

    # 对注释图像进行相同的随机裁剪（如果需要）
    annotation_shape = tf.shape(annotation)
    cropped_annotation = tf.image.random_crop(annotation, size=[annotation_shape[0] // 2, annotation_shape[1] // 2, 1])
    annotation = tf.image.resize(cropped_annotation, [annotation_shape[0], annotation_shape[1]], method='nearest')

    return image, annotation


# 数据归一化调整
def normal_img(input_images, input_anno):
    input_images = tf.cast(input_images, tf.float32)
    input_images = input_images / 127.5 - 1  # 归一化到 [-1, 1]
    input_anno = tf.cast(input_anno, tf.float32)  # 确保标签也是float32类型
    input_anno = input_anno / 255.0  # 如果标签是0-255之间，归一化到 [0, 1]
    return input_images, input_anno


# 加载函数
def load_images(input_images_path, input_anno_path, augment=False):
    input_image = read_jpg(input_images_path)
    input_anno = read_png(input_anno_path)

    input_image = tf.image.resize(input_image, (256, 256))
    input_anno = tf.image.resize(input_anno, (256, 256), method='nearest')  # 使用最近邻插值以保持二值标签不变

    if augment:
        input_image, input_anno = augment_image(input_image, input_anno)

    return normal_img(input_image, input_anno)


def dataset():
    # 读取图像和目标图像
    images = glob.glob(r"./data/images/*.png")
    anno = glob.glob(r"./data/masks/*.png")

    # 现在对读取进来的数据进行制作batch
    np.random.seed(1)
    index = np.random.permutation(len(images))  # 随机打乱7390个数
    images = np.array(images)[index]
    anno = np.array(anno)[index]

    # 创建dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, anno))

    # 测试数据量和训练数据量，20%测试。
    test_count = int(len(images) * 0.2)
    train_count = len(images) - test_count

    # 取出训练数据和测试数据
    data_train = dataset.skip(test_count)
    data_test = dataset.take(test_count)

    # 映射加载和归一化图像，并选择性地应用数据增强
    data_train = data_train.map(lambda x, y: load_images(x, y, augment=True),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_test = data_test.map(lambda x, y: load_images(x, y, augment=False),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 设置batch size
    BATCH_SIZE = 3
    data_train = data_train.shuffle(100).batch(BATCH_SIZE)
    data_test = data_test.batch(BATCH_SIZE)

    return data_train, data_test


# 示例调用
data_train, data_test = dataset()