# import tensorflow as tf
# import numpy as np
# import glob
#
#
# def read_jpg(path, channels=1):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_jpeg(img, channels=channels)  # 指定为单通道
#     return img
#
#
# def read_png(path, channels=1):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_png(img, channels=channels)  # 指定为单通道
#     return img
#
#
# # 定义数据增强函数
# def augment_image(image, annotation):
#     # 随机水平翻转
#     if tf.random.uniform(()) > 0.5:
#         image = tf.image.flip_left_right(image)
#         annotation = tf.image.flip_left_right(annotation)
#
#     # 随机旋转（90度的倍数）
#     k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
#     image = tf.image.rot90(image, k=k)
#     annotation = tf.image.rot90(annotation, k=k)
#
#     # 随机亮度调整
#     image = tf.image.random_brightness(image, max_delta=0.2)
#
#     # 随机对比度调整
#     image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
#
#     return image, annotation
#
#
# # 数据归一化调整
# def normal_img(input_images, input_anno):
#     input_images = tf.cast(input_images, tf.float32)
#     input_images = input_images / 127.5 - 1  # 归一化到 [-1, 1]
#     input_anno = tf.cast(input_anno, tf.float32)  # 确保标签也是float32类型
#     input_anno = input_anno / 255.0  # 如果标签是0-255之间，归一化到 [0, 1]
#     return input_images, input_anno
#
#
# # 加载函数
# def load_images(input_images_path, input_anno_path, augment=False):
#     input_image = read_jpg(input_images_path, channels=1)  # 读取为单通道
#     input_anno = read_png(input_anno_path, channels=1)  # 注释图像通常已经是单通道
#
#     # 调整图像和注释的大小为512x512，并确保通道数为1
#     input_image = tf.image.resize(input_image, (512, 512))
#     input_anno = tf.image.resize(input_anno, (512, 512), method='nearest')  # 使用最近邻插值以保持二值标签不变
#
#     if augment:
#         input_image, input_anno = augment_image(input_image, input_anno)
#
#     return normal_img(input_image, input_anno)
#
#
# def dataset():
#     # 读取图像和目标图像
#     images = glob.glob(r"./data/images/*.jpg")  # 假设输入图像是 .jpg 格式
#     anno = glob.glob(r"./data/masks/*.png")
#
#     # 现在对读取进来的数据进行制作batch
#     np.random.seed(1)
#     index = np.random.permutation(len(images))  # 随机打乱图片路径顺序
#     images = np.array(images)[index]
#     anno = np.array(anno)[index]
#
#     # 创建dataset
#     dataset = tf.data.Dataset.from_tensor_slices((images, anno))
#
#     # 测试数据量和训练数据量，20%测试。
#     test_count = int(len(images) * 0.2)
#     train_count = len(images) - test_count
#
#     # 取出训练数据和测试数据
#     data_train = dataset.skip(test_count)
#     data_test = dataset.take(test_count)
#
#     # 映射加载和归一化图像，并选择性地应用数据增强
#     data_train = data_train.map(lambda x, y: load_images(x, y, augment=True),
#                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     data_test = data_test.map(lambda x, y: load_images(x, y, augment=False),
#                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     # 设置batch size
#     BATCH_SIZE = 3
#     data_train = data_train.shuffle(100).batch(BATCH_SIZE)
#     data_test = data_test.batch(BATCH_SIZE)
#
#     return data_train, data_test
#
#
# # 示例调用
# data_train, data_test = dataset()
import tensorflow as tf
import numpy as np
import glob


def read_jpg(path, channels=1):
    img = tf.io.read_file(path)  # 直接接受张量路径
    img = tf.image.decode_jpeg(img, channels=channels)  # 指定为单通道
    return img


def read_png(path, channels=1):
    img = tf.io.read_file(path)  # 直接接受张量路径
    img = tf.image.decode_png(img, channels=channels)  # 指定为单通道
    return img

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
    input_image = read_jpg(input_images_path, channels=1)  # 读取为单通道
    input_anno = read_png(input_anno_path, channels=1)

    # 调整图像和注释的大小为512x512，并确保通道数为1
    input_image = tf.image.resize(input_image, (512, 512))
    input_anno = tf.image.resize(input_anno, (512, 512), method='nearest')  # 使用最近邻插值以保持二值标签不变

    if augment:
        input_image, input_anno = augment_image(input_image, input_anno)

    return normal_img(input_image, input_anno)


def dataset():
    # 读取图像和目标图像
    images = glob.glob(r"./data/images/*.png")  # 假设输入图像是 .jpg 格式
    anno = glob.glob(r"./data/masks/*.png")

    # 确认路径列表为纯字符串
    images = list(map(str, images))
    anno = list(map(str, anno))

    # 随机打乱图片路径顺序
    np.random.seed(1)
    index = np.random.permutation(len(images))
    images = np.array(images)[index]
    anno = np.array(anno)[index]

    # 创建dataset时确保路径是字符串类型
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