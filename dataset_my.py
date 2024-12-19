import tensorflow as tf
import numpy as np
import glob



def read_jpg(path):
    img=tf.io.read_file(path)
    img=tf.image.decode_jpeg(img,channels=3)
    return img


def read_png(path):
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img,channels=1)
    return img


#现在编写归一化的函数
def normal_img(input_images,input_anno):
    input_images=tf.cast(input_images,tf.float32)
    input_images=input_images/127.5-1
    input_anno-=1
    return input_images,input_anno


#加载函数
def load_images(input_images_path,input_anno_path):
    input_image=read_jpg(input_images_path)
    input_anno=read_png(input_anno_path)
    input_image=tf.image.resize(input_image,(256,256))
    input_anno=tf.image.resize(input_anno,(256,256))
    return normal_img(input_image,input_anno)


def dataset():
    # 读取图像和目标图像
    images = glob.glob(r"d:/temp/images/*.png")
    anno = glob.glob(r"d:/temp/json_to_mask/*.png")

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
    data_train = dataset.skip(test_count)  # 跳过前test的数据
    data_test = dataset.take(test_count)  # 取前test的数据
    data_train = data_train.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_test = data_test.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 现在开始batch的制作,不制作batch会使维度由4维降为3维
    BATCH_SIZE = 3
    data_train = data_train.shuffle(100).batch(BATCH_SIZE)
    data_test = data_test.batch(BATCH_SIZE)

    return data_train, data_test
