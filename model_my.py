import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import VGG16



# def fcn(pretrained_weights=None, input_size=(256, 256, 3)):
#     # 加载vgg16
#     conv_base = VGG16(weights='imagenet', input_shape=input_size, include_top=False)
#
#     # 现在创建多输出模型,三个output
#     layer_names = [
#         'block5_conv3',
#         'block4_conv3',
#         'block3_conv3',
#         'block5_pool']
#
#     # 得到这几个曾输出的列表，为了方便就直接使用列表推倒式了
#     layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]
#
#     # 创建一个多输出模型，这样一张图片经过这个网络之后，就会有多个输出值了
#     multiout_model = Model(inputs=conv_base.input, outputs=layers_output)
#
#     multiout_model.trainable = True
#     inputs = Input(shape=input_size)
#
#     # 这个多输出模型会输出多个值，因此前面用多个参数来接收即可。
#     out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multiout_model(inputs)
#
#     # 现在将最后一层输出的结果进行上采样,然后分别和中间层多输出的结果进行相加，实现跳级连接
#     x1 = Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(out)
#
#     # 上采样之后再加上一层卷积来提取特征
#     x1 = Conv2D(512, 3, padding='same', activation='relu')(x1)
#
#     # 与多输出结果的倒数第二层进行相加，shape不变
#     x2 = tf.add(x1, out_block5_conv3)
#
#     # x2进行上采样
#     x2 = Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(x2)
#     # 直接拿到x3，不使用
#     x3 = tf.add(x2, out_block4_conv3)
#
#     # x3进行上采样
#     x3 = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x3)
#
#     # 增加卷积提取特征
#     x3 = Conv2D(256, 3, padding='same', activation='relu')(x3)
#     x4 = tf.add(x3, out_block3_conv3)
#
#     # x4还需要再次进行上采样，得到和原图一样大小的图片，再进行分类
#     x5 = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x4)
#
#     # 继续进行卷积提取特征
#     x5 = Conv2D(128, 3, padding='same', activation='relu')(x5)
#
#     # 最后一步，图像还原
#     preditcion = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='softmax')(x5)
#
#     model = Model(inputs=inputs, outputs=preditcion)
#     model.summary()
#
#     # 加载预训练模型
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model



def unet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(inputs)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv6)

    up7 = Conv2D(
        256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)

    up8 = Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)

    up9 = Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(
        optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

