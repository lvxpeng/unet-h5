from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from model_my import  unet
from dataset_my import dataset
import tensorflow as tf


# 调整显存使用情况，避免显存占满
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_train, data_test = dataset()

# unet
model = unet()
model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['acc'])
# fcn
# model = fcn()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="d:/temp/log", histogram_freq=1)
model.fit(data_train, epochs=100, batch_size = 8, validation_data=data_test, callbacks = [tensorboard_callback])

model.save('unet_model.h5')
# 加载保存的模型
# new_model=tf.keras.models.load_model('FCN_model.h5')
