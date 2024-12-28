from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from model_my import  unet
from dataset_my import dataset
import tensorflow as tf


# 调整显存使用情况，避免显存占满
# 调整显存使用情况，避免显存占满
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_train, data_test = dataset()

# 使用已经编译好的模型
from tensorflow.keras.optimizers import Adam

model = unet()
optimizer = Adam(learning_rate=1e-5, clipvalue=0.5)  # 添加梯度裁剪并降低学习率
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# 添加回调函数
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

tensorboard_callback = TensorBoard(log_dir="./log", histogram_freq=1)

# 开始训练
history = model.fit(
    data_train,
    epochs=300,
    validation_data=data_test,
    callbacks=[ reduce_lr, tensorboard_callback]
)

# 保存模型
model.save('unet_model.hdf5')
