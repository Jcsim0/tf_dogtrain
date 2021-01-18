import math, os
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from ResNet.resnet50_model import ResNet50
from tensorflow.keras.utils import plot_model


DATA_DIR = 'E:/01Jcsim/01project/Python/dogtrain/mainproject/ResNet'
TRAIN_DIR = DATA_DIR + '/train_Img/'
VALID_DIR = DATA_DIR + '/valid_img/'

# 每次训练张数
BATCH_SIZE = 16
WEIGHTS_PATH = './resnet50_pre_weights.h5'
logging = TensorBoard()

if __name__ == "__main__":
    num_train_samples = sum([len(files) for root, dirs, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for root, dirs, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    # 图像生成器实例化
    train_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator()
    val_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator()

    # 训练图片生成器
    train_generator = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        class_mode='categorical',
        shuffle=True,
        batch_size=BATCH_SIZE
    )
    val_generator = val_gen.flow_from_directory(
        VALID_DIR,
        target_size=(224, 224),
        class_mode='categorical',
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    finetuned_model = ResNet50()

    plot_model(finetuned_model, show_shapes=True, to_file='model.png')
    # 加载权重
    finetuned_model.load_weights(WEIGHTS_PATH, by_name=True)
    # 获取模型默认的Labels序列
    classes = list(iter(train_generator.class_indices))

    # 编译模型
    finetuned_model.compile(optimizer=Adam(lr=0.0001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    for c in train_generator.class_indices:
        classes[train_generator.class_indices[c]] = c
    finetuned_model.classes = classes
    print(finetuned_model.classes)

    early_stopping = EarlyStopping(patience=10)
    # 容错机制
    checkpointer = ModelCheckpoint('resnet50_best.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)
    # 学习模型
    finetuned_model.fit_generator(train_generator,
                                  steps_per_epoch=num_train_steps,
                                  epochs=10,
                                  callbacks=[logging, early_stopping, checkpointer],
                                  validation_data=val_generator,
                                  validation_steps=num_valid_steps)
    # 保存模型
    finetuned_model.save('resnet50_final.h5')
