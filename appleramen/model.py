import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class_to_category = {
    0: 'apple pie',
    1: 'cannoli',
    2: 'ramen'
}

base_model1 = VGG16(
    weights='imagenet', include_top=False, input_shape=(228, 228, 3))
base_model2 = InceptionV3(
    weights='imagenet', include_top=False, input_shape=(228, 228, 3))
base_model3 = DenseNet121(
    weights='imagenet', include_top=False, input_shape=(228, 228, 3))


def extract_features1(train_gen):
    features = np.zeros(shape=(1, 7, 7, 512))
    labels = np.zeros(shape=(1))
    batch_size = 1

    i = 0
    for input_batch, labels_batch in train_gen:
        features_batch = base_model1.predict(input_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= 1:
            break
    return features


def extract_features2(train_gen):
    features = np.zeros(shape=(1, 5, 5, 2048))
    labels = np.zeros(shape=(1))
    batch_size = 1

    i = 0
    for input_batch, labels_batch in train_gen:
        features_batch = base_model2.predict(input_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= 1:
            break
    return features


def extract_features3(train_gen):
    features = np.zeros(shape=(1, 7, 7, 1024))
    labels = np.zeros(shape=(1))
    batch_size = 1

    i = 0
    for input_batch, labels_batch in train_gen:
        features_batch = base_model3.predict(input_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= 1:
            break
    return features


def ensemble(model1, model2, model3, f1, f2, f3):
    outputs = [model1.predict(f1), model2.predict(f2), model3.predict(f3)]
    avg = tf.keras.layers.average(outputs)
    return avg


def predict(path, model1, model2, model3):
    tt = ImageDataGenerator(rescale=1/255)
    test = tt.flow_from_directory(path, target_size=(
        228, 228), batch_size=1, class_mode='binary')
    f1 = extract_features1(test)
    f2 = extract_features2(test)
    f3 = extract_features3(test)
    pred = ensemble(model1, model2, model3, f1, f2, f3)

    return class_to_category[int(np.argmax(pred, axis=1))]


def create_models():
    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten(input_shape=(7, 7, 512)))
    model1.add(tf.keras.layers.Dense(256, activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001), input_dim=(7*7*512)))
    model1.add(tf.keras.layers.Dropout(0.5))
    model1.add(tf.keras.layers.Dense(3, activation='softmax'))
    model1.load_weights('./model weights/model1-v3.h5')

    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten(input_shape=(5, 5, 2048)))
    model2.add(tf.keras.layers.Dense(256, activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001), input_dim=(5*5*2048)))
    model2.add(tf.keras.layers.Dropout(0.5))
    model2.add(tf.keras.layers.Dense(3, activation='softmax'))
    model2.load_weights('./model weights/model2-v3.h5')

    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.Flatten(input_shape=(7, 7, 1024)))
    model3.add(tf.keras.layers.Dense(256, activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001), input_dim=(7*7*1024)))
    model3.add(tf.keras.layers.Dropout(0.5))
    model3.add(tf.keras.layers.Dense(3, activation='softmax'))
    model3.load_weights('./model weights/model3-v3.h5')

    return model1, model2, model3
