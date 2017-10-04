import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import INPUT_SHAPE, train_generator, val_generator

def load_data():
    # load original data
    d = pd.read_csv("./data/driving_log.csv")
    dX = d[['center', 'left', 'right']]
    dy = d['steering']
    dX_train, dX_val, dy_train, dy_val = train_test_split(dX, dy, test_size=0.2, random_state=0)
    # get traing and validation x and y
    image_paths_traing = np.concatenate([dX_train.center, dX_train.left, dX_train.right])
    steering_angles_training = np.concatenate([dy_train, dy_train+0.2, dy_train-0.2])
    image_paths_val = np.concatenate([dX_val.center])
    steering_angles_val = np.concatenate([dy_val])
    return image_paths_traing, steering_angles_training, image_paths_val, steering_angles_val

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def train_model(model, image_paths_traing, steering_angles_training, image_paths_val, steering_angles_val):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')
    
    model.compile(loss="mse", optimizer = Adam(lr=1.0e-3))
    
    history_object = model.fit_generator(
        train_generator("./data/", image_paths_traing, steering_angles_training, batch_size=128),
        samples_per_epoch = 20000,
        nb_epoch = 5,
        max_q_size = 1,
        validation_data = val_generator("./data/", image_paths_val, steering_angles_val, batch_size=128),
        nb_val_samples = len(image_paths_val),
        callbacks=[checkpoint],
        verbose=1)

def main():
    image_paths_traing, steering_angles_training, image_paths_val, steering_angles_val = load_data()
    model = build_model()
    train_model(model, image_paths_traing, steering_angles_training, image_paths_val, steering_angles_val)
    
if __name__ == '__main__':
    main()