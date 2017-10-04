import numpy as np
import cv2

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def train_generator(data_dir, image_paths, steering_angles, batch_size):
    """
    Generate batch training data given image paths and associated steering angles
    """
    X = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    Y = np.empty(batch_size)
    while True:
        cnt = 0
        for i in np.random.permutation(range(len(image_paths))):
            # current image path and steering angle
            image_path = data_dir + image_paths[i].strip()
            steering_angle = steering_angles[i]
            # data augmentation through flip image
            if np.random.random() < 0.5:
                image = np.fliplr(cv2.imread(image_path))
                steering_angle = - steering_angle
            else:
                image = cv2.imread(image_path)
            # image crop
            image = image[60:-25, :, :]
            # image resize
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
            # iterate
            X[cnt] = image
            Y[cnt] = steering_angle
            cnt += 1
            if cnt==batch_size:
                break
        yield X, Y
        
def val_generator(data_dir, image_paths, steering_angles, batch_size):
    """
    Generate batch training data given image paths and associated steering angles
    """
    X = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    Y = np.empty(batch_size)
    while True:
        cnt = 0
        for i in np.random.permutation(range(len(image_paths))):
            # current image path and steering angle
            image_path = data_dir + image_paths[i].strip()
            image = cv2.imread(image_path)
            steering_angle = steering_angles[i]
            # image crop
            image = image[60:-25, :, :]
            # image resize
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
            # iterate
            X[cnt] = image
            Y[cnt] = steering_angle
            cnt += 1
            if cnt==batch_size:
                break
        yield X, Y 