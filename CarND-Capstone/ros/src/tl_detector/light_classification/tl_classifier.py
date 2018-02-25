from styx_msgs.msg import TrafficLight
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = load_model(DIR_PATH + '/sim-classifier-8.h5')
        self.model._make_predict_function() 
        self.graph = tf.get_default_graph()
        self.light_state = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        classification_tl_key = {0: TrafficLight.RED, 1: TrafficLight.YELLOW, 2: TrafficLight.GREEN, 4: TrafficLight.UNKNOWN}

        resized = cv2.resize(image, (80,60))/255.

        test_img = np.array([resized])
        # run the prediction
        with self.graph.as_default():
            model_predict = self.model.predict(test_img)
            if model_predict[0][np.argmax(model_predict[0])] > 0.5:
                self.light_state = classification_tl_key[np.argmax(model_predict[0])]

        return self.light_state