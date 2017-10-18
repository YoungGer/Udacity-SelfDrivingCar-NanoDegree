import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from utils import camera_calibration, perspective_trans, combined_thresh, fit_poly, fit_poly2, add_params2img, impose_lane
from utils_detection import VehicleDetection, load_file

class Road_Process():
    def __init__(self):
        # global parameters
        ## camera calibration params
        self.mtx = None
        self.dist = None
        ## perspective transform params
        self.M = None
        self.Minv = None
        
        # fram update params
        self.q_left_fit = deque(maxlen=3)
        self.q_right_fit = deque(maxlen=3)
        self.last_left_fit = None
        self.last_right_fit = None
        self.last_fit = False

        # vehicle detection
        model = load_file('model.pkl')
        clf_final = model['clf']
        scaler = model['scaler']
        self.vd = VehicleDetection(clf_final, scaler)
    
    def sanity_check(self, left_fit, right_fit):
        if abs(left_fit[0]-right_fit[0])/abs(right_fit[0])>1.0 and abs(left_fit[0]-right_fit[0])/abs(left_fit[0])>1.0:
            return False
        else:
            return True
        
    def preprocess(self, cali_img_dir, nx, ny, src, dst):
        # camera calibration
        mtx, dist = camera_calibration(cali_img_dir, nx, ny)
        # perspective transform
        M, Minv = perspective_trans(src, dst)
        ## camera calibration params
        self.mtx = mtx
        self.dist = dist
        ## perspective transform params
        self.M = M
        self.Minv = Minv

    def pipeline(self, img, frame=False):
        """
        input: original image
        """
        # undistort image
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx) 
        # gradient map
        gradient_map = combined_thresh(undist)
        # perspective transform
        warped = cv2.warpPerspective(gradient_map, self.M, (gradient_map.shape[1], gradient_map.shape[0]))
        
        if frame:
            # find lines
            ## first fit left and right lines
            if self.last_fit:
                left_fit, right_fit = fit_poly2(warped, self.last_left_fit, self.last_right_fit)
            else:
                left_fit, right_fit = fit_poly(warped)
                # self.q_left_fit.clear()
                # self.q_right_fit.clear()
            ## sanity check
            if self.sanity_check(left_fit, right_fit):
                self.last_fit = True
                self.last_left_fit, self.last_right_fit = left_fit, right_fit
                self.q_left_fit.append(left_fit)
                self.q_right_fit.append(right_fit)
            else:
                self.last_fit = False
                self.last_left_fit, self.last_right_fit = None, None
            ## add queues
            if not self.q_left_fit or not self.q_right_fit:
                self.q_left_fit.append(left_fit)
                self.q_right_fit.append(right_fit)
            left_fit, right_fit = np.mean(np.array(self.q_left_fit), axis=0), np.mean(np.array(self.q_right_fit), axis=0)
        else:
            left_fit, right_fit = fit_poly(warped)

        # add parameters to img
        undist_withtext = add_params2img(undist, left_fit, right_fit)
        # inverse perspective transform
        rst_lane = impose_lane(warped, undist_withtext, left_fit, right_fit, self.Minv)

        # vehicle detection
        rst = self.vd.pipeline(undist, rst_lane)
        return rst