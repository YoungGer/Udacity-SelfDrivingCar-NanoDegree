#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        curr_pose = [self.pose.pose.position.x, self.pose.pose.position.y]
        self.curr_idx = self.get_closest_waypoint(curr_pose)


    def waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints
        self.wp_L = len(self.base_waypoints)

        # Added ---------------------------------------
        self.stop_line_cache = []
        stop_line_positions = self.config['stop_line_positions']
        for index, stop_line in enumerate(stop_line_positions):
            stop_wp_idx = self.get_closest_waypoint(stop_line)
            rospy.loginfo("stop_line---------------------  " + str(stop_wp_idx))
            self.stop_line_cache.append((index, stop_line, stop_wp_idx))

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        # current location
        curr_x, curr_y = pose[0], pose[1]
        # iterate
        closest_dist = float("inf")
        closest_idx = None

        for i, wp in enumerate(self.base_waypoints):
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            dist = (wp_x-curr_x)**2 + (wp_y-curr_y)**2
            if dist<closest_dist:
                closest_dist = dist
                closest_idx = i
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_wp = None

        # get light_wp
        for idx, stop_line, stop_wp_idx in self.stop_line_cache:
            stop_ahead = stop_wp_idx - self.curr_idx
            #rospy.loginfo("stop_ahead: ---------------------" + str(stop_ahead))
            if stop_ahead < 0:
                stop_ahead += self.wp_L
            if (stop_ahead < 200):
                if ( (self.pose.pose.position.x - stop_line[0])**2 + (self.pose.pose.position.y - stop_line[1])**2 < 10000) :
                    light_wp = stop_wp_idx


        # get state
        if light_wp is not None:
            #state = self.get_light_state(light)
            state = TrafficLight.RED
            rospy.loginfo("Here get wp---------------------")
            return light_wp, state
        
        # final return
        rospy.loginfo("no wp *****************************")
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
