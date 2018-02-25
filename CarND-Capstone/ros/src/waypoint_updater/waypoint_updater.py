#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32, Float32
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # rospy.Subscriber('/traffic_waypoint', PoseStamped, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.red_light_pub = rospy.Publisher('red_light', Int32, queue_size=1)
        self.dist2stop_pub = rospy.Publisher('dist2stop', Float32, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None
        self.base_waypoints = None
        self.final_waypoints = None
        self.light_wp = -1
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        # get current pose
        #rospy.loginfo("pose_cb -----------------------")
        self.current_pose = msg.pose
        # get final wps through current_pose and final_waypoints
        if self.base_waypoints:
            self.get_final_waypoints()
        # publish
        if self.final_waypoints:
            self.publish_final_waypoints()

    def waypoints_cb(self, msg):
        # TODO: Implement
        self.base_waypoints = msg.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.light_wp = msg.data

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # def distance(self, waypoints, wp1, wp2):
    #     dist = 0
    #     dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    #     for i in range(wp1, wp2+1):
    #         dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
    #         wp1 = i
    #     return dist
    def distance(self, car_pose, waypoint):
        dist_x = car_pose.position.x - waypoint.pose.pose.position.x
        dist_y = car_pose.position.y - waypoint.pose.pose.position.y
        dist_z = car_pose.position.z - waypoint.pose.pose.position.z
        return math.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

    # Added -----------------------------------------------
    def find_closet_waypoint(self):
        # current location
        curr_x, curr_y = self.current_pose.position.x, self.current_pose.position.y
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

    def get_final_waypoints(self):
        # get final waypoints
        closest_idx = self.find_closet_waypoint()
        self.final_waypoints = self.base_waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]
        N = len(self.final_waypoints)
        # pub red_light to dbw node
        self.red_light_pub.publish(1 if self.light_wp != -1 else -1)


        # no red bulb
        if self.light_wp == -1:
            rospy.loginfo("no wp---------------------")
            self.dist2stop_pub.publish(1000000.0)
            for i in range(len(self.final_waypoints)):
                self.final_waypoints[i].twist.twist.linear.x = 25 * 0.447
            return

        # red bulb
        # rospy.loginfo("Here get wp---------------------")
        # tl_dist_effective = self.distance(self.current_pose, self.base_waypoints[self.light_wp])
        # vel = self.current_velocity
        # decel = 1.0 * vel / tl_dist_effective

        # for i in range(len(self.final_waypoints) - 1):
        #     if i == 0:
        #         dist = self.distance(self.current_pose, self.final_waypoints[0])
        #     else:
        #         dist = self.distance(self.final_waypoints[i - 1].pose.pose, self.final_waypoints[i])
        #     vel -= decel * dist

        #     self.final_waypoints[i].twist.twist.linear.x = vel


        # for i in range(len(self.final_waypoints)):
        #     self.final_waypoints[i].twist.twist.linear.x = 0
        # return


        rospy.loginfo("Here get wp---------------------")
        dist_stop = self.distance(self.current_pose, self.base_waypoints[self.light_wp])
        self.dist2stop_pub.publish(dist_stop)
        devel = 1.0 * self.current_velocity / dist_stop
        vel = self.current_velocity
        for i in range(N):
            curr_dist_stop = self.distance(self.final_waypoints[i].pose.pose, self.base_waypoints[self.light_wp])
            wp_vel = devel * curr_dist_stop * 0.8
            wp_vel = max(wp_vel, 0)
            self.final_waypoints[i].twist.twist.linear.x = wp_vel
        return


    def publish_final_waypoints(self):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(lane)



if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
