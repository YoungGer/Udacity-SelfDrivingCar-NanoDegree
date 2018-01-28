#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
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

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # rospy.Subscriber('/traffic_waypoint', PoseStamped, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None
        self.base_waypoints = None
        self.final_waypoints = None
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        # get current pose
        self.current_pose = msg.pose
        # get final wps through current_pose and final_waypoints
        self.get_final_waypoints()
        # publish
        if self.final_waypoints:
            self.publish_final_waypoints()



    def waypoints_cb(self, msg):
        # TODO: Implement
        self.base_waypoints = msg.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

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
        # add waypoints speed
        for i in range(len(self.final_waypoints)):
            self.final_waypoints[i].twist.twist.linear.x = 25 * 0.447
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
