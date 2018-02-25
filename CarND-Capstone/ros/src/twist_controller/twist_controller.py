import rospy
from pid import PID
from yaw_controller import YawController
import math
from math import tanh

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, \
        vehicle_mass, fuel_capacity, decel_limit, wheel_radius):
        # TODO: Implement
        self.pid_controller = PID(1.0, 0.2, 4)
        self.yaw_controller = YawController(
            wheel_base = wheel_base,
            steer_ratio = steer_ratio,
            min_speed = min_speed,
            max_lat_accel = max_lat_accel,
            max_steer_angle = max_steer_angle
        )
	
        self.prev_throttle = 0
        self.prev_time = None

        total_vehicle_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.max_brake_torque = total_vehicle_mass * abs(decel_limit) * wheel_radius

    def control(self, twist_cmd, current_velocity, dbw_enabled, dist2stop):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
	
	#return 0.1,0,0
        # corner case
        if not self.prev_time:
            self.prev_time = rospy.get_time()
            return 0, 0, 0

        # get current params----------------------------------
        ## from /twist_cmd
        desired_linear_velocity = twist_cmd.twist.linear.x
        desired_angular_velocity = twist_cmd.twist.angular.z
        ## from /current_velocity
        current_linear_velocity = current_velocity.twist.linear.x
        current_angular_velocity = current_velocity.twist.angular.z
        ## from /dbw_enabled


        # pid controller----------------------------------
        delta_v = desired_linear_velocity - current_linear_velocity
        delta_t = float(rospy.get_time() - self.prev_time)
        self.prev_time = rospy.get_time()
        control = self.pid_controller.step(
            error = delta_v,
            sample_time = delta_t
        )
        if control > 0:
            throttle = tanh(control)
	    throttle = max(0.0, min(1.0, throttle))
  	    if throttle - self.prev_throttle > 0.1:
	        throttle = self.prev_throttle + 0.1
        else:
            throttle = 0

        # brake
        rospy.loginfo("Test dist2stop---------------------" + str(dist2stop))
        brake = 0.0
        if delta_v <= 0.0:
            if dist2stop < 15:
                brake = 500.0
            elif dist2stop < 20:
                brake = 100.0
            elif dist2stop < 30:
                brake = 30.0
            elif dist2stop < 60:
                brake = 1.0
            throttle = 0.0
        # if delta_v < 0.0:
        #     brake = tanh(math.fabs(control))

        # yaw_controller----------------------------------
        steering = self.yaw_controller.get_steering(
            linear_velocity = desired_linear_velocity,
            angular_velocity = desired_angular_velocity,
            current_velocity = current_linear_velocity)
	
        self.prev_throttle = throttle
        return throttle, brake, steering
