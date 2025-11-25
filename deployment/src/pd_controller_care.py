#!/usr/bin/env python3
"""pd_controller_ros2.py – ROS 2 version of the original PD controller node.

Behaviour
---------
* Listens to WAYPOINT_TOPIC (`Float32MultiArray`) and REACHED_GOAL_TOPIC (`Bool`).
* Computes linear/angular velocity commands with a simple PD‑style heuristic.
* Publishes geometry_msgs/Twist on the velocity topic defined in robot.yaml.
* Continuously measures and reports distance traveled.
* Also checks total elapsed time and triggers goal reached when time exceeds 1 minute.

Run after installing your Python package (or directly with `ros2 run` / `python`).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import yaml
import argparse

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from utils import clip_angle  # assumes utils.py provides this helper
ROBOT_CONFIG_PATH ="../config/robot.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_cfg = yaml.safe_load(f)

MAX_V: float = robot_cfg["max_v"]
MAX_W: float = robot_cfg["max_w"]
DT: float = 1.0 / robot_cfg["frame_rate"]

RATE: int = 4
EPS: float = 1e-8
WAYPOINT_TIMEOUT: float = 1.0
DISTANCE_REPORT_INTERVAL: float = 0.1
MAX_TIME: float = 30000.0


class PDControllerNode():

    def __init__(self, args: argparse.Namespace) -> None:
    
        rospy.init_node("pd_controller_node", anonymous=True)
        self.controller_type = args.control

        waypoint_topic = "/waypoint"
        vel_topic = "/cmd_vel"

        self.waypoint: Optional[np.ndarray] = None
        self._last_wp_time: float = 0.0
        self.reached_goal: bool = False
        self.reverse_mode: bool = False

        self.total_distance: float = 0.0
        self.last_velocity_time: float = time.time()
        self.last_report_time: float = time.time()
        self.current_velocity: float = 0.0

        self.start_time: float = time.time()
        self.total_time: float = 0.0

        self.vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=1)
        rospy.Subscriber(waypoint_topic, Float32MultiArray, self._waypoint_cb, queue_size=1)
        rospy.Subscriber("/topoplan/reached_goal", Bool, self._goal_cb, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / RATE), self._timer_cb)
        rospy.loginfo(
            "PD controller node initialised – waiting for waypoints…"
        )

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        self.waypoint = np.asarray(msg.data, dtype=float)
        self._last_wp_time = time.time()
        rospy.loginfo(f"Waypoint received: {self.waypoint.tolist()}")

    def _goal_cb(self, msg: Bool) -> None:
        self.reached_goal = msg.data
        if self.reached_goal:
            rospy.loginfo(f"Total distance: {self.total_distance:.3f} m")
            rospy.loginfo(f"Total time: {self.total_time:.3f} s")

    def _waypoint_valid(self) -> bool:
        return (
            self.waypoint is not None
            and (time.time() - self._last_wp_time) < WAYPOINT_TIMEOUT
        )

    def _pd_control(self, wp: np.ndarray) -> Tuple[float, float]:
        """Compute (v, w) for 2D or 4D waypoint."""
        if wp.size == 2:
            dx, dy = wp
            use_heading = False
        elif wp.size == 4:
            dx, dy, hx, hy = wp
            use_heading = np.abs(dx) < EPS and np.abs(dy) < EPS
        else:
            raise ValueError("Waypoint must be 2D or 4D vector")

        if use_heading:
            v = 0.0
            desired_yaw = np.arctan2(hy, hx)
        elif abs(dx) < EPS:
            v = 0.0
            desired_yaw = np.sign(dy) * np.pi / 2
        else:
            v = dx / DT
            desired_yaw = np.arctan(dy / dx)

        if self.controller_type != "nomad":
            MAX_ROTATION_ONLY_ANGLE = np.deg2rad(30)
            if abs(desired_yaw) > MAX_ROTATION_ONLY_ANGLE:
                v = 0.0

        w = clip_angle(desired_yaw) / DT

        return float(np.clip(v, 0.0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))

    def _update_distance(self, velocity: float) -> None:
        current_time = time.time()
        dt = current_time - self.last_velocity_time

        if velocity > 0.0:
            distance = velocity * dt
            self.total_distance += distance
            if self.total_distance > 100.0:
                self._goal_cb(Bool(data=True))

        self.last_velocity_time = current_time

        if current_time - self.last_report_time >= DISTANCE_REPORT_INTERVAL:
            rospy.loginfo(f"Current distance: {self.total_distance:.3f} m")
            self.last_report_time = current_time

    def _timer_cb(self, event) -> None:
        vel_msg = Twist()

        current_time = time.time()
        self.total_time = current_time - self.start_time

        print("Current time:", self.total_time)
        if self.total_time > MAX_TIME:
            self._goal_cb(Bool(data=True))

        if self.reached_goal:
            self.vel_pub.publish(vel_msg)
            rospy.loginfo("Reached goal – stopping controller.")
            rospy.signal_shutdown("Reached goal. stopping controller.")
            return

        if self._waypoint_valid():
            v, w = self._pd_control(self.waypoint)
            if self.reverse_mode:
                v *= -1.0
            vel_msg.linear.x = v
            vel_msg.angular.z = w

            self.current_velocity = abs(v)
            self._update_distance(self.current_velocity)

            rospy.loginfo(f"Publishing velocity: v={v:.3f}, w={w:.3f}")
        else:
            self._update_distance(0.0)

        self.vel_pub.publish(vel_msg)

    def _run(self):
        rospy.spin()

def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control", type=str, default="care", help="control type (nomad, care)"
    )

    args, _ = parser.parse_known_args()

    node = PDControllerNode(args)
    node._run()


if __name__ == "__main__":
    main()
