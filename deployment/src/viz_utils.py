from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
import rospy
import numpy as np

def make_path_marker(points, marker_id, r, g, b, frame_id="base_link"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "multi_paths"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = 0.05  # line width
    marker.color.a = 1.0
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b

    points = np.array(points)

    # Case 1: multiple waypoints (N, 2)
    if points.ndim == 2 and points.shape[1] == 2:
        for x, y in points:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), 0.0
            marker.points.append(p)

    # Case 2: single point (2,)
    elif points.ndim == 1 and points.shape[0] == 2:
        p = Point()
        p.x, p.y, p.z = float(points[0]), float(points[1]), 0.0
        marker.points.append(p)

    else:
        rospy.logwarn(f"Unsupported points shape: {points.shape}")

    return marker


def viz_chosen_wp(chosen_waypoint, waypoint_viz_pub):
    marker = Marker()
    marker.header.frame_id = "base_link"   # or "odom", "base_link" depending on your TF
    marker.header.stamp = rospy.Time.now()

    marker.ns = "points"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    # Example 2D point (x, y, z=0)
    marker.pose.position.x = chosen_waypoint[0]
    marker.pose.position.y = chosen_waypoint[1]
    marker.pose.position.z = 0.0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Sphere size
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    # Color (red)
    marker.color.a = 1.0  # alpha
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    waypoint_viz_pub.publish(marker)

def Marker_process(points, id, selected_num, length=8):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns= "points"
    marker.id = id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    if selected_num == id:
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    else:
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    for i in range(length):
        p = Point()
        p.x = points[2 * i]
        p.y = points[2 * i + 1]
        p.z = 0
        marker.points.append(p)
    return marker

def Marker_process_goal(points, marker, length=1):
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns= "points"
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    
    for i in range(length):
        p = Point()
        p.x = points[2 * i]
        p.y = points[2 * i + 1]
        p.z = 1
        marker.points.append(p)
    return marker