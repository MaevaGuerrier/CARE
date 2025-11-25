import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32MultiArray, Int32
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from utils import msg_to_pil, to_numpy, transform_images, load_model, pil_to_msg
from vint_train.training.train_utils import get_action
from collections import deque
from typing import Tuple, List

# import tf2_ros
# import tf2_sensor_msgs.tf2_sensor_msgs
import argparse
import os

from viz_utils import *

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC,
                        CLOSEST_NODE_TOPIC,
                        DEPTH_POINT_CLOUD_TOPIC)

TOPOMAP_IMAGES_DIR = "../topomaps/images"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]  # Hz

PIXELS_PER_M = 60.0  # px for 1 m (feel free to tune)
ORIGIN_Y_RATIO = 0.95  # where to anchor trajectories vertically


BASE_LINK = "base_link"


def _load_model(model_name: str, device: torch.device):
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    mconf_path = model_paths[model_name]["config_path"]
    ckpt_path = model_paths[model_name]["ckpt_path"]
    with open(mconf_path, "r") as f:
        model_params = yaml.safe_load(f)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, model_params, device).to(device).eval()
    return model, model_params


class NavigationNode:
    """Sub‑goal navigation with topomap + trajectory visualisation."""

    def __init__(self, args: argparse.Namespace):
        rospy.init_node("navigate_care", anonymous=True)
        self.args = args

        # Torch / model ------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")

        self.model, self.model_params = _load_model(args.model, self.device)
        rospy.loginfo(f"Using model type: {self.model_params['model_type']}")
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.context_size: int = self.model_params["context_size"]

        if self.model_params["model_type"] == "nomad":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        self.bridge = CvBridge()
        self.context_queue: Deque[np.ndarray] = deque(maxlen=self.context_size + 1)
        self.last_ctx_time = rospy.Time.now()
        self.ctx_dt = 1 / RATE

        self.current_waypoint = np.zeros(2)
        self.obstacle_points = None

        self.top_view_size = (400, 400)

        self.safety_margin = 0.05
        self.proximity_threshold = 5

        self.top_view_resolution = self.top_view_size[0] / self.proximity_threshold
        self.top_view_sampling_step = 5

        self.DIM = (640, 480)

        self.topomap, self.goal_node = self._load_topomap(args.dir, args.goal_node)

        self.closest_node = 0

        rospy.Subscriber(IMAGE_TOPIC, Image, self._image_cb, queue_size=1)
        rospy.Subscriber(DEPTH_POINT_CLOUD_TOPIC, PointCloud2, self._pointcloud_callback)

        self.waypoint_pub = rospy.Publisher(
            WAYPOINT_TOPIC, Float32MultiArray, queue_size=1
        )
        rospy.loginfo(f"Publishing waypoints to {WAYPOINT_TOPIC}")
        self.sampled_actions_pub = rospy.Publisher(
            SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1
        )
        
        self.viz_pub = rospy.Publisher("navigation_viz", Image, queue_size=1)
        self.subgoal_pub = rospy.Publisher("navigation_subgoal", Image, queue_size=1)
        self.goal_pub_img = rospy.Publisher("navigation_goal", Image, queue_size=1)
        self.path_pub = rospy.Publisher("traj_nomad", MarkerArray, queue_size=10)
        self.path_pub_care = rospy.Publisher("traj_care", MarkerArray, queue_size=10)
        # self.pub_obstacles = rospy.Publisher("obstacles", Image, queue_size=10)
        self.pub_pcd = rospy.Publisher("pointcloud_viz", PointCloud2, queue_size=10)
        self.image_pub = rospy.Publisher("obstacles", Image, queue_size=10)

        # we might have duplicated topics here but we copy same structure as other baselines
        # The topics are used for the metrics
        # TODO: clean up eventually
        self.waypoint_viz_pub = rospy.Publisher("viz_wp", PoseStamped, queue_size=1)
        self.distances_pub = rospy.Publisher("/distances", Float32MultiArray, queue_size=1)
        self.goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
        self.goal_img_pub = rospy.Publisher("/topoplan/goal_img", Image, queue_size=1)
        self.subgoal_img_pub = rospy.Publisher("/topoplan/subgoal_img", Image, queue_size=1)
        self.closest_node_img_pub = rospy.Publisher("/topoplan/closest_node_img", Image, queue_size=1)
        self.closest_node_pub = rospy.Publisher(CLOSEST_NODE_TOPIC, Int32, queue_size=10)

        # rate = rospy.Rate(RATE)
        rospy.Timer(rospy.Duration(1.0 / RATE), self._timer_cb)

    def _load_topomap(
        self, dir_path: str, goal_node: int
    ) -> Tuple[List[PILImage.Image], int]:
        topomap_filenames = sorted(
            os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, dir_path)),
            key=lambda x: int(x.split(".")[0]),
        )
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{dir_path}"
        num_nodes = len(os.listdir(topomap_dir))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))

        assert -1 <= goal_node < len(topomap), "Invalid goal index for the topomap"
        if goal_node == -1:
            goal_node = len(topomap) - 1

        return topomap, goal_node

    def _image_cb(self, msg: Image):
        self.context_queue.append(msg_to_pil(msg))

    def _pointcloud_callback(self, msg: PointCloud2):        
        points = np.array(
            [
                [p[0], p[1], p[2]]
                for p in pc2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
            ]
        )
        # OAK D PRO Y VERTICAL DOWN
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        # Y is vertical (down), so filter ground:
        mask = (Z > 0) & (Z <= self.proximity_threshold) & (Y >= -0.05)
        # handling ceiling points
        points_filtered = points[mask]

        header = msg.header
        filtered_msg = pc2.create_cloud_xyz32(header, points_filtered.tolist())
        self.pub_pcd.publish(filtered_msg)

        self._update_top_view_and_obstacles(X[mask], Y[mask], Z[mask])

    def _update_top_view_and_obstacles(self, X, Y, Z_0):
        Z = np.maximum(Z_0 - self.safety_margin, 1e-3)
        img_x = np.int32(self.top_view_size[0] // 2 + X * self.top_view_resolution)
        img_y = np.int32(self.top_view_size[1] - Z * self.top_view_resolution)

        valid = (
            (img_x >= 0)
            & (img_x < self.top_view_size[0])
            & (img_y >= 0)
            & (img_y < self.top_view_size[1])
        )
        img_x = img_x[valid]
        img_y = img_y[valid]
        depth_vals = Z[valid]
        real_x = X[valid]
        real_z = Z[valid]

        sampled_obstacles = []
        for x in range(0, self.top_view_size[0], self.top_view_sampling_step):
            col_idxs = np.where(img_x == x)[0]
            if len(col_idxs) > 0:
                closest_idx = col_idxs[np.argmax(depth_vals[col_idxs])]
                sampled_obstacles.append([real_x[closest_idx], real_z[closest_idx]])

        if sampled_obstacles:
            img = np.ones(
                (self.top_view_size[1], self.top_view_size[0], 3), dtype=np.uint8
            )
            obs_array = np.array(sampled_obstacles)
            x_local = obs_array[:, 1]
            y_local = -obs_array[:, 0]
            self.obstacle_points = np.stack([x_local, y_local], axis=1)

            for x, y in self.obstacle_points:
                # Convert (x, y) in meters to pixel coordinates
                px = int(self.top_view_size[0] // 2 + y * self.top_view_resolution)
                py = int(self.top_view_size[1] - x * self.top_view_resolution)

                if 0 <= px < self.top_view_size[0] and 0 <= py < self.top_view_size[1]:
                    color = (0, 165, 255)  # orange
                    cv2.circle(img, (px, py), 3, color, -1)

            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_img.header.stamp = rospy.Time.now()
            ros_img.header.frame_id = BASE_LINK
            self.image_pub.publish(ros_img)

        else:
            img = np.ones(
                (self.top_view_size[1], self.top_view_size[0], 3), dtype=np.uint8
            )
            self.obstacle_points = None
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_img.header.stamp = rospy.Time.now()
            ros_img.header.frame_id = BASE_LINK
            self.image_pub.publish(ros_img)

    def compute_repulsive_force(
        self, point: np.ndarray, obstacles: np.ndarray, influence_range=1.0
    ) -> np.ndarray:
        rep_force = np.zeros(2)
        if obstacles is None:
            return rep_force
        for obs in obstacles:
            vec = point - obs
            dist = np.linalg.norm(vec)
            if dist < 1e-6 or dist > influence_range:
                continue
            rep_force += (1.0 / dist**3) * (vec / dist)
        return rep_force

    def apply_repulsive_forces_to_trajectories(
        self, trajectories: np.ndarray
    ) -> np.ndarray:
        if self.obstacle_points is None or len(self.obstacle_points) == 0:
            return trajectories * (MAX_V / RATE)

        updated_trajs = trajectories.copy()
        for i in range(updated_trajs.shape[0]):
            max_force = np.zeros(2)
            max_magnitude = 0.0

            for j in range(updated_trajs.shape[1]):
                pt = updated_trajs[i, j] * (MAX_V / RATE)
                rep_force = self.compute_repulsive_force(pt, self.obstacle_points)
                mag = np.linalg.norm(rep_force)
                if mag > max_magnitude:
                    max_magnitude = mag
                    max_force = rep_force

            angle = np.arctan2(max_force[1], max_force[0])
            angle = np.clip(angle, -np.pi / 4, np.pi / 4)
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            updated_trajs[i] = (rotation_matrix @ updated_trajs[i].T).T

        return updated_trajs  # * (MAX_V / RATE)

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return np.pi
        return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

    def _select_closest_traj_angle(
        self, trajs: np.ndarray, default_idx: int = 0
    ) -> int:
        if self.obstacle_points is None or len(self.obstacle_points) == 0:
            return default_idx

        prev_wp = self.current_waypoint

        if np.linalg.norm(prev_wp) < 1e-3:
            return default_idx

        cand_wps = trajs[:, self.args.waypoint]
        angles = np.array([self._angle_between(prev_wp, wp) for wp in cand_wps])
        return int(np.argmin(angles))

    def _timer_cb(self, event):
        if len(self.context_queue) <= self.context_size:
            return

        # 모델 타입에 따라 다른 처리
        if self.model_params["model_type"] == "nomad":
            self._timer_cb_nomad()

        # 목표 도달 시 로그 출력
        if self.closest_node == self.goal_node:
            rospy.loginfo("Reached goal! Stopping...")

    # Publish helpers
    def _publish_goal_images(self, sg_img: PILImage.Image, goal_img: PILImage.Image):
        """Publish current sub‑goal an d final goal images as ROS sensor_msgs/Image."""
        for img, pub in [(sg_img, self.subgoal_pub), (goal_img, self.goal_pub_img)]:
            cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)

    def _publish_msgs(self, traj_batch: np.ndarray, chosen: np.ndarray):
        # sampled actions
        actions_msg = Float32MultiArray()
        actions_msg.data = [0.0] + [float(x) for x in traj_batch.flatten()]
        self.sampled_actions_pub.publish(actions_msg)

        # chosen waypoint
        # rospy.loginfo(f"Publishing waypoint: {chosen}")
        wp_msg = Float32MultiArray()
        wp_msg.data = [float(chosen[0]), float(chosen[1]), 0.0, 0.0]
        self.waypoint_pub.publish(wp_msg)

        # goal status
        reached = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached))


    def _publish_sg_cnode_goal_imgs(self, topomap, closest_node, end, goal_node, model_params, crop=True):

        goal_img = transform_images(topomap[goal_node], model_params["image_size"], center_crop=crop, return_img=True)
        goal_img_msg = pil_to_msg(goal_img)
        goal_img_msg.header.stamp = rospy.Time.now()
        goal_img_msg.header.frame_id = "base_footprint"
        goal_img_msg.encoding = "rgb8"
        self.goal_img_pub.publish(goal_img_msg)

        subgoal_img = transform_images(topomap[end], model_params["image_size"], center_crop=crop, return_img=True)
        subgoal_img_msg = pil_to_msg(subgoal_img)
        subgoal_img_msg.header.stamp = rospy.Time.now()
        subgoal_img_msg.header.frame_id = "base_footprint"
        subgoal_img_msg.encoding = "rgb8"
        self.subgoal_img_pub.publish(subgoal_img_msg)


        closest_node_img = transform_images(topomap[closest_node], model_params["image_size"], center_crop=crop, return_img=True)
        closest_node_img_msg = pil_to_msg(closest_node_img)
        closest_node_img_msg.header.stamp = rospy.Time.now()
        closest_node_img_msg.header.frame_id = "base_footprint"
        closest_node_img_msg.encoding = "rgb8"
        self.closest_node_img_pub.publish(closest_node_img_msg)


    def _timer_cb_nomad(self):
        """NOMAD 모델을 위한 타이머 콜백 처리 (APF 포함)"""
        # -----------------------------------------------------------------
        # 1. Compute closest node via distance prediction
        # -----------------------------------------------------------------
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)

        # Build batch of (obs, goal) tensors
        obs_images = transform_images(
            list(self.context_queue),
            self.model_params["image_size"],
            center_crop=False,
        ).to(self.device)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)  # merge context

        batch_goal_imgs = []
        for g_idx in range(start, end + 1):
            g_img = transform_images(
                self.topomap[g_idx], self.model_params["image_size"], center_crop=False
            )
            batch_goal_imgs.append(g_img)
        goal_tensor = torch.cat(batch_goal_imgs, dim=0).to(self.device)

        mask = torch.zeros(1, device=self.device, dtype=torch.long)

        self._publish_sg_cnode_goal_imgs(topomap=self.topomap, 
                                         closest_node=self.closest_node, 
                                         end=end, 
                                         goal_node=self.goal_node, 
                                         model_params=self.model_params, 
                                         crop=True)

        with torch.no_grad():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_tensor), 1, 1, 1),
                goal_img=goal_tensor,
                input_goal_mask=mask.repeat(len(goal_tensor)),
            )
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists_np = to_numpy(dists.flatten())

            distances_msg = Float32MultiArray()
            distances_msg.data = dists_np
            self.distances_pub.publish(distances_msg)

        min_idx = int(np.argmin(dists_np))
        self.closest_node = start + min_idx
        rospy.loginfo(f"Closest node: {self.closest_node}")
        closest_node_msg = Int32()
        closest_node_msg.data = self.closest_node
        self.closest_node_pub.publish(closest_node_msg)

        sg_idx = min(
            min_idx + int(dists_np[min_idx] < self.args.close_threshold),
            len(goal_tensor) - 1,
        )
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
        sg_global_idx = start + sg_idx
        sg_pil = self.topomap[sg_global_idx]
        goal_pil = self.topomap[self.goal_node]

        with torch.no_grad():
            if obs_cond.ndim == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            len_traj = self.model_params["len_traj_pred"]
            naction = torch.randn(
                (self.args.num_samples, len_traj, 2), device=self.device
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample

        traj_batch = to_numpy(get_action(naction))

        is_apf_applied = (
            self.obstacle_points is not None and len(self.obstacle_points) > 0
        )

        self.original_trajectories = traj_batch.copy()

        ma = MarkerArray()
        for idx, paths in enumerate(self.original_trajectories):
            r = 1.0
            g = 0.0
            b = 0.0
            marker = make_path_marker(paths, idx, r, g, b, frame_id="oak-d-base-frame") # base_link
            ma.markers.append(marker)
        self.path_pub.publish(ma)

        traj_batch = self.apply_repulsive_forces_to_trajectories(traj_batch)

        ma = MarkerArray()
        for idx, paths in enumerate(traj_batch):
            r = 0.0
            g = 0.0
            b = 1.0
            marker = make_path_marker(paths, idx, r, g, b, frame_id="oak-d-base-frame") # base_link
            ma.markers.append(marker)
        self.path_pub_care.publish(ma)

        chosen_idx = self._select_closest_traj_angle(traj_batch, default_idx=0)
        chosen_waypoint = traj_batch[chosen_idx][self.args.waypoint]
        self.current_waypoint = chosen_waypoint
        self._publish_msgs(traj_batch, chosen_waypoint)
        self._publish_viz_image(traj_batch, is_apf_applied)
        self._publish_goal_images(sg_pil, goal_pil)

    def _publish_viz_image(self, traj_batch: np.ndarray, is_apf_applied: bool = False):
        frame = np.array(self.context_queue[-1])  # latest RGB frame
        img_h, img_w = frame.shape[:2]
        viz = frame.copy()

        cx = img_w // 2
        cy = int(img_h * 0.95)

        pixels_per_m = 3.0
        lateral_scale = 1.0
        horizontal_scale = 1.0
        # lateral_scale = 16.0
        # horizontal_scale = 16.0
        robot_symbol_length = 10

        cv2.line(
            viz,
            (cx - robot_symbol_length, cy),
            (cx + robot_symbol_length, cy),
            (255, 0, 0),
            2,
        )  # Blue
        cv2.line(
            viz,
            (cx, cy - robot_symbol_length),
            (cx, cy + robot_symbol_length),
            (255, 0, 0),
            2,
        )  # Blue

        for i, traj in enumerate(traj_batch):
            pts = []
            pts.append((cx, cy))

            acc_x, acc_y = 0.0, 0.0
            for dx, dy in traj:
                acc_x += dx
                acc_y += dy
                if is_apf_applied:
                    px = int(cx - acc_y * pixels_per_m)
                    py = int(cy - acc_x * pixels_per_m)
                else:
                    px = int(cx - acc_y * pixels_per_m * lateral_scale)
                    py = int(cy - acc_x * pixels_per_m * horizontal_scale)
                pts.append((px, py))

            if len(pts) >= 2:
                # Change colors when APF is applied
                if is_apf_applied:
                    color = (
                        (0, 0, 255) if i == 0 else (180, 0, 255)
                    )  # Blue for main, purple for others
                else:
                    color = (
                        (0, 255, 0) if i == 0 else (255, 200, 0)
                    )  # Original green and yellow
                cv2.polylines(viz, [np.array(pts, dtype=np.int32)], False, color, 2)

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = rospy.Time.now()
        self.viz_pub.publish(img_msg)

    def _run(self):
        rospy.spin()


def main():
    parser = argparse.ArgumentParser("Topological navigation with APF (ROS 2)")
    parser.add_argument("--model", "-m", default="nomad")
    parser.add_argument(
        "--dir",
        "-d",
        default="mist_office",
        help="sub‑directory under ../topomaps/images/",
    )
    parser.add_argument(
        "--goal-node", "-g", type=int, default=-1, help="Goal node index (-1 = last)"
    )
    parser.add_argument("--waypoint", "-w", type=int, default=2)
    parser.add_argument("--close-threshold", "-t", type=float, default=3.0)
    parser.add_argument("--radius", "-r", type=int, default=4)
    parser.add_argument("--num-samples", "-n", type=int, default=8)

    args = parser.parse_args()

    # rclpy.init()
    try:
        node = NavigationNode(args)
        node._run()
    except KeyboardInterrupt:
        exit()

if __name__ == "__main__":
    main()
