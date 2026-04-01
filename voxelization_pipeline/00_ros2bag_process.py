# ROS2 bag (mcap) to NPZ exporter
# Run this script in a ROS2 environment where rosbag2_py and cv_bridge are available.

import argparse
import numpy as np
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def get_point_cloud_from_rgbd(color_img, depth_img, intrinsic):
    """
    Convert RGB-D image to point cloud + RGB features.
    intrinsic: [fx, fy, cx, cy]
    """
    fx, fy, cx, cy = intrinsic
    height, width = depth_img.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    depth = depth_img.astype(float) / 1000.0
    valid = depth > 0

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    coords = np.stack([x, y, z], axis=-1)
    features = color_img[valid].astype(float) / 255.0
    return coords, features


def export_ros2_bag_to_npz(
    bag_path,
    output_npz,
    color_topic,
    depth_topic,
    info_topic,
    max_frames=None,
):
    bridge = CvBridge()

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    required_topics = [color_topic, depth_topic, info_topic]
    missing_topics = [topic for topic in required_topics if topic not in topic_types]
    if missing_topics:
        raise ValueError(f"Missing topics in ROS2 bag: {missing_topics}")

    color_msg_type = get_message(topic_types[color_topic])
    depth_msg_type = get_message(topic_types[depth_topic])
    info_msg_type = get_message(topic_types[info_topic])

    intrinsic = None
    last_color = None
    coords_frames = []
    features_frames = []
    rgb_frames = []

    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == info_topic and intrinsic is None:
            msg = deserialize_message(data, info_msg_type)
            k = getattr(msg, "k", None)
            if k is None:
                k = getattr(msg, "K", None)
            if k is None:
                raise AttributeError("CameraInfo has neither 'k' nor 'K' field")
            intrinsic = [k[0], k[4], k[2], k[5]]
            continue

        if topic == color_topic:
            msg = deserialize_message(data, color_msg_type)
            last_color = bridge.imgmsg_to_cv2(msg, "rgb8")
            continue

        if topic == depth_topic and last_color is not None and intrinsic is not None:
            msg = deserialize_message(data, depth_msg_type)
            depth_img = bridge.imgmsg_to_cv2(msg, "16UC1")
            coords, rgb_features = get_point_cloud_from_rgbd(last_color, depth_img, intrinsic)

            coords_frames.append(coords.astype(np.float32))
            features_frames.append(rgb_features.astype(np.float32))
            rgb_frames.append(last_color.astype(np.uint8).copy())

            if max_frames is not None and len(coords_frames) >= max_frames:
                break

    if intrinsic is None:
        raise ValueError(f"Camera info topic not found or empty: {info_topic}")

    np.savez_compressed(
        output_npz,
        coords_list=np.array(coords_frames, dtype=object),
        features_list=np.array(features_frames, dtype=object),
        rgb_frames=np.array(rgb_frames, dtype=object),
        intrinsic=np.array(intrinsic, dtype=np.float32),
        color_topic=color_topic,
        depth_topic=depth_topic,
        info_topic=info_topic,
    )

    print(f"Saved {len(coords_frames)} frames to {output_npz}")


def build_parser():
    parser = argparse.ArgumentParser(description="Export ROS2 mcap bag folder to NPZ point clouds")
    parser.add_argument("--bag-path", required=True, help="ROS2 bag folder path (contains metadata.yaml and .mcap)")
    parser.add_argument("--output-npz", required=True, help="Output npz path")
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    export_ros2_bag_to_npz(
        bag_path=args.bag_path,
        output_npz=args.output_npz,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        info_topic=args.info_topic,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
