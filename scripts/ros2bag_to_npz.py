import numpy as np
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def export_ros2_bag_to_npz(
    bag_path: str,
    output_npz: str,
    color_topic: str,
    depth_topic: str,
    info_topic: str,
    max_frames: int = None,
):
    bridge = CvBridge()

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    required = [color_topic, depth_topic, info_topic]
    missing = [t for t in required if t not in topic_types]
    if missing:
        raise ValueError(f"Missing topics in ROS2 bag: {missing}")

    color_msg_type = get_message(topic_types[color_topic])
    depth_msg_type = get_message(topic_types[depth_topic])
    info_msg_type = get_message(topic_types[info_topic])

    intrinsic = None
    last_color = None
    last_color_ts = None
    rgb_frames = []
    depth_frames = []
    timestamps = []

    while reader.has_next():
        topic, data, ts = reader.read_next()

        if topic == info_topic and intrinsic is None:
            msg = deserialize_message(data, info_msg_type)
            k = getattr(msg, "k", None)
            if k is None:
                k = getattr(msg, "K", None)
            if k is None:
                raise AttributeError("CameraInfo has neither 'k' nor 'K' field")
            intrinsic = np.array([k[0], k[4], k[2], k[5]], dtype=np.float64)
            continue

        if topic == color_topic:
            msg = deserialize_message(data, color_msg_type)
            last_color = bridge.imgmsg_to_cv2(msg, "rgb8")
            last_color_ts = ts
            continue

        if topic == depth_topic and last_color is not None and intrinsic is not None:
            msg = deserialize_message(data, depth_msg_type)
            depth_img = bridge.imgmsg_to_cv2(msg, "16UC1")

            rgb_frames.append(last_color.copy())
            depth_frames.append(depth_img.copy())
            timestamps.append(last_color_ts)

            if max_frames is not None and len(rgb_frames) >= max_frames:
                break

    if intrinsic is None:
        raise ValueError(f"Camera info topic not found or empty: {info_topic}")
    if len(rgb_frames) == 0:
        raise ValueError("No aligned RGB-D frame pairs found")

    rgb_arr = np.array(rgb_frames, dtype=np.uint8)     # (N, H, W, 3)
    depth_arr = np.array(depth_frames, dtype=np.uint16) # (N, H, W)
    ts_arr = np.array(timestamps, dtype=np.int64)

    np.savez_compressed(
        output_npz,
        rgb=rgb_arr,
        depth=depth_arr,
        intrinsic=intrinsic,
        timestamps=ts_arr,
    )
    print(f"     Saved {len(rgb_frames)} frames to {output_npz}")
    return output_npz