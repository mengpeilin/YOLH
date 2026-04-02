import numpy as np


def keyframe_discovery(
    hand_positions: np.ndarray,   # (N, 3)
    hand_open_states: np.ndarray, # (N,) bool
    valid_mask: np.ndarray,       # (N,) bool - from hand pose
    stopping_delta: float = 0.005,  # threshold for "near zero velocity" in meters
    stopped_buffer_init: int = 4,
):
    """
    Detect keyframes using PerAct-style heuristic adapted for hand tracking.
    
    Keyframe criteria:
    1. Hand velocity is near zero (position change < stopping_delta)
    2. Gripper state changed, OR it's the last frame, OR hand is stopped
    """
    num_frames = len(hand_positions)
    if num_frames < 2:
        return [0] if num_frames == 1 else []

    # Compute per-frame velocity (position delta)
    velocities = np.zeros(num_frames)
    for i in range(1, num_frames):
        if valid_mask[i] and valid_mask[i-1]:
            velocities[i] = np.linalg.norm(hand_positions[i] - hand_positions[i-1])
        else:
            velocities[i] = 0.0  # Can't compute velocity without valid pose

    keyframes = []
    prev_gripper_open = hand_open_states[0]
    stopped_buffer = 0

    for i in range(num_frames):
        # Check if velocity is near zero
        small_delta = velocities[i] < stopping_delta

        # Check if gripper state hasn't changed in neighborhood (stable)
        gripper_stable = True
        if 0 < i < num_frames - 1:
            gripper_stable = (hand_open_states[i] == hand_open_states[max(0, i-1)] and
                              hand_open_states[i] == hand_open_states[min(num_frames-1, i+1)])

        # Determine if stopped
        is_last = (i == num_frames - 1)
        stopped = (stopped_buffer <= 0 and small_delta and
                   not is_last and gripper_stable)
        stopped_buffer = stopped_buffer_init if stopped else stopped_buffer - 1

        # A frame is a keyframe if:
        # - Gripper state changed
        # - Last frame
        # - Robot stopped (and not first frame)
        if i != 0 and (hand_open_states[i] != prev_gripper_open or is_last or stopped):
            keyframes.append(i)

        prev_gripper_open = hand_open_states[i]

    # Remove redundant consecutive keyframes
    if len(keyframes) > 1:
        if (keyframes[-1] - 1) == keyframes[-2]:
            keyframes.pop(-2)

    return keyframes


def detect_keyframes(
    hand_pose_path: str,
    hand_states_path: str,
    output_path: str,
    stopping_delta: float = 0.005,
):
    pose_data = np.load(hand_pose_path)
    positions = pose_data["positions"]    # (N, 3)
    valid = pose_data["valid"]            # (N,) bool

    hand_open = np.load(hand_states_path) # (N,) bool

    assert len(positions) == len(hand_open), \
        f"Frame count mismatch: poses={len(positions)}, states={len(hand_open)}"

    keyframes = keyframe_discovery(positions, hand_open, valid,
                                   stopping_delta=stopping_delta)

    keyframes = np.array(keyframes, dtype=np.int64)
    np.save(output_path, keyframes)
    print(f"Found {len(keyframes)} keyframes: {keyframes.tolist()}")
    print(f"Saved to {output_path}")
    return output_path