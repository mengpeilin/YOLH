"""Abstract base class for robot interfaces."""

import numpy as np
from abc import ABC, abstractmethod


class RobotInterface(ABC):
    """
    Abstract interface for robot communication.
    Subclass and implement for your specific robot hardware.
    """

    @abstractmethod
    def get_observation(self) -> tuple:
        """Returns (rgb, depth) as numpy arrays from the camera.

        Returns:
            rgb: (H, W, 3) uint8 array.
            depth: (H, W) uint16 array in millimetres.
        """
        ...

    @abstractmethod
    def get_joint_angles(self) -> np.ndarray:
        """Returns (5,) array of arm joint angles in radians.

        Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
        """
        ...

    @abstractmethod
    def send_action(self, pos: np.ndarray, rot6d: np.ndarray, width: float):
        """Send a single action step to the robot (base frame).

        Args:
            pos: (3,) target TCP position in base frame [metres].
            rot6d: (6,) target TCP orientation as rotation-6d.
            width: target gripper width [metres].
        """
        ...

    def stop(self):
        """Stop the robot and release resources."""
        pass
