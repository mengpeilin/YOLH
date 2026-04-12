#!/usr/bin/env python3
"""LeRobot control client for two-machine deployment.

Captures RGB-D + joint state from local hardware, sends observations to a
remote inference server via ZMQ, receives per-step actions, and executes
them on the robot.

Usage:
    python lerobot_control.py --config configs/camera_calibration.yaml \
                              --host 192.168.1.100
"""

import argparse

from policy.utils.config import load_config
from interface.lerobot_controller import LeRobotController
from interface.lerobot_interface import RealSenseSO101Interface
from interface.zmq_interface import OBS_PORT, ACT_PORT


def main():
    parser = argparse.ArgumentParser(description="LeRobot ZMQ control client")
    parser.add_argument("--config", required=True, help="Path to deployment YAML")
    parser.add_argument("--host", required=True, help="Inference server IP address")
    parser.add_argument("--obs-port", type=int, default=OBS_PORT)
    parser.add_argument("--act-port", type=int, default=ACT_PORT)
    parser.add_argument("--serial-port", default="/dev/ttyACM0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-inference-step", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_steps = args.max_steps or cfg.get("max_steps", 300)
    num_inf_step = args.num_inference_step or cfg.get("num_inference_step", 20)

    robot = RealSenseSO101Interface(cfg, serial_port=args.serial_port)
    controller = LeRobotController(robot, args.host, args.obs_port, args.act_port)

    try:
        controller.run(max_steps, num_inf_step)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        controller.stop()


if __name__ == "__main__":
    main()
