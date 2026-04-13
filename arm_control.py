#!/usr/bin/env python3
"""Control client for YOLH two-machine deployment."""

import argparse
import threading
import time

from policy.utils.config import load_config
from interface.lerobot_interface import RealSenseSO101Interface
from interface.zmq_interface import ACT_PORT, OBS_PORT, ZmqReceiver, ZmqSender


def _observation_loop(robot, obs_sender: ZmqSender, stop_event: threading.Event):
    try:
        while not stop_event.is_set():
            rgb, depth = robot.get_observation()
            joints = robot.get_joint_angles()
            obs_sender.send(
                {
                    "rgb": rgb,
                    "depth": depth,
                    "joint_angles": joints,
                    "timestamp": time.time(),
                }
            )
    except Exception as exc:
        if not stop_event.is_set():
            print(f"Observation loop stopped: {exc}")
            stop_event.set()


def _action_loop(robot, act_receiver: ZmqReceiver, stop_event: threading.Event):
    try:
        while not stop_event.is_set():
            data = act_receiver.recv(timeout_ms=100)
            if data is None:
                continue

            action = data.get("action")
            if action is None:
                continue
            print(f"Received action: {action}")
            robot.send_action(action[:3], action[3:9], float(action[9]))
    except Exception as exc:
        if not stop_event.is_set():
            print(f"Action loop stopped: {exc}")
            stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="LeRobot ZMQ control client")
    parser.add_argument("--config", required=True, help="Path to deployment YAML")
    parser.add_argument("--host", required=True, help="Inference server IP address")
    parser.add_argument("--obs-port", type=int, default=OBS_PORT)
    parser.add_argument("--act-port", type=int, default=ACT_PORT)
    parser.add_argument("--serial-port", default="/dev/ttyACM0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    robot = RealSenseSO101Interface(cfg, serial_port=args.serial_port)
    obs_sender = ZmqSender(args.obs_port, host=args.host)
    act_receiver = ZmqReceiver(args.act_port, host=args.host)
    stop_event = threading.Event()

    obs_thread = threading.Thread(
        target=_observation_loop,
        args=(robot, obs_sender, stop_event),
        daemon=True,
    )
    act_thread = threading.Thread(
        target=_action_loop,
        args=(robot, act_receiver, stop_event),
        daemon=True,
    )

    print(
        f"Control loop started obs_port={args.obs_port} act_port={args.act_port} "
        f"host={args.host}"
    )

    try:
        obs_thread.start()
        act_thread.start()

        while obs_thread.is_alive() and act_thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_event.set()
        obs_thread.join(timeout=1.0)
        act_thread.join(timeout=1.0)
        robot.stop()
        obs_sender.close()
        act_receiver.close()
        print("Controller stopped.")


if __name__ == "__main__":
    main()
