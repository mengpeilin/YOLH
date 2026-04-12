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
import queue
import threading

from policy.utils.config import load_config
from interface.so101_interface import RealSenseSO101Interface
from interface.zmq_interface import ZmqSender, ZmqReceiver, OBS_PORT, ACT_PORT


class LeRobotController:
    """Concurrent observation sender + action executor."""

    def __init__(self, robot, inference_host: str, obs_port: int, act_port: int):
        self.robot = robot
        self.action_queue: queue.Queue = queue.Queue(maxsize=200)
        self._running = False

        self.obs_sender = ZmqSender(obs_port, host=inference_host)
        self.act_receiver = ZmqReceiver(act_port, host=inference_host)

    # ── background thread: receive actions ──

    def _recv_loop(self):
        while self._running:
            data = self.act_receiver.recv(timeout_ms=100)
            if data is not None and data.get("action") is not None:
                self.action_queue.put(data["action"])

    # ── main control loop ──

    def run(self, max_steps: int, num_inference_step: int):
        self._running = True
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()
        print(f"Control loop started (max_steps={max_steps}, "
              f"inference_step={num_inference_step})")

        try:
            for t in range(max_steps):
                # send observation every N steps
                if t % num_inference_step == 0:
                    rgb, depth = self.robot.get_observation()
                    joints = self.robot.get_joint_angles()
                    self.obs_sender.send({
                        "rgb": rgb,
                        "depth": depth,
                        "joint_angles": joints,
                        "timestamp": t,
                    })

                # execute action from inference
                try:
                    action = self.action_queue.get(timeout=2.0)
                    self.robot.send_action(action[:3], action[3:9], float(action[9]))
                except queue.Empty:
                    print(f"[t={t}] Waiting for action ...")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.robot.stop()
        self.obs_sender.close()
        self.act_receiver.close()
        print("Controller stopped.")


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
