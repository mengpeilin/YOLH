"""Controller for two-machine LeRobot deployment."""

import queue
import threading

from interface.zmq_interface import ZmqSender, ZmqReceiver


class LeRobotController:
    """Concurrent observation sender + action executor."""

    def __init__(self, robot, inference_host: str, obs_port: int, act_port: int):
        self.robot = robot
        self.action_queue: queue.Queue = queue.Queue(maxsize=200)
        self._running = False

        self.obs_sender = ZmqSender(obs_port, host=inference_host)
        self.act_receiver = ZmqReceiver(act_port, host=inference_host)

    def _recv_loop(self):
        while self._running:
            data = self.act_receiver.recv(timeout_ms=100)
            if data is not None and data.get("action") is not None:
                self.action_queue.put(data["action"])

    def run(self, max_steps: int, num_inference_step: int):
        self._running = True
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()
        print(
            f"Control loop started (max_steps={max_steps}, "
            f"inference_step={num_inference_step})"
        )

        try:
            for t in range(max_steps):
                if t % num_inference_step == 0:
                    rgb, depth = self.robot.get_observation()
                    joints = self.robot.get_joint_angles()
                    self.obs_sender.send(
                        {
                            "rgb": rgb,
                            "depth": depth,
                            "joint_angles": joints,
                            "timestamp": t,
                        }
                    )

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
