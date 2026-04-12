import threading

class InferenceState:
    def __init__(self):
        self.running = True
        self.latest_obs = None
        self.action_step = 0
        self.lock = threading.Lock()

    def update_observation(self, obs: dict):
        with self.lock:
            self.latest_obs = obs

    def get_latest_observation(self):
        with self.lock:
            return self.latest_obs

    def get_action_step(self) -> int:
        with self.lock:
            return self.action_step

    def next_action_step(self) -> int:
        with self.lock:
            step = self.action_step
            self.action_step += 1
            return step

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self) -> bool:
        with self.lock:
            return self.running