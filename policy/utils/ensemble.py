"""Utilities for temporal action chunk buffering and inference runtime state."""

import threading

import numpy as np


class EnsembleBuffer:
    def __init__(self, mode: str = "act", k: float = 0.01, hold_last: bool = True):
        self.mode = mode
        self.k = k
        self.hold_last = hold_last
        self._chunks = []
        self._last_action = None
        self._lock = threading.Lock()

    def add_chunk(self, actions: np.ndarray, start_step: int):
        if actions is None or len(actions) == 0:
            return
        with self._lock:
            self._chunks.append((start_step, np.asarray(actions, dtype=np.float32)))

    def get_action(self, step: int):
        with self._lock:
            self._chunks = [
                (start_step, actions)
                for start_step, actions in self._chunks
                if start_step + len(actions) > step
            ]

            candidates = []
            for start_step, actions in self._chunks:
                idx = step - start_step
                if 0 <= idx < len(actions):
                    candidates.append((start_step, actions[idx]))

            if not candidates:
                return None if not self.hold_last else self._last_action

            candidates.sort(key=lambda item: item[0])
            start_steps = np.array([item[0] for item in candidates], dtype=np.float32)
            action_list = np.stack([item[1] for item in candidates], axis=0)

            if self.mode == "old":
                action = action_list[0]
            elif self.mode == "new":
                action = action_list[-1]
            elif self.mode == "avg":
                action = action_list.mean(axis=0)
            else:
                weights = np.exp(-self.k * (step - start_steps))
                weights = weights / np.clip(weights.sum(), 1e-8, None)
                action = (action_list * weights[:, None]).sum(axis=0)

            self._last_action = action.astype(np.float32)
            return self._last_action