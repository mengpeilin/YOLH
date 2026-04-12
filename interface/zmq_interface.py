"""ZMQ communication for two-machine robot deployment.

Architecture:
    Control (camera + robot)
        ├─ ZmqSender(obs_port)   ──PUSH──▶  Inference ZmqReceiver(obs_port)
        └─ ZmqReceiver(act_port) ◀──PULL──  Inference ZmqSender(act_port)

Serialization uses pickle over a trusted local network.
"""

import pickle
from typing import Optional

import zmq

OBS_PORT = 5555
ACT_PORT = 5556


class ZmqSender:
    """ZMQ PUSH socket. Binds locally when *host* is None, connects to remote otherwise."""

    def __init__(self, port: int, host: str = None):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.SNDHWM, 3)
        if host:
            self._sock.connect(f"tcp://{host}:{port}")
        else:
            self._sock.bind(f"tcp://*:{port}")

    def send(self, data: dict):
        self._sock.send(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))

    def close(self):
        self._sock.close()
        self._ctx.term()


class ZmqReceiver:
    """ZMQ PULL socket. Binds locally when *host* is None, connects to remote otherwise."""

    def __init__(self, port: int, host: str = None):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.RCVHWM, 3)
        if host:
            self._sock.connect(f"tcp://{host}:{port}")
        else:
            self._sock.bind(f"tcp://*:{port}")

    def recv(self, timeout_ms: int = -1) -> Optional[dict]:
        """Blocking recv. Returns None on timeout when *timeout_ms* >= 0."""
        if timeout_ms >= 0:
            if self._sock.poll(timeout_ms):
                return pickle.loads(self._sock.recv())
            return None
        return pickle.loads(self._sock.recv())

    def recv_latest(self) -> Optional[dict]:
        """Drain queue and return only the most recent message."""
        latest = None
        while self._sock.poll(0):
            latest = pickle.loads(self._sock.recv())
        return latest

    def close(self):
        self._sock.close()
        self._ctx.term()
