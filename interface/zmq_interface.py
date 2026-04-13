"""ZMQ transport helpers for YOLH control and inference."""

import pickle
from typing import Any, Optional

import numpy as np
import zmq

OBS_PORT = 5555
ACT_PORT = 5556

_NDARRAY_TAG = "__ndarray__"
_TUPLE_TAG = "__tuple__"


def _encode_message(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError("Object dtype arrays are not supported by ZMQ transport")
        array = np.ascontiguousarray(obj)
        return {
            _NDARRAY_TAG: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {key: _encode_message(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_encode_message(value) for value in obj]

    if isinstance(obj, tuple):
        return {_TUPLE_TAG: [_encode_message(value) for value in obj]}

    return obj


def _decode_message(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get(_NDARRAY_TAG):
            array = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
            return array.reshape(tuple(obj["shape"]))

        if _TUPLE_TAG in obj:
            return tuple(_decode_message(value) for value in obj[_TUPLE_TAG])

        return {key: _decode_message(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_decode_message(value) for value in obj]

    return obj


def _serialize_message(data: dict) -> bytes:
    encoded = _encode_message(data)
    return pickle.dumps(encoded, protocol=4)


def _deserialize_message(payload: bytes) -> dict:
    try:
        encoded = pickle.loads(payload)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to decode a legacy numpy pickle payload. Restart both "
            "arm_control.py and inference.py after updating zmq_interface.py."
        ) from exc
    return _decode_message(encoded)


class ZmqSender:
    """PUSH socket wrapper."""

    def __init__(self, port: int, host: str = None):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.SNDHWM, 3)
        if host:
            self._sock.connect(f"tcp://{host}:{port}")
        else:
            self._sock.bind(f"tcp://*:{port}")

    def send(self, data: dict):
        self._sock.send(_serialize_message(data))

    def close(self):
        self._sock.close()
        self._ctx.term()


class ZmqReceiver:
    """PULL socket wrapper."""

    def __init__(self, port: int, host: str = None):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.RCVHWM, 3)
        if host:
            self._sock.connect(f"tcp://{host}:{port}")
        else:
            self._sock.bind(f"tcp://*:{port}")

    def recv(self, timeout_ms: int = -1) -> Optional[dict]:
        """Receive one message, or None on timeout."""
        if timeout_ms >= 0:
            if self._sock.poll(timeout_ms):
                return _deserialize_message(self._sock.recv())
            return None
        return _deserialize_message(self._sock.recv())

    def recv_latest(self) -> Optional[dict]:
        """Drain the queue and return the newest message."""
        latest = None
        while self._sock.poll(0):
            latest = _deserialize_message(self._sock.recv())
        return latest

    def close(self):
        self._sock.close()
        self._ctx.term()
