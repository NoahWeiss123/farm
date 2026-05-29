"""ROS-TCP wire format primitives.

The Unity ``Unity.Robotics.ROSTCPConnector`` library frames every payload as::

    +----+--------------+----+----------------+
    | 4B |    topic     | 4B |     body       |
    | N  |  N UTF-8 B   | M  |    M bytes     |
    +----+--------------+----+----------------+

Both lengths are little-endian unsigned 32-bit. The body itself is a
ROS-style serialization of the message: primitives are little-endian,
strings are length-prefixed (4B LE) UTF-8 bytes, and variable-length
arrays carry a 4B LE count followed by element-by-element bytes.

This module provides:

* ``read_frame`` / ``write_frame`` — the outer framing.
* ``Reader`` / ``Writer`` — a small cursor pair for walking a body buffer.

Message-specific (de)serializers live in ``messages.py`` and use ``Reader``
/ ``Writer`` so the wire format stays in one place.
"""

from __future__ import annotations

import socket
import struct
from io import BytesIO


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise ``ConnectionError``."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[str, bytes]:
    """Read one (topic, body) frame from ``sock``.

    Raises ``ConnectionError`` when the peer hangs up cleanly between
    frames, ``ValueError`` on malformed lengths.
    """
    raw = read_exact(sock, 4)
    (topic_len,) = struct.unpack("<I", raw)
    if topic_len > 1024:
        raise ValueError(f"topic length absurd: {topic_len}")
    topic = read_exact(sock, topic_len).decode("utf-8")
    raw = read_exact(sock, 4)
    (body_len,) = struct.unpack("<I", raw)
    if body_len > 16 * 1024 * 1024:
        raise ValueError(f"body length absurd: {body_len}")
    body = read_exact(sock, body_len)
    return topic, body


def write_frame(sock: socket.socket, topic: str, body: bytes) -> None:
    """Serialize and send one (topic, body) frame on ``sock``."""
    topic_bytes = topic.encode("utf-8")
    sock.sendall(struct.pack("<I", len(topic_bytes)))
    sock.sendall(topic_bytes)
    sock.sendall(struct.pack("<I", len(body)))
    sock.sendall(body)


class Reader:
    """Cursor over a message body. Mirrors the Unity-side serializer order."""

    def __init__(self, body: bytes) -> None:
        self._buf = body
        self._pos = 0

    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def _take(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._buf):
            raise ValueError(f"read past end of body: need {n} have {self.remaining()}")
        out = self._buf[self._pos:end]
        self._pos = end
        return out

    def bool(self) -> bool:
        return self._take(1) != b"\x00"

    def int32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def uint32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def float32(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def float64(self) -> float:
        return struct.unpack("<d", self._take(8))[0]

    def string(self) -> str:
        n = self.uint32()
        return self._take(n).decode("utf-8")

    def float64_array(self) -> list[float]:
        n = self.uint32()
        return list(struct.unpack(f"<{n}d", self._take(8 * n)))

    def string_array(self) -> list[str]:
        n = self.uint32()
        return [self.string() for _ in range(n)]


class Writer:
    """Cursor for building a message body."""

    def __init__(self) -> None:
        self._buf = BytesIO()

    def to_bytes(self) -> bytes:
        return self._buf.getvalue()

    def bool(self, v: bool) -> None:
        self._buf.write(b"\x01" if v else b"\x00")

    def int32(self, v: int) -> None:
        self._buf.write(struct.pack("<i", v))

    def uint32(self, v: int) -> None:
        self._buf.write(struct.pack("<I", v))

    def float32(self, v: float) -> None:
        self._buf.write(struct.pack("<f", v))

    def float64(self, v: float) -> None:
        self._buf.write(struct.pack("<d", v))

    def string(self, v: str) -> None:
        data = v.encode("utf-8")
        self.uint32(len(data))
        self._buf.write(data)

    def float64_array(self, vs: list[float] | tuple[float, ...]) -> None:
        self.uint32(len(vs))
        if vs:
            self._buf.write(struct.pack(f"<{len(vs)}d", *vs))

    def string_array(self, vs: list[str]) -> None:
        self.uint32(len(vs))
        for v in vs:
            self.string(v)
