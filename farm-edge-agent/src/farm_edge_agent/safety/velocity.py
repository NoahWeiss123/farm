"""Joint and TCP velocity clamp for outbound action chunks."""

from __future__ import annotations

import math
from dataclasses import replace

from . import ActionChunk, Pose


class VelocityCap:
    """Clamp a chunk so no joint exceeds `joint_max` rad/s and the TCP linear
    speed between consecutive waypoints stays under `tcp_max_mps`.

    Joint velocities are clamped element-wise. TCP waypoint speed is computed
    from consecutive Euclidean distance over the chunk's `duration_s`; if it
    exceeds the cap, every waypoint after the first is pulled in toward its
    predecessor by the same scale factor.
    """

    def __init__(self, joint_max: float, tcp_max_mps: float) -> None:
        if joint_max <= 0:
            raise ValueError("joint_max must be > 0")
        if tcp_max_mps <= 0:
            raise ValueError("tcp_max_mps must be > 0")
        self._joint_max = joint_max
        self._tcp_max = tcp_max_mps

    def clamp(self, chunk: ActionChunk) -> tuple[ActionChunk, bool]:
        was_clamped = False

        new_joint_velocities = [list(v) for v in chunk.joint_velocities]
        for step in new_joint_velocities:
            for i, v in enumerate(step):
                if v > self._joint_max:
                    step[i] = self._joint_max
                    was_clamped = True
                elif v < -self._joint_max:
                    step[i] = -self._joint_max
                    was_clamped = True

        new_waypoints = list(chunk.tcp_waypoints)
        if len(new_waypoints) >= 2 and chunk.duration_s > 0:
            per_step = chunk.duration_s / max(1, len(new_waypoints) - 1)
            clamped: list[Pose] = [new_waypoints[0]]
            for prev, curr in zip(new_waypoints[:-1], new_waypoints[1:], strict=True):
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                dz = curr.z - prev.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                speed = dist / per_step
                if speed > self._tcp_max:
                    scale = self._tcp_max / speed
                    clamped.append(
                        Pose(
                            x=prev.x + dx * scale,
                            y=prev.y + dy * scale,
                            z=prev.z + dz * scale,
                            rx=curr.rx,
                            ry=curr.ry,
                            rz=curr.rz,
                        )
                    )
                    was_clamped = True
                else:
                    clamped.append(curr)
            new_waypoints = clamped

        new_chunk = replace(
            chunk,
            joint_velocities=new_joint_velocities,
            tcp_waypoints=new_waypoints,
        )
        return new_chunk, was_clamped
