"""Schemas for the Quest teleop topics.

Only the messages the existing Quest publisher / HUD wire up are modeled.
Each class is a thin frozen-ish container with ``read``/``write`` that
matches the order in which the Unity-side serializer walks the fields.

Field order matters — Unity's auto-generated serializers don't tag fields,
they just stream them. If anything here disagrees with the upstream
``Unity.Robotics.ROSTCPConnector`` field order, deserialization will quietly
read garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .wire import Reader, Writer


@dataclass
class Time:
    sec: int = 0
    nsec: int = 0

    @classmethod
    def read(cls, r: Reader) -> Time:
        return cls(sec=r.int32(), nsec=r.uint32())

    def write(self, w: Writer) -> None:
        w.int32(self.sec)
        w.uint32(self.nsec)


@dataclass
class Header:
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""

    @classmethod
    def read(cls, r: Reader) -> Header:
        stamp = Time.read(r)
        frame_id = r.string()
        return cls(stamp=stamp, frame_id=frame_id)

    def write(self, w: Writer) -> None:
        self.stamp.write(w)
        w.string(self.frame_id)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def read(cls, r: Reader) -> Point:
        return cls(x=r.float64(), y=r.float64(), z=r.float64())

    def write(self, w: Writer) -> None:
        w.float64(self.x)
        w.float64(self.y)
        w.float64(self.z)


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def read(cls, r: Reader) -> Quaternion:
        return cls(x=r.float64(), y=r.float64(), z=r.float64(), w=r.float64())

    def write(self, w: Writer) -> None:
        w.float64(self.x)
        w.float64(self.y)
        w.float64(self.z)
        w.float64(self.w)


@dataclass
class Pose:
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)

    @classmethod
    def read(cls, r: Reader) -> Pose:
        return cls(position=Point.read(r), orientation=Quaternion.read(r))

    def write(self, w: Writer) -> None:
        self.position.write(w)
        self.orientation.write(w)


@dataclass
class PoseStamped:
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)

    @classmethod
    def read(cls, r: Reader) -> PoseStamped:
        return cls(header=Header.read(r), pose=Pose.read(r))

    def write(self, w: Writer) -> None:
        self.header.write(w)
        self.pose.write(w)


@dataclass
class Twist:
    linear: Point = field(default_factory=Point)
    angular: Point = field(default_factory=Point)

    @classmethod
    def read(cls, r: Reader) -> Twist:
        return cls(linear=Point.read(r), angular=Point.read(r))

    def write(self, w: Writer) -> None:
        self.linear.write(w)
        self.angular.write(w)


@dataclass
class TwistStamped:
    header: Header = field(default_factory=Header)
    twist: Twist = field(default_factory=Twist)

    @classmethod
    def read(cls, r: Reader) -> TwistStamped:
        return cls(header=Header.read(r), twist=Twist.read(r))

    def write(self, w: Writer) -> None:
        self.header.write(w)
        self.twist.write(w)


@dataclass
class OVR2ROSInputs:
    """quest2ros/OVR2ROSInputs — controller button + stick state.

    Extended with ``thumb_stick_click`` at the tail so old Unity
    publishers that omit the byte still decode the first six fields
    cleanly (the bridge falls back to ``False`` if the body is short).
    """

    button_upper: bool = False
    button_lower: bool = False
    thumb_stick_horizontal: float = 0.0
    thumb_stick_vertical: float = 0.0
    press_index: float = 0.0
    press_middle: float = 0.0
    thumb_stick_click: bool = False

    @classmethod
    def read(cls, r: Reader) -> OVR2ROSInputs:
        button_upper = r.bool()
        button_lower = r.bool()
        thumb_stick_horizontal = r.float32()
        thumb_stick_vertical = r.float32()
        press_index = r.float32()
        press_middle = r.float32()
        try:
            thumb_stick_click = r.bool()
        except Exception:
            thumb_stick_click = False
        return cls(
            button_upper=button_upper,
            button_lower=button_lower,
            thumb_stick_horizontal=thumb_stick_horizontal,
            thumb_stick_vertical=thumb_stick_vertical,
            press_index=press_index,
            press_middle=press_middle,
            thumb_stick_click=thumb_stick_click,
        )

    def write(self, w: Writer) -> None:
        w.bool(self.button_upper)
        w.bool(self.button_lower)
        w.float32(self.thumb_stick_horizontal)
        w.float32(self.thumb_stick_vertical)
        w.float32(self.press_index)
        w.float32(self.press_middle)
        w.bool(self.thumb_stick_click)


@dataclass
class StringMsg:
    data: str = ""

    @classmethod
    def read(cls, r: Reader) -> StringMsg:
        return cls(data=r.string())

    def write(self, w: Writer) -> None:
        w.string(self.data)


@dataclass
class BoolMsg:
    data: bool = False

    @classmethod
    def read(cls, r: Reader) -> BoolMsg:
        return cls(data=r.bool())

    def write(self, w: Writer) -> None:
        w.bool(self.data)


@dataclass
class JointState:
    """sensor_msgs/JointState — name + position + velocity + effort arrays."""

    header: Header = field(default_factory=Header)
    name: list[str] = field(default_factory=list)
    position: list[float] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)

    @classmethod
    def read(cls, r: Reader) -> JointState:
        return cls(
            header=Header.read(r),
            name=r.string_array(),
            position=r.float64_array(),
            velocity=r.float64_array(),
            effort=r.float64_array(),
        )

    def write(self, w: Writer) -> None:
        self.header.write(w)
        w.string_array(self.name)
        w.float64_array(self.position)
        w.float64_array(self.velocity)
        w.float64_array(self.effort)


# Topic → schema. Anything not listed gets a best-effort log-and-drop.
QUEST_TOPIC_SCHEMAS: dict[str, type] = {
    "/q2r_right_hand_pose": PoseStamped,
    "/q2r_left_hand_pose": PoseStamped,
    "/q2r_right_hand_twist": TwistStamped,
    "/q2r_left_hand_twist": TwistStamped,
    "/q2r_right_hand_inputs": OVR2ROSInputs,
    "/q2r_left_hand_inputs": OVR2ROSInputs,
    "/teleop_data_collector/episode_event": StringMsg,
    "/uf850/real_control_enable": BoolMsg,
}


def encode(msg) -> bytes:
    """Serialize a message dataclass to its wire body."""
    w = Writer()
    msg.write(w)
    return w.to_bytes()


def decode(schema: type, body: bytes):
    """Deserialize ``body`` into an instance of ``schema``."""
    return schema.read(Reader(body))
