"""Structured error code catalog. Codes follow the `FARM-Exxxx` pattern.

The catalog grows in task 008. This module ships only the codes the config
loader / doctor reference today; later tasks expand it.
"""

from enum import Enum


class ErrorCode(str, Enum):
    NO_CAMERA = "FARM-E1001"
    CALIBRATION_STALE = "FARM-E1002"
    GPU_COLD_START = "FARM-E1003"
    API_KEY_REJECTED = "FARM-E1004"
    WS_DROPPED = "FARM-E1005"
    VERSION_MISMATCH = "FARM-E1006"
    NETWORK_BLOCKED = "FARM-E1007"
    DRIVER_REQUIRES_ARM_IP = "FARM-E1008"
    CONFIG_NOT_FOUND = "FARM-E1009"
    ENV_VAR_MISSING = "FARM-E1010"
    CAPABILITY_CARD_INVALID = "FARM-E2001"
    SAFETY_ENVELOPE = "FARM-E3001"
    WATCHDOG_TIMEOUT = "FARM-E3002"
