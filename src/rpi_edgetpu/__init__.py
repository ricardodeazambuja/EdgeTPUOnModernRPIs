"""rpi-edgetpu: Client library and setup CLI for Coral Edge TPU on modern Raspberry Pi OS."""

__version__ = "0.1.0"

from .client import EdgeTPUClient, EdgeTPUBusyError

__all__ = ["EdgeTPUClient", "EdgeTPUBusyError"]
