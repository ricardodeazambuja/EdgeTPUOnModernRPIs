"""rpi-edgetpu: Client library and setup CLI for Coral Edge TPU on modern Raspberry Pi OS."""

__version__ = "0.3.1"

from .client import EdgeTPUClient, EdgeTPUError, EdgeTPUBusyError

__all__ = ["EdgeTPUClient", "EdgeTPUError", "EdgeTPUBusyError"]
