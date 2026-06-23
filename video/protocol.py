"""Shared wire protocol for the Pi camera server and desktop client.

Pure standard library so it can run unchanged on the Raspberry Pi and the
desktop. Messages are newline-delimited UTF-8 JSON objects.
"""
import json

PING = "PING"
START_SESSION = "START_SESSION"
BOOKMARK = "BOOKMARK"
STOP_SESSION = "STOP_SESSION"
GET_FILE = "GET_FILE"


def encode_message(msg: dict) -> bytes:
    """Serialize a message dict to a newline-terminated UTF-8 JSON frame."""
    return (json.dumps(msg) + "\n").encode("utf-8")


def decode_message(line: bytes) -> dict:
    """Parse one JSON frame; a trailing newline is optional."""
    return json.loads(line.decode("utf-8").rstrip("\n"))


def make_request(cmd: str, **params) -> dict:
    """Build a request message: {"cmd": cmd, ...params}."""
    return {"cmd": cmd, **params}


def make_ok(**fields) -> dict:
    """Build a success response: {"ok": True, ...fields}."""
    return {"ok": True, **fields}


def make_error(message: str) -> dict:
    """Build a failure response: {"ok": False, "error": message}."""
    return {"ok": False, "error": message}
