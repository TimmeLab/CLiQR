"""PiCameraClient must not pay the dead-IPv6-link-local connect stall on every
request: resolve once, prefer IPv4, cache the working address for the session.

getaddrinfo('picamera0.local') lists fe80::… (IPv6 link-local, dead) BEFORE the
wired 192.168.50.2, so a fresh resolve+connect per call wasted ~2-5s. See
docs/superpowers/specs/2026-07-25-bookmark-connect-latency-design.md.
"""
import socket

import pytest

from hardware import pi_camera
from hardware.pi_camera import PiCameraClient
from video import protocol

# getaddrinfo 5-tuples in the order the real resolver returns them: dead IPv6
# link-local first, working wired IPv4 second.
IPV6_DEAD = (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             ("fe80::3ecc:2efd:e24b:cc5", 8770, 0, 10))
IPV4_GOOD = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             ("192.168.50.2", 8770))


class _FakeSock:
    """Stand-in for a connected socket: context-manager + PING responder."""
    def __init__(self, sockaddr):
        self.sockaddr = sockaddr
        self.sent = b""
        self.closed = False
    def sendall(self, data):
        self.sent += data
    def recv(self, _n):
        return protocol.encode_message(protocol.make_ok(pong=True))
    def settimeout(self, _t):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *_a):
        self.closed = True


@pytest.fixture
def resolver(monkeypatch):
    """Patch getaddrinfo to return [IPv6-dead, IPv4-good] and count calls."""
    calls = {"n": 0}
    def fake_getaddrinfo(host, port, *a, **k):
        calls["n"] += 1
        return [IPV6_DEAD, IPV4_GOOD]
    monkeypatch.setattr(pi_camera.socket, "getaddrinfo", fake_getaddrinfo)
    return calls


def _patch_connect_addr(monkeypatch, client, dead_families=(socket.AF_INET6,),
                        record=None):
    """Replace _connect_addr with a fake: OSError for dead families, else a
    _FakeSock. Records the sockaddr of every attempt into `record` if given."""
    def fake(info):
        if record is not None:
            record.append(info)
        if info[0] in dead_families:
            raise OSError("connection refused")
        return _FakeSock(info[4])
    monkeypatch.setattr(client, "_connect_addr", fake)


def test_connect_tries_ipv4_first_and_never_the_dead_ipv6(resolver, monkeypatch):
    client = PiCameraClient("picamera0.local", 8770)
    attempts = []
    _patch_connect_addr(monkeypatch, client, record=attempts)

    sock = client._connect()

    assert sock.sockaddr == IPV4_GOOD[4]
    assert [a[0] for a in attempts] == [socket.AF_INET], \
        "IPv4 must be tried first and the dead IPv6 never reached"
    assert client._addr == IPV4_GOOD


def test_connect_resolves_once_then_uses_cache(resolver, monkeypatch):
    client = PiCameraClient("picamera0.local", 8770)
    _patch_connect_addr(monkeypatch, client)

    client._connect()
    client._connect()

    assert resolver["n"] == 1, "second connect must reuse the cached address"


def test_stale_cache_reresolves_once(resolver, monkeypatch):
    client = PiCameraClient("picamera0.local", 8770)
    client._addr = IPV4_GOOD  # primed, but now dead
    attempts = []
    # The cached IPv4 fails this run; a fresh resolve must connect again.
    _patch_connect_addr(monkeypatch, client,
                        dead_families=(socket.AF_INET6,), record=attempts)
    # Make the FIRST attempt (the cached one) fail, later ones succeed.
    real_fake = client._connect_addr
    state = {"first": True}
    def failing_then_ok(info):
        if state["first"]:
            state["first"] = False
            raise OSError("stale cached ip")
        return real_fake(info)
    monkeypatch.setattr(client, "_connect_addr", failing_then_ok)

    sock = client._connect()

    assert sock.sockaddr == IPV4_GOOD[4]
    assert resolver["n"] == 1, "exactly one re-resolution after the stale cache"


def test_connect_falls_back_when_ipv4_times_out(resolver, monkeypatch):
    client = PiCameraClient("picamera0.local", 8770)
    attempts = []
    # IPv4 dead this time -> must fall through to IPv6 and return it.
    _patch_connect_addr(monkeypatch, client,
                        dead_families=(socket.AF_INET,), record=attempts)

    sock = client._connect()

    assert sock.sockaddr == IPV6_DEAD[4]
    assert [a[0] for a in attempts] == [socket.AF_INET, socket.AF_INET6]
    assert client._addr == IPV6_DEAD


def test_request_returns_error_dict_when_all_addresses_fail(monkeypatch):
    client = PiCameraClient("picamera0.local", 8770)
    def boom():
        raise OSError("no route to host")
    monkeypatch.setattr(client, "_connect", boom)

    resp = client._request(protocol.make_request(protocol.PING))

    assert resp.get("ok") is not True
    assert "picamera0.local:8770" in resp.get("error", "")
