# Bookmark connect-latency fix: cache the resolved IPv4 per session

**Date:** 2026-07-25
**Status:** Approved (design)
**Scope:** `hardware/pi_camera.py` only (plus tests)

## Problem

Video↔trace sync on a full run (`raw_data_2026-07-24_12-02-14`, ACG-26-3) showed the
video running ~25 frames (~0.2s) ahead of the capacitance trace mid/late session,
after the earlier CFR-drift fix already removed the gross ~950-frame drift.

Root cause, confirmed by direct measurement from the recording desktop:

```
getaddrinfo('picamera0.local', 8770) -> [ fe80::… (IPv6 link-local, DEAD), 192.168.50.2 (IPv4, good) ]
connect IPv6 : FAILS after ~2024 ms   (tried first, every call)
connect IPv4 : OK in ~24 ms
hostname PING (what the recorder does): 2060–4749 ms, spread 2689 ms
direct-IP PING                         :    1–23 ms, spread 22 ms
```

`PiCameraClient._request` calls `socket.create_connection(('picamera0.local', 8770),
timeout=5.0)` on **every** request. `getaddrinfo` returns the dead IPv6 link-local
address first, so each call wastes ~2–4.7s connecting to it before falling back to
the working IPv4 in ~24ms. The **call-to-call variance** in that stall (2060 vs
4749 ms) differs between the Start and Stop bookmarks and skews the two-bookmark
`slope`, which is what shifts the video against the trace over the session.

Transport is **wired only** (wifi is disabled on the Pi; the "roaming wlan0 guest
SSID" comments in this file are stale). The Pi's reply itself is <1ms.

### What this fix does and does not claim

- **Confirmed wins:** bookmark round-trip drops from ~2–5s to ~20ms, removing the
  2–5s uncertainty window and the near-`timeout=5.0` stalls that risk bookmark
  retries / lost anchors (a prior run lost its anchor this way). `host_after`
  becomes a tight proxy for the Pi's bookmark-exec instant.
- **Not asserted here:** that this alone zeroes the 25-frame residual. Because the
  Pi reply is <1ms even now, a smaller secondary component (reply-latency variance
  under encoding load) may remain. That is validated only by re-recording after
  this lands, and is out of scope for this change.

## Design

All changes are inside `hardware/pi_camera.py`.

### Address cache on the client instance

- New instance field `self._addr` (default `None`): a cached `(family, socktype,
  proto, sockaddr)` tuple of the address that last connected successfully.
- The session-lived singleton `session_controls.camera_client` is created once at
  session start and reused for `start_session`, both bookmarks, `stop_session`, and
  `get_file`. Caching on the instance therefore resolves once per session; every
  later call takes the fast direct-IP path.

### `_connect()` helper (replaces the bare `create_connection` in `_request`)

```
CONNECT_TIMEOUT_S = 0.1   # per-address; IPv4 measured ~1–24 ms, 4x+ margin

def _connect(self):
    # 1. Fast path: reuse the cached address.
    if self._addr is not None:
        try:
            return self._connect_addr(self._addr)
        except OSError:
            self._addr = None      # stale (e.g. DHCP change) -> fall through, re-resolve

    # 2. Resolve, IPv4 before IPv6, connect the first that answers; cache it.
    infos = socket.getaddrinfo(self.host, self.port, proto=socket.IPPROTO_TCP)
    infos.sort(key=lambda i: i[0] != socket.AF_INET)   # AF_INET first
    last_exc = None
    for info in infos:
        try:
            sock = self._connect_addr(info)
            self._addr = info      # cache the working address
            return sock
        except OSError as exc:
            last_exc = exc
    raise last_exc if last_exc else OSError("no addresses for %s" % self.host)
```

- `_connect_addr(info)` opens a socket for `info`'s family, sets
  `CONNECT_TIMEOUT_S`, connects to its sockaddr, then restores `self.timeout` for
  the send/recv phase, and returns the connected socket.
- `_request` wraps `self._connect()` in the existing `try/except Exception ->
  make_error(...)` so callers still get an error dict, never an exception. The
  per-request full timeout (`self.timeout`) continues to govern send/recv; only the
  connect phase uses the short per-address timeout.

### Comment cleanup

Replace the stale "roaming wlan0 on a guest SSID" wording in `_request`'s except
block with an accurate note: transport is the wired link; name resolution can list
a dead IPv6 link-local first, which is why the connected address is cached.

## Error handling

- Connect failure → same `make_error(f"{exc} [{host}:{port}]")` dict as today.
- A cached address that stops connecting (IP reassigned mid-session) self-heals:
  the failed fast-path connect clears `self._addr` and the same `_connect` call
  re-resolves once.
- If both IPv4 and IPv6 fail (Pi down), `_connect` raises, `_request` returns the
  error dict, and `_bookmark_with_retry` retries exactly as it does now.

## Testing

Mirror `tests/test_bookmark_retry.py` (monkeypatch) and
`tests/test_hardware_camera_persistence.py` (live loopback socket).

1. **IPv4-first, skip dead IPv6:** monkeypatch `getaddrinfo` to return
   `[IPv6-dead, IPv4-good]`; a fake connector records attempt order. Assert the
   IPv4 address is tried first (the sort puts `AF_INET` ahead of the dead IPv6
   link-local) and the client connects to it without ever attempting the IPv6 one.
   A separate case where the IPv4 connect is made to time out asserts the fallback
   then abandons it within the short per-address timeout and moves on.
2. **Resolve once, then cache:** two successive `_request` calls; assert
   `getaddrinfo` is called exactly once and the second call connects straight to the
   cached address.
3. **Stale-cache re-resolve:** prime the cache, then make the cached address fail;
   assert exactly one re-resolution and a successful connect to the new address.
4. **Error path unchanged:** all addresses fail → `_request` returns a
   `make_error` dict (no exception), so `_bookmark_with_retry` still retries.
5. **Live loopback:** a real `socketserver` on `127.0.0.1` answers `PING`; assert a
   normal `_request` round-trips and populates `self._addr`.

## Out of scope

- Reply-latency variance under encoding load (possible secondary residual).
- Any change to the desktop callers, `_bookmark_with_retry`, the anchor math in
  `video/trimcrop.py`, or the h5 schema.
- Switching the configured host away from `picamera0.local` (kept by decision).
- Pi-side IPv6/AAAA disabling (an ops option, not this code change).
