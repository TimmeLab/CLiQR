from video import protocol


def test_encode_decode_roundtrip():
    msg = {"cmd": protocol.PING, "n": 1}
    line = protocol.encode_message(msg)
    assert line.endswith(b"\n")
    assert protocol.decode_message(line) == msg


def test_decode_tolerates_missing_newline():
    assert protocol.decode_message(b'{"cmd": "PING"}') == {"cmd": "PING"}


def test_make_request():
    assert protocol.make_request(protocol.BOOKMARK, sensor_id=3) == {
        "cmd": "BOOKMARK",
        "sensor_id": 3,
    }


def test_make_ok_and_error():
    assert protocol.make_ok(frame_index=5) == {"ok": True, "frame_index": 5}
    err = protocol.make_error("boom")
    assert err == {"ok": False, "error": "boom"}
