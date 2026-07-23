"""Camera (Pi video) controls UI card.

Lets the operator enable concurrent video, point at the Pi, designate which
sensor's Start bookmarks the video, and test the connection.
"""
import threading

import solara
from utils import state

# Snapshot opens a throwaway camera on the Pi (create + configure + start +
# capture), which takes longer than a ping; use a generous timeout so a
# slow-but-successful capture isn't misreported as a failure.
CAMERA_SNAPSHOT_TIMEOUT = 15.0


def test_snapshot():
    """Capture one still from the Pi in a background thread; open the dialog.

    Runs off the render thread because opening the camera takes ~1-2 s and would
    otherwise freeze the UI. Sets snapshot_image on success, snapshot_error
    otherwise; snapshot_pending drives the dialog's loading state.
    """
    state.snapshot_error.set("")
    state.snapshot_image.set("")
    state.snapshot_pending.set(True)
    state.show_snapshot_dialog.set(True)

    def _worker():
        try:
            client = state.make_camera_client(timeout=CAMERA_SNAPSHOT_TIMEOUT)
            resp = client.snapshot()
        except Exception as exc:
            state.snapshot_error.set(str(exc))
            state.snapshot_pending.set(False)
            return
        if resp.get("ok"):
            state.snapshot_image.set(resp.get("image", ""))
        else:
            state.snapshot_error.set(resp.get("error", "snapshot failed"))
        state.snapshot_pending.set(False)

    threading.Thread(target=_worker, daemon=True).start()


def test_connection() -> bool:
    """Ping the Pi camera server and update camera_status. Returns reachability."""
    try:
        client = state.make_camera_client()
        ok = client.ping()
    except Exception:
        ok = False
    # Persist status so it survives a refresh (via set_session, not bare .set).
    state.set_session("camera_status", "connected" if ok else "disconnected")
    return ok


@solara.component
def CameraControlsCard():
    """UI for configuring and testing concurrent Pi video capture."""
    with solara.Card(title="Video Capture (Pi Camera)",
                     style={"margin-bottom": "20px"}):
        solara.Switch(label="Enable concurrent video",
                      value=state.camera_enabled.value,
                      on_value=lambda v: state.set_session("camera_enabled", v),
                      disabled=state.recording_all.value)

        if state.camera_disk_warning.value:
            solara.Warning(f"⚠ {state.camera_disk_warning.value}")

        if state.camera_stall_warning.value:
            solara.Warning(f"⚠ {state.camera_stall_warning.value}")

        if state.camera_enabled.value:
            with solara.Row(style={"gap": "10px"}):
                solara.InputText(label="Pi host/IP",
                                 value=state.camera_host.value,
                                 on_value=lambda v: state.set_session("camera_host", v),
                                 disabled=state.recording_all.value)
                solara.InputInt(label="Port",
                                value=state.camera_port.value,
                                on_value=lambda v: state.set_session("camera_port", v),
                                disabled=state.recording_all.value)

            solara.Select(
                label="Camera sensor (whose Start bookmarks the video)",
                value=state.camera_sensor_id.value,
                values=list(range(1, 25)),
                on_value=lambda v: state.set_session("camera_sensor_id", v),
                disabled=state.recording_all.value)

            with solara.Row(style={"margin-top": "10px", "gap": "10px"}):
                solara.Button(label="Test connection", on_click=test_connection,
                              color="secondary",
                              disabled=state.recording_all.value)
                solara.Button(label="Test snapshot", on_click=test_snapshot,
                              color="secondary",
                              disabled=state.recording_all.value)
                status = state.camera_status.value
                if status == "connected":
                    solara.Success("✓ Connected")
                elif status == "disconnected":
                    solara.Error("✗ Not reachable")
                else:
                    solara.Text("Status: unknown")


@solara.component
def SnapshotDialog():
    """Modal showing the latest camera test snapshot for an alignment check."""
    if not state.show_snapshot_dialog.value:
        return

    def close_dialog():
        state.show_snapshot_dialog.set(False)

    with solara.Card(
        title="Camera Snapshot (alignment check)",
        style={
            "position": "fixed", "top": "50%", "left": "50%",
            "transform": "translate(-50%, -50%)", "z-index": "1000",
            "min-width": "640px", "max-width": "90vw",
            "background": "white", "box-shadow": "0 4px 6px rgba(0,0,0,0.1)",
        },
    ):
        if state.snapshot_pending.value:
            solara.Text("Capturing… (opening camera on the Pi)")
        elif state.snapshot_error.value:
            solara.Error(f"Snapshot failed: {state.snapshot_error.value}")
        elif state.snapshot_image.value:
            solara.HTML(
                tag="img",
                attributes={
                    "src": f"data:image/jpeg;base64,{state.snapshot_image.value}",
                    "style": "max-width:100%; height:auto; display:block;",
                },
            )
        else:
            solara.Text("No image.")

        with solara.Row(style={"margin-top": "10px", "gap": "10px"}):
            solara.Button(label="Refresh", on_click=test_snapshot,
                          color="primary",
                          disabled=state.snapshot_pending.value)
            solara.Button(label="Close", on_click=close_dialog)
