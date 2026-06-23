"""Camera (Pi video) controls UI card.

Lets the operator enable concurrent video, point at the Pi, designate which
sensor's Start bookmarks the video, and test the connection.
"""
import solara
from utils import state


def test_connection() -> bool:
    """Ping the Pi camera server and update camera_status. Returns reachability."""
    try:
        client = state.make_camera_client()
        ok = client.ping()
    except Exception:
        ok = False
    state.camera_status.set("connected" if ok else "disconnected")
    return ok


@solara.component
def CameraControlsCard():
    """UI for configuring and testing concurrent Pi video capture."""
    with solara.Card(title="Video Capture (Pi Camera)",
                     style={"margin-bottom": "20px"}):
        solara.Switch(label="Enable concurrent video",
                      value=state.camera_enabled.value,
                      on_value=state.camera_enabled.set,
                      disabled=state.recording_all.value)

        if state.camera_enabled.value:
            with solara.Row(style={"gap": "10px"}):
                solara.InputText(label="Pi host/IP",
                                 value=state.camera_host.value,
                                 on_value=state.camera_host.set,
                                 disabled=state.recording_all.value)
                solara.InputInt(label="Port",
                                value=state.camera_port.value,
                                on_value=state.camera_port.set,
                                disabled=state.recording_all.value)

            solara.Select(
                label="Camera sensor (whose Start bookmarks the video)",
                value=state.camera_sensor_id.value,
                values=list(range(1, 25)),
                on_value=state.camera_sensor_id.set,
                disabled=state.recording_all.value)

            with solara.Row(style={"margin-top": "10px", "gap": "10px"}):
                solara.Button(label="Test connection", on_click=test_connection,
                              color="secondary",
                              disabled=state.recording_all.value)
                status = state.camera_status.value
                if status == "connected":
                    solara.Success("✓ Connected")
                elif status == "disconnected":
                    solara.Error("✗ Not reachable")
                else:
                    solara.Text("Status: unknown")
