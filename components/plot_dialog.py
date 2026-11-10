"""
Plot dialog component for displaying sensor test data.

The recording loop shares Solara's asyncio event loop with UI rendering, so
anything slow done while rendering stalls sampling on every board. This dialog
therefore (1) renders client-side with ECharts -- the server only serializes a
few hundred points to JSON, no server-side rasterizing -- and (2) builds the
chart options only when the plotted data changes (use_memo), and reads no
frequently-updated reactive state (e.g. sensor_states, which the per-sensor
timers rewrite every second), so an open dialog does not re-render repeatedly.
"""
import solara
from utils import state

TRACE_COLORS = ['#2196F3', '#FF9800']


def build_echarts_option(traces, title):
    """Build an ECharts option dict for the test plot.

    Args:
        traces: list of {'label', 'cap_data', 'time_data'} dicts (one per sipper).
        title: chart title.

    Returns:
        JSON-serializable ECharts option dict. X is seconds relative to each
        trace's first timestamp, or sample index when time_data is empty.
    """
    series = []
    use_time = all(len(t['time_data']) > 0 for t in traces)
    for i, trace in enumerate(traces):
        cap = [float(c) for c in trace['cap_data']]
        if use_time:
            t0 = float(trace['time_data'][0])
            xs = [float(t) - t0 for t in trace['time_data']]
        else:
            xs = list(range(len(cap)))
        series.append({
            'name': trace['label'] or 'Capacitance',
            'type': 'line',
            'data': [[x, y] for x, y in zip(xs, cap)],
            'showSymbol': False,
            'animation': False,
            'lineStyle': {'width': 1.5},
            'color': TRACE_COLORS[i % len(TRACE_COLORS)],
        })

    option = {
        'title': {'text': title, 'left': 'center', 'textStyle': {'fontSize': 14}},
        'tooltip': {'trigger': 'axis'},
        'grid': {'left': 70, 'right': 20, 'top': 50, 'bottom': 50},
        'xAxis': {
            'type': 'value',
            'name': 'Time (seconds)' if use_time else 'Sample',
            'nameLocation': 'middle',
            'nameGap': 30,
        },
        'yAxis': {
            'type': 'value',
            'name': 'Capacitance (picofarads)',
            'nameLocation': 'middle',
            'nameGap': 55,
            'scale': True,
        },
        'animation': False,
        'series': series,
    }
    if traces[0]['label'] is not None:
        option['legend'] = {'top': 25}
    return option


@solara.component
def TestPlotDialog():
    """
    Dialog that displays a plot of recent sensor data.
    """
    plot_data = state.test_plot_data.value

    # Hooks must run unconditionally; rebuild only when new data is fetched.
    option = solara.use_memo(
        lambda: None if plot_data is None else build_echarts_option(
            plot_data['traces'],
            f"Sensor {plot_data['sensor_id']}{plot_data.get('animal_text', '')}"
            f" - Last {len(plot_data['traces'][0]['cap_data'])} Samples"),
        dependencies=[plot_data],
    )

    if not state.show_test_dialog.value or plot_data is None:
        return

    sensor_id = plot_data['sensor_id']
    animal_text = plot_data.get('animal_text', '')

    def close_dialog():
        state.show_test_dialog.set(False)
        state.test_plot_data.set(None)

    def refresh_dialog():
        # Re-fetch latest samples for this sensor
        from components.sensor_card import test_sensor
        test_sensor(sensor_id)

    with solara.Card(
        style={
            "position": "fixed",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "z-index": "1000",
            "min-width": "600px",
            "max-width": "800px",
            "background": "white",
            "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
        }
    ):
        # Header
        with solara.Row(style={"justify-content": "space-between", "margin-bottom": "10px"}):
            solara.Text(
                f"Sensor {sensor_id}{animal_text} - Recent Data",
                style={"font-weight": "bold", "font-size": "16px"}
            )
            solara.Button(
                label="✕",
                on_click=close_dialog,
                style={"min-width": "30px", "padding": "5px"}
            )

        # Rendered in the browser; no server-side rasterizing on the event loop.
        solara.FigureEcharts(option=option, attributes={"style": "height: 400px; width: 100%"})

        # Action buttons
        with solara.Row(style={"justify-content": "center", "gap": "10px", "margin-top": "10px"}):
            solara.Button(
                label="Refresh",
                on_click=refresh_dialog,
                color="secondary"
            )
            solara.Button(
                label="Close",
                on_click=close_dialog,
                color="primary"
            )


    # Overlay background (to darken the rest of the page)
    if state.show_test_dialog.value:
        solara.HTML(
            tag="div",
            style={
                "position": "fixed",
                "top": "0",
                "left": "0",
                "width": "100%",
                "height": "100%",
                "background": "rgba(0, 0, 0, 0.5)",
                "z-index": "999"
            }
        )
