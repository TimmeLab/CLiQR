"""
Plot dialog component for displaying sensor test data.
"""
import solara
import matplotlib.pyplot as plt
from utils import state


@solara.component
def TestPlotDialog():
    """
    Dialog that displays a plot of recent sensor data.
    """
    if not state.show_test_dialog.value or state.test_plot_data.value is None:
        return

    plot_data = state.test_plot_data.value
    sensor_id = plot_data['sensor_id']
    cap_data = plot_data['cap_data']
    time_data = plot_data['time_data']

    # Get sensor's animal ID if available
    sensor = state.sensor_states.value.get(sensor_id)
    animal_text = f" [{sensor.animal_id}]" if sensor and sensor.animal_id else ""

    def close_dialog():
        state.show_test_dialog.set(False)
        state.test_plot_data.set(None)

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

        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 4))

        # Convert time_data to relative seconds if they're timestamps
        if len(time_data) > 0:
            time_rel = [t - time_data[0] for t in time_data]
            ax.plot(time_rel, cap_data, linewidth=1.5, color='#2196F3')
            ax.set_xlabel('Time (seconds)', fontsize=12)
        else:
            ax.plot(cap_data, linewidth=1.5, color='#2196F3')
            ax.set_xlabel('Sample', fontsize=12)

        ax.set_ylabel('Capacitance (picofarads)', fontsize=12)
        ax.set_title(f'Sensor {sensor_id}{animal_text} - Last {len(cap_data)} Samples', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Display the plot
        solara.FigureMatplotlib(fig)
        plt.close(fig)  # Clean up

        # Close button
        with solara.Row(style={"justify-content": "center", "margin-top": "10px"}):
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
