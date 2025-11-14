import numpy as np
import streamlit as st
from plotly import graph_objects as go


def spectrum_to_image(data, wavenumber, size, percentile_delta=1e-3):
    average = np.mean(data, axis=0)

    fig_spectra = go.Figure()
    fig_spectra.add_trace(
        go.Scatter(
            x=wavenumber,
            y=average,
            mode="markers",
            marker=dict(size=2),
            name="Average Spectrum",
        )
    )
    fig_spectra.update_layout(
        xaxis=dict(
            title="Wavenumber",
            ticksuffix="cm⁻¹",
        ),
        yaxis=dict(type="log"),
    )
    state = st.plotly_chart(fig_spectra, on_select="rerun")

    selected_channel = 0
    if len(state["selection"]["point_indices"]) > 0:
        selected_channel = state["selection"]["point_indices"][0]
    st.write("Selected channel:", selected_channel)

    channel_data = np.rot90(np.reshape(np.array(data[:, selected_channel]), size[0:2]))
    if st.toggle("Normalized selection", value=True):
        channel_data -= np.mean(channel_data)
        channel_data /= np.std(channel_data)
    channel_data = np.clip(
        channel_data,
        np.quantile(channel_data, percentile_delta),
        np.quantile(channel_data, 1 - percentile_delta),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=channel_data,
            colorscale="Viridis",
        )
    )
    fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig)


def image_to_spectrum(data, wavenumber, size, percentile_delta=1e-3):
    average = np.rot90(np.reshape(np.mean(data, axis=1), size[0:2]))
    average = np.clip(
        average,
        np.quantile(average, percentile_delta),
        np.quantile(average, 1 - percentile_delta),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=average,
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Average",
        height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    state = st.plotly_chart(fig, on_select="rerun", selection_mode="box")

    x_l = 0
    x_r = size[0] - 1
    y_d = 0
    y_u = size[1] - 1
    if len(state["selection"]["box"]) > 0:
        points = state["selection"]["box"][0]
        x_l = int(np.ceil(points["x"][0]))
        x_r = int(np.floor(points["x"][1]))
        y_u = int(np.ceil(points["y"][0]))
        y_d = int(np.floor(points["y"][1]))

    cube = np.reshape(data, (size[0], size[1], size[2]))
    cube = cube[x_l:x_r, y_d:y_u, :]
    spectrum = np.mean(cube, axis=(0, 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wavenumber,
            y=spectrum,
            name="Selected Spectrum",
        )
    )
    fig.update_layout(
        xaxis=dict(
            title="Wavenumber",
            ticksuffix="cm⁻¹",
        ),
        yaxis=dict(type="log"),
    )
    state = st.plotly_chart(fig)
