import numpy as np
import streamlit as st
from plotly import graph_objects as go


def explore_fwhm(
    data,
    size,
    peak: int = 0,
    feature: int = 0,
):
    plot_data = np.rot90(np.reshape(np.array(data[:, peak, feature]), size[0:2]))
    col1, col2 = st.columns(2)
    if col1.toggle("Clip outliers", value=True):
        percentile_delta = col2.number_input(
            "Percentile delta",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.001,
        )
        plot_data = np.clip(
            plot_data,
            np.quantile(plot_data, percentile_delta),
            np.quantile(plot_data, 1 - percentile_delta),
        )
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=plot_data,
            colorscale="Viridis",
        )
    )
    fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig)
