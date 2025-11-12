from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@st.cache_data
def pca_var(data, n=40):
    pca = PCA(n_components=n)
    pca.fit(data)
    return pca.explained_variance_ratio_


@st.cache_data
def pca_x(data, n=40):
    pca = PCA(n_components=n)
    x = pca.fit_transform(data)
    return x


uploaded_file = st.file_uploader("Choose a Witec file", type=["mat"])
if uploaded_file is not None:
    data = loadmat(uploaded_file)
    data = [v for k, v in data.items() if k.startswith("Struct")][0]
else:
    root = Path(__file__).parent
    DATA = root / "data/P4-Raman-Scan-150x150um.mat"

    data = loadmat(DATA)
    data = data["Struct_ScanPiezo003SpecData1"]
size = np.append(data["imagesize"][0, 0][0], data["data"][0, 0].shape[1])
data = data["data"][0, 0]

with st.sidebar:
    sigma = st.number_input(
        "Gaussian filter sigma",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    n = st.number_input(
        "Number of PCA components",
        min_value=1,
        max_value=size[2],
        value=40,
    )
    n = int(n)
data = gaussian_filter1d(data, sigma=sigma, axis=1)

with st.expander("Ratios"):
    ratio = pca_var(data, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ratio))
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.cumsum(ratio)))
    fig.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig)

transformed = pca_x(data, n)
transformed = np.reshape(transformed, (size[0], size[1], n))
channel = st.number_input(
    "Channel",
    min_value=0,
    max_value=n - 1,
    value=0,
)
target = transformed[:, :, channel]
target = np.clip(
    a=target,
    a_min=np.percentile(target, 1),
    a_max=np.percentile(target, 99),
)
fig = go.Figure()
fig.add_trace(go.Heatmap(z=np.rot90(target), colorscale="Viridis"))
fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
st.plotly_chart(fig)

kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(
    transformed.reshape(-1, n)
    # data
)
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=np.rot90(np.reshape(np.array(kmeans.labels_), size[0:2])),
        colorscale="Viridis",
    )
)
fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
st.plotly_chart(fig)

y = st.number_input(
    "X",
    min_value=0,
    max_value=size[0] - 1,
    value=size[0] // 2,
)
x = st.number_input(
    "Y",
    min_value=0,
    max_value=size[1] - 1,
    value=size[1] // 2,
)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(size[2]),
        y=data.reshape(size[0], size[1], -1)[x, y, :],
        mode="lines",
        name=f"({x}, {y})",
    )
)
st.plotly_chart(fig)
