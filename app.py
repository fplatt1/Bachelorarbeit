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
    x = pca.fit(data)
    st.write(x)
    return pca.explained_variance_ratio_


@st.cache_data
def pca_x(data, n=40):
    pca = PCA(n_components=n)
    x = pca.fit_transform(data)
    st.write(x)
    return np.ascontiguousarray(x)


@st.cache_data
def pca_analyzis(data, n=40):
    pca_instance = PCA(n_components=n)
    pca_data = pca_instance.fit(data)
    return pca_data


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
        max_value=size[-1],
        value=40,
    )
    n = int(n)
data = gaussian_filter1d(data, sigma=sigma, axis=1)

pca_data = pca_analyzis(data, n)

with st.expander("Ratios"):
    # ratio = pca_var(data, n)
    ratio = pca_data.explained_variance_ratio_

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ratio))
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.cumsum(ratio)))
    fig.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig)

# transformed = pca_x(data, n)
transformed = pca_data.transform(data)
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

with st.sidebar:
    algorithm = st.selectbox(
        "Clustering algorithm",
        options=[None, "KMeans"],
    )
    match algorithm:
        case "KMeans":
            n_clusters = st.number_input(
                "Number of clusters",
                min_value=2,
                max_value=20,
                value=8,
            )
            clusters = KMeans(
                n_clusters=n_clusters,
                random_state=0,
                n_init="auto",
            ).fit(transformed.reshape(-1, n))
        case _:
            st.error("Unknown algorithm")
            st.stop()

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=np.rot90(np.reshape(np.array(clusters.labels_), size[0:2])),
        colorscale="Viridis",
    )
)
fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
st.plotly_chart(fig)

col1, col2 = st.columns(2)
y = col1.number_input("X", min_value=0, max_value=size[0] - 1, value=size[0] // 2)
x = col2.number_input("Y", min_value=0, max_value=size[1] - 1, value=size[1] // 2)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(size[-1]),
        y=data.reshape(size[0], size[1], -1)[x, y, :],
        mode="lines",
        name=f"({x}, {y})",
    )
)
for i in range(clusters.n_clusters):
    mask = clusters.labels_ == i
    centroid = np.mean(data.reshape(-1, size[-1])[mask, :], axis=0)
    fig.add_trace(
        go.Scatter(
            # x=np.arange(size[-1]),
            y=centroid,
            mode="lines",
            name=f"Centroid {i}",
        )
    )
st.plotly_chart(fig)

offset = 0
t = transformed.reshape(-1, n)
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=transformed[..., offset + 0].reshape(-1),
        y=transformed[..., offset + 1].reshape(-1),
        z=transformed[..., offset + 2].reshape(-1),
        mode="markers",
        marker=dict(
            size=2,
            color=clusters.labels_,
            colorscale="Viridis",
            opacity=0.8,
        ),
    )
)
fig.update_layout(
    height=1000,
)
st.plotly_chart(fig)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=transformed[..., offset + 0].reshape(-1),
        y=transformed[..., offset + 1].reshape(-1),
        mode="markers",
        marker=dict(
            size=2,
            color=clusters.labels_,
            colorscale="Viridis",
            opacity=0.8,
        ),
    )
)
fig.update_layout(
    height=1000,
)
st.plotly_chart(fig)
