from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from bachelorarbeit.plot.spectrum_and_image import image_to_spectrum, spectrum_to_image
from bachelorarbeit.reduction import fwhm_analyzis, pca_analyzis, pca_transformed
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn.cluster import KMeans

uploaded_file = st.file_uploader("Choose a Witec file", type=["mat"])
if uploaded_file is not None:
    data = loadmat(uploaded_file)
    data = [v for k, v in data.items() if k.startswith("Struct")][0]
else:
    root = Path(__file__).parent.parent
    DATA = root / "data/P4-Raman-Scan-150x150um.mat"

    data = loadmat(DATA)
    # st.write(data)
    data = data["Struct_ScanPiezo003SpecData1"]
    # print(data)
size = np.append(data["imagesize"][0, 0][0], data["data"][0, 0].shape[1])
wavenumber = np.array(data["axisscale"][0, 0][1, 0][0])
data = data["data"][0, 0]

with st.sidebar:
    sigma = st.number_input(
        "Gaussian filter sigma",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    pca_n = st.number_input(
        "Number of PCA components",
        min_value=1,
        max_value=size[-1],
        value=40,
    )
    pca_n = int(pca_n)
    prominence = st.number_input(
        "Peak prominence",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
data = gaussian_filter1d(data, sigma=sigma, axis=1)

pca_data = pca_analyzis(data, pca_n)
fwhm_data = fwhm_analyzis(data, prominence)

col1, col2 = st.columns(2)
peak = col1.number_input(
    "Peak",
    min_value=0,
    max_value=fwhm_data.shape[1] - 1,
    value=0,
)
features = ["Localtion", "Height", "FWHM"]
feature = col2.selectbox(
    label="Feature",
    options=np.arange(len(features)),
    format_func=lambda x: features[x],
)
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=np.rot90(np.reshape(np.array(fwhm_data[:, peak, feature]), size[0:2])),
        colorscale="Viridis",
    )
)
fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
st.plotly_chart(fig)
st.stop()

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

transformed = pca_transformed(data, pca_n)
transformed = np.reshape(transformed, (size[0], size[1], pca_n))
channel = st.number_input(
    "Channel",
    min_value=0,
    max_value=pca_n - 1,
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
            ).fit(transformed.reshape(-1, pca_n))
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

# fig.add_trace(
#     go.Scatter(
#         x=wavenumber,
#         y=data.reshape(size[0], size[1], -1)[x, y, :],
#         mode="lines",
#         name=f"({x}, {y})",
#     )
# )
# for i in range(clusters.n_clusters):
#     mask = clusters.labels_ == i
#     centroid = np.mean(data.reshape(-1, size[-1])[mask, :], axis=0)
#     peaks, _ = find_peaks(centroid, prominence=prominences)
#     fig.add_trace(
#         go.Scatter(
#             x=wavenumber,
#             y=centroid,
#             mode="lines",
#             name=f"Centroid {i}",
#         )
#     )
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=wavenumber[peaks],
#     #         y=centroid[peaks],
#     #         mode="markers",
#     #         # marker=dict(color="red"),
#     #         name=f"Peaks {i}",
#     #     )
#     # )

fig.update_layout(
    xaxis=dict(
        title="Wavenumber",
        ticksuffix="cm⁻¹",
    ),
    yaxis=dict(type="log"),
)
st.plotly_chart(fig)

# three_vs_two = st.toggle("3D vs 2D PCA plot", value=True)
# dim_z = 2
# with st.form("my_form"):
#     col1, col2, col3 = st.columns(3)
#     dim_x = col1.number_input("PCA X axis", min_value=0, max_value=n - 1, value=0)
#     dim_y = col2.number_input("PCA Y axis", min_value=0, max_value=n - 1, value=1)
#     if not three_vs_two:
#         dim_z = col3.number_input("PCA Z axis", min_value=0, max_value=n - 1, value=2)
#     submitted = st.form_submit_button("Submit")

# t = transformed.reshape(-1, n)
# fig = go.Figure()
# if not three_vs_two:
#     fig.add_trace(
#         go.Scatter3d(
#             x=transformed[..., dim_x].reshape(-1),
#             y=transformed[..., dim_y].reshape(-1),
#             z=transformed[..., dim_z].reshape(-1),
#             mode="markers",
#             marker=dict(
#                 size=2,
#                 color=clusters.labels_,
#                 colorscale="Viridis",
#                 opacity=0.8,
#             ),
#         )
#     )
#     # fig.update_layout(
#     #     height=1000,
#     # )
#     # st.plotly_chart(fig)
# else:
#     # fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=transformed[..., dim_x].reshape(-1),
#             y=transformed[..., dim_y].reshape(-1),
#             mode="markers",
#             marker=dict(
#                 size=2,
#                 color=clusters.labels_,
#                 colorscale="Viridis",
#                 opacity=0.8,
#             ),
#         )
#     )
# fig.update_layout(
#     height=1000,
# )
# st.plotly_chart(fig)
