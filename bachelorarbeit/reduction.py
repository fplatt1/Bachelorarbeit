import numpy as np
import numpy.typing as npt
import streamlit as st
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn.decomposition import PCA


@st.cache_data
def pca_analyzis(data, n=40):
    pca_instance = PCA(n_components=n)
    pca_data = pca_instance.fit(data)
    return pca_data


@st.cache_data
def pca_transformed(data, n=40):
    pca_data = pca_analyzis(data, n)
    return pca_data.transform(data)


@st.cache_data
def fwhm_analyzis(
    data: npt.NDArray,
    prominence: float,
    ignore: list[int] | None = None,
):
    average = np.mean(data, axis=0)
    peaks, properties = find_peaks(average, prominence=prominence, width=(None, None))
    f = 0.1
    if ignore is not None:
        mask = np.zeros_like(peaks, dtype=bool)
        w = properties["widths"]
        for ig in ignore:
            mask |= (properties["left_ips"] - w * f <= ig) & (
                ig <= properties["right_ips"] + w * f
            )
        peaks = peaks[~mask]
        for key in properties:
            properties[key] = properties[key][~mask]

    # features = np.zeros((data.shape[0], number_peaks or len(peaks), 3))
    features = np.zeros((data.shape[0], len(peaks), 2))
    for i in range(data.shape[0]):
        for j in range(peaks.size):
            w = properties["widths"][j]
            idx_l = np.clip(
                int(np.floor(properties["left_ips"][j] - w * f)),
                0,
                data.shape[1] - 1,
            )
            idx_r = np.clip(
                int(np.ceil(properties["right_ips"][j] + w * f)),
                0,
                data.shape[1] - 1,
            )
            peak_region = data[i, idx_l : idx_r + 1]
            peak_idx = np.argmax(peak_region) + idx_l
            features[i, j, 0] = peak_idx
            features[i, j, 1] = data[i, peak_idx]
            # features[i, :, 1] = widths

        # datum = data[i, :]
        # # peaks, _ = find_peaks(datum, prominence=prominence)
        # # prominences = peak_prominences(average, peaks)[0]
        # # peaks = peaks[np.argsort(prominences)[::-1]]
        # widths = peak_widths(datum, peaks)[0]
        # features[i, :, 0] = datum[peaks]
        # features[i, :, 1] = widths
        # # number_peaks_local = min(number_peaks, len(peaks))
        # # features[i, :number_peaks_local, 0] = peaks[:number_peaks_local]
        # # features[i, :number_peaks_local, 1] = datum[peaks][:number_peaks_local]
        # # features[i, :number_peaks_local, 2] = widths[:number_peaks_local]
    return features
