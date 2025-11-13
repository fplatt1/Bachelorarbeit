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
    data: npt.NDArray, prominence: float, number_peaks: int | None = None
):
    average = np.mean(data, axis=0)
    peaks, _ = find_peaks(average, prominence=prominence)
    print(len(peaks))

    if number_peaks is None:
        number_peaks = len(peaks)

    features = np.zeros((data.shape[0], number_peaks or len(peaks), 3))
    for i in range(data.shape[0]):
        datum = data[i, :]
        peaks, _ = find_peaks(datum, prominence=prominence)
        if i < 20:
            print(len(peaks))
        # prominences = peak_prominences(average, peaks)[0]
        # peaks = peaks[np.argsort(prominences)[::-1]]
        widths = peak_widths(average, peaks)[0]
        number_peaks_local = min(number_peaks, len(peaks))
        features[i, :number_peaks_local, 0] = peaks[:number_peaks_local]
        features[i, :number_peaks_local, 1] = datum[peaks][:number_peaks_local]
        features[i, :number_peaks_local, 2] = widths[:number_peaks_local]
    return features
