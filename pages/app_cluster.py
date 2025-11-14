import logging

import numpy as np
import streamlit as st
from plotly import graph_objs as go

try:
    from funktionen_streamlit import run_pca_dbscan_analysis, run_feature_engineering_k_mean_analysis
except ImportError:
    st.error(
        "Fehler: Die Datei 'funktionen_streamlit.py' konnte nicht gefunden werden. Stelle sicher, dass sie im selben Verzeichnis wie 'app_cluster.py' liegt."
    )
    st.stop()

LOG_FILE = "log.txt"
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
)


@st.cache_data
def analyze(file_bytes, analysis_method: str):
    match analysis_method:
        case "K-Means":
            results = run_feature_engineering_k_mean_analysis(file_bytes)
        case "DBSCAN":
            results = run_pca_dbscan_analysis(file_bytes)
        case _:
            raise Exception("Unbekannte Analyse-Methode ausgewählt.")
    return results


def plot_cluster_map(cluster_map, unique_labels, title):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            # z=np.transpose(cluster_map),
            z=np.flipud(cluster_map),
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return fig


def plot_mean_spectra(mean_spectra, plot_labels, y_limit):
    fig = go.Figure()

    for spectrum, label in zip(mean_spectra, plot_labels, strict=True):
        fig.add_trace(
            go.Scatter(
                x=spectrum.spectral_axis,
                y=spectrum.spectral_data,
                name=label,
            )
        )
    fig.update_layout(
        xaxis=dict(title="Raman Shift (cm⁻¹)"),
        yaxis=dict(title="Intensity (a.u.)", range=[0, y_limit]),
    )
    return fig


# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Raman-Karten Analyse (PCA + Clustering)")

# Sidebar für Optionen
st.sidebar.header("Analyse-Einstellungen")
analysis_method = st.sidebar.selectbox(
    label="Wähle die Clustering-Methode:",
    options=(None, "K-Means", "DBSCAN"),
    index=0,
)

if analysis_method is None:
    st.info("Bitte wähle eine Analyse-Methode in der Seitenleiste aus.")
    st.stop()

# Hauptbereich für Datei-Upload
st.header("1. Raman-Karte hochladen")
uploaded_file = st.file_uploader("Wähle eine .mat-Datei (Witec)", type=["mat"])

if uploaded_file is None:
    st.error("Datei konnte nicht geladen werden!")
    st.stop()
else:
    st.success(f"Datei '{uploaded_file.name}' geladen.")

# Lese die Bytes der Datei
file_bytes = uploaded_file.getvalue()
results = analyze(file_bytes, analysis_method)

# --- Ergebnisse anzeigen ---
st.header("2. Ergebnisse")

# Zeige das Analyse-Log in einem ausklappbaren Bereich
with st.expander("Analyse-Log anzeigen (Terminal-Ausgabe)"):
    with open(LOG_FILE) as f:
        logs = f.read()
    st.text_area("Log", logs, height=300)

if results and results["success"]:
    st.success("Analyse erfolgreich abgeschlossen!")

    # --- Plot 1: Cluster-Karte ---
    st.subheader("Cluster-Karte")
    fig_map = plot_cluster_map(
        results["cluster_map"], results["unique_labels"], results["map_title"]
    )
    st.plotly_chart(fig_map)
    # st.pyplot(fig_map)

    # --- Plot 2: Mittlere Spektren ---
    st.subheader("Mittlere Spektren")
    fig_spectra = plot_mean_spectra(
        results["mean_spectra"], results["plot_labels"], results["y_limit"]
    )
    st.plotly_chart(fig_spectra)
    # st.pyplot(fig_spectra)

elif results:
    st.error(f"Analyse fehlgeschlagen. Fehler: {results['error']}")
