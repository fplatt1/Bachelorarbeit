import logging

import numpy as np
import streamlit as st
from plotly import graph_objs as go

try:
    from funktionen_streamlit import run_pca_dbscan_analysis, run_feature_engineering_k_mean_analysis, run_feature_engineering_som_analysis
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
        case "SOM":
            results = run_feature_engineering_som_analysis(file_bytes)
        case _:
            raise Exception("Unbekannte Analyse-Methode ausgewählt.")
    return results


def plot_cluster_map(cluster_map, unique_labels):
    # 1. Dimensionen holen
    num_rows, num_cols = cluster_map.shape

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(cluster_map),
            colorscale="Viridis",
            # Einziger Unterschied zur Feature-Map: 
            # Wir formatieren die Legende für ganze Zahlen (Cluster 0, 1, 2...)
            colorbar=dict(
                title="Cluster",
                tickmode='array',
                tickvals=unique_labels,
                ticktext=[str(l) for l in unique_labels]  # noqa: E741
            )
        )
    )
    
    # 2. Layout exakt von deiner funktionierenden Feature-Map übernommen
    fig.update_layout(
        width=800, 
        height=800, 
        
        
        # X-Achse begrenzen
        xaxis=dict(
            range=[0, num_cols - 1],
            constrain="domain" # Verhindert Whitespace
        ),
        
        # Y-Achse an X koppeln und begrenzen
        yaxis=dict(
            scaleanchor="x", 
            scaleratio=1,
            range=[0, num_rows - 1],
            constrain="domain"
        )
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
    options=(None, "K-Means", "DBSCAN", "SOM"),
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

# --- HAUPT-ERGEBNIS LOGIK ---
if results and results["success"]:
    st.success("Analyse erfolgreich abgeschlossen!")

    # 1. Feature-Maps
    if results.get("feature_names") and results.get("feature_maps") is not None:
        st.subheader("Feature-Maps (vor PCA)")
        feature_names = results["feature_names"]
        feature_maps = results["feature_maps"]
        
        # Auswahl der Feature
        chosen = st.selectbox("Wähle ein Feature zur Anzeige:", options=feature_names)
        idx = feature_names.index(chosen)
        feature_map = feature_maps[:, :, idx]
        
        # -----------------------------------------
        def plot_feature_map(feature_map, title, cmap="Viridis"):
            # 1. Dimensionen holen
            num_rows, num_cols = feature_map.shape
            
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(z=np.flipud(feature_map), colorscale=cmap, colorbar=dict(title=chosen))
            )
            fig.update_layout(
                width=800, 
                height=800, 
                title=title,
                # X-Achse begrenzen
                xaxis=dict(
                    range=[0, num_cols - 1],
                    constrain="domain" # Verhindert Whitespace
                ),
                # Y-Achse an X koppeln und begrenzen
                yaxis=dict(
                    scaleanchor="x", 
                    scaleratio=1,
                    range=[0, num_rows - 1],
                    constrain="domain"
                )
            )
            return fig
        # -----------------------------------------

        st.plotly_chart(plot_feature_map(feature_map, f"Feature: {chosen}"), use_container_width=False)

    # 2. Cluster-Karte (Wird immer angezeigt, wenn success=True)
    st.subheader("Cluster-Karte")
    if results.get("cluster_map") is not None:
        fig_map = plot_cluster_map(
            results["cluster_map"], results["unique_labels"]
        )
        st.plotly_chart(fig_map)
    else:
        st.warning("Keine Cluster-Karte verfügbar.")

    # 3. Mittlere Spektren (Wird immer angezeigt, wenn success=True)
    st.subheader("Mittlere Spektren")
    if results.get("mean_spectra") is not None:
        fig_spectra = plot_mean_spectra(
            results["mean_spectra"], results["plot_labels"], results["y_limit"]
        )
        st.plotly_chart(fig_spectra)
    else:
        st.warning("Keine Spektren verfügbar.")

# Fehlerbehandlung: Greift nur, wenn results["success"] False ist oder results leer ist
elif results and not results.get("success", False):
    st.error(f"Analyse fehlgeschlagen. Fehler: {results.get('error', 'Unbekannter Fehler')}")