import logging
import os
import tempfile

import numpy as np
import ramanspy as rp
from kneed import KneeLocator
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
# logger.handlers.clear()
# console_handler = logging.StreamHandler()
# formatter = logging.Formatter(
#     fmt="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


def Laden_Vorverarbeitung(Dateipfad):
    logger.info("Starte Vorverarbeitung...")
    # Lädt die Raman-Karte vom temporären Dateipfad
    raman_image = rp.load.witec(Dateipfad, laser_excitation=488.047)  # type: ignore
    pipeline = rp.preprocessing.Pipeline(
        [
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.Gaussian(),
            rp.preprocessing.baseline.ASLS(),
        ]
    )

    raman_image = pipeline.apply(raman_image)

    cropper_si = rp.preprocessing.misc.Cropper(region=(400, 700))
    karte_silizium = cropper_si.apply(raman_image)

    cropper_graphen = rp.preprocessing.misc.Cropper(region=(1200, 3500))
    karte_graphen = cropper_graphen.apply(raman_image)
    logger.info(type(karte_graphen))

    if karte_graphen.spectral_data.shape[-1] == 0:
        raise ValueError(
            f"Der Graphen-Bereich (1200-3500 cm⁻¹) enthält keine Datenpunkte."
        )

    si_referenz_intensitaet = karte_silizium.spectral_data.max()
    logger.info(
        f"Interner Si-Standard (I_Si0) gefunden: {si_referenz_intensitaet:.2f} a.u."
    )

    if si_referenz_intensitaet > 0:
        karte_silizium.spectral_data /= si_referenz_intensitaet
        karte_graphen.spectral_data /= si_referenz_intensitaet

    h, w, _ = karte_graphen.spectral_data.shape
    flat_spectra = karte_graphen.spectral_data.reshape((h * w, -1))
    valid_mask_1d = np.sum(np.abs(flat_spectra), axis=1) > 1e-6

    if np.sum(valid_mask_1d) == 0:
        raise ValueError("Keine gültigen Spektren nach der Vorverarbeitung gefunden.")
    logger.info(f"Anzahl der gültigen Spektren gefunden: {np.sum(valid_mask_1d)}")

    return valid_mask_1d, karte_silizium, karte_graphen, h, w


def PCA(valid_mask_1d, preprocessed_image, h, w, variance_threshold=0.90):
    logger.info("Starte PCA-Analyse...")
    valid_mask_2d = valid_mask_1d.reshape((h, w))

    image_for_pca = preprocessed_image[valid_mask_2d]
    data_for_pca = image_for_pca.spectral_data.reshape(
        -1, image_for_pca.spectral_data.shape[-1]
    )

    n_components_calc = 100

    pca_model = SklearnPCA(
        n_components=n_components_calc, svd_solver="randomized", random_state=42
    )

    scores_np = pca_model.fit_transform(data_for_pca)
    loadings = pca_model.components_

    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    optimal_pcs = pca_model.n_components_

    final_scores = scores_np[:, :optimal_pcs].T
    final_loadings = loadings[:optimal_pcs, :]

    return final_scores, final_loadings, optimal_pcs


def finde_optimales_k(daten, k_max=10):
    logger.info("Suche nach optimaler Clusteranzahl (K) mittels Silhouetten-Analyse...")

    num_samples = daten.shape[0]
    sample_size = min(10000, num_samples // 10)

    # Stelle sicher, dass sample_size größer als k_max ist, sonst schlägt die Analyse fehl
    if sample_size <= k_max:
        logger.info(
            f"  WARNUNG: Stichprobengröße ({sample_size}) ist zu klein. Verwende alle {num_samples} Proben."
        )
        sample_size = num_samples
        daten_sample = daten
    else:
        sample_indices = np.random.choice(num_samples, size=sample_size, replace=False)
        daten_sample = daten[sample_indices]
        logger.info(f"  Analysiere eine Stichprobe von {sample_size} Spektren...")

    silhouette_scores = []
    k_range = range(2, k_max + 1)

    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans_model.fit_predict(daten_sample)

        score = silhouette_score(daten_sample, cluster_labels)
        silhouette_scores.append(score)
        logger.info(f"  Silhouetten-Score für K={k}: {score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"-> Optimales K gefunden: {optimal_k} (höchster Silhouetten-Score)")
    return optimal_k


def filtere_graphen_spektren(karte_graphen, valid_mask_1d, h, w):
    logger.info("Filtere Spektren: Trenne Graphen von Substrat...")

    graphen_mask_1d = np.copy(valid_mask_1d)

    G_PEAK_FENSTER = (1550, 1620)
    G_BASELINE_FENSTER_LINKS = (1450, 1550)
    G_BASELINE_FENSTER_RECHTS = (1620, 1720)
    G_PEAK_PROMINENZ_SCHWELLE = 0.01

    valid_indices = np.where(valid_mask_1d)[0]
    y_coords, x_coords = np.unravel_index(valid_indices, (h, w))

    for i in range(len(valid_indices)):
        original_index = valid_indices[i]
        spectrum = karte_graphen[y_coords[i], x_coords[i]]

        g_peak_mask = (spectrum.spectral_axis >= G_PEAK_FENSTER[0]) & (
            spectrum.spectral_axis <= G_PEAK_FENSTER[1]
        )
        baseline_mask_links = (
            spectrum.spectral_axis >= G_BASELINE_FENSTER_LINKS[0]
        ) & (spectrum.spectral_axis <= G_BASELINE_FENSTER_LINKS[1])
        baseline_mask_rechts = (
            spectrum.spectral_axis >= G_BASELINE_FENSTER_RECHTS[0]
        ) & (spectrum.spectral_axis <= G_BASELINE_FENSTER_RECHTS[1])

        g_peak_intensity_max = (
            spectrum.spectral_data[g_peak_mask].max() if g_peak_mask.any() else 0
        )
        baseline_links_mean = (
            np.mean(spectrum.spectral_data[baseline_mask_links])
            if baseline_mask_links.any()
            else g_peak_intensity_max
        )
        baseline_rechts_mean = (
            np.mean(spectrum.spectral_data[baseline_mask_rechts])
            if baseline_mask_rechts.any()
            else g_peak_intensity_max
        )

        lokale_baseline = (baseline_links_mean + baseline_rechts_mean) / 2
        g_peak_prominenz = g_peak_intensity_max - lokale_baseline

        if g_peak_prominenz < G_PEAK_PROMINENZ_SCHWELLE:
            graphen_mask_1d[original_index] = False

    substrat_mask_1d = valid_mask_1d & ~graphen_mask_1d

    logger.info(
        f"  {np.sum(graphen_mask_1d)} Graphen-Spektren und {np.sum(substrat_mask_1d)} Substrat-Spektren gefunden."
    )

    return graphen_mask_1d, substrat_mask_1d


def K_Mean(valid_mask_1d, scores, h, w):
    logger.info("Starte K-Means-Clustering...")

    pca_scores_for_clustering = np.array(scores).T

    optimal_k = finde_optimales_k(pca_scores_for_clustering, k_max=10)

    logger.info(f"Führe finales K-Means-Clustering mit K={optimal_k} durch...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    cluster_labels = kmeans_model.fit_predict(pca_scores_for_clustering)
    return cluster_labels


def identifiziere_cluster(
    mean_spectra_graphen,
    mean_spectra_silizium,
    gefundene_cluster_ids,
    substrat_label=None,
):
    logger.info(
        "Starte finale, hierarchische Identifizierung (inkl. PMMA/Strain/Doping-Check)..."
    )

    # --- Hilfsfunktion für den Fit ---
    def lorentzian(x, amplitude, center, width, offset):
        return offset + (amplitude * (width**2 / ((x - center) ** 2 + width**2)))

    # --- Fenster, Schwellenwerte und Referenzpositionen ---
    SI_PEAK_FENSTER = (500, 540)
    D_PEAK_FENSTER = (1300, 1400)
    G_PEAK_FENSTER = (1550, 1620)
    TWOD_PEAK_FENSTER = (2600, 2800)
    G_BASELINE_FENSTER_LINKS = (1450, 1550)
    G_BASELINE_FENSTER_RECHTS = (1620, 1720)
    G_PEAK_PROMINENZ_SCHWELLE = 0.01
    FWHM_GRENZE_SLG = 39
    FWHM_GRENZE_BLG = 50
    I2D_IG_GRENZE_SLG = 1.3
    ID_IG_GRENZE_NIEDRIG = 0.3
    ID_IG_GRENZE_HOCH = 1.0

    G_PEAK_REF = 1580
    TWOD_PEAK_REF = 2700
    SHIFT_THRESHOLD = 10

    PMMA_CH_FENSTER = (2800, 3100)
    PMMA_CO_FENSTER = (1720, 1750)
    PMMA_SCHWELLE = 3.0

    cluster_identitaeten = {}

    for i, spectrum_graphen in enumerate(mean_spectra_graphen):
        if substrat_label is not None and gefundene_cluster_ids[i] == substrat_label:
            cluster_identitaeten[f"Cluster {i}"] = "Substrat"
            continue

        g_peak_mask = (spectrum_graphen.spectral_axis >= G_PEAK_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= G_PEAK_FENSTER[1]
        )
        pmma_ch_mask = (spectrum_graphen.spectral_axis >= PMMA_CH_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= PMMA_CH_FENSTER[1]
        )
        pmma_co_mask = (spectrum_graphen.spectral_axis >= PMMA_CO_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= PMMA_CO_FENSTER[1]
        )

        g_region_intensity = (
            np.mean(spectrum_graphen.spectral_data[g_peak_mask])
            if g_peak_mask.any()
            else 1e-9
        )
        pmma_ch_intensity = (
            np.mean(spectrum_graphen.spectral_data[pmma_ch_mask])
            if pmma_ch_mask.any()
            else 0
        )
        pmma_co_intensity = (
            np.mean(spectrum_graphen.spectral_data[pmma_co_mask])
            if pmma_co_mask.any()
            else 0
        )

        pmma_ratio = (pmma_ch_intensity + pmma_co_intensity) / g_region_intensity
        logger.info(
            f"  Cluster {gefundene_cluster_ids[i]}: PMMA/G-Verhältnis = {pmma_ratio:.2f}"
        )

        if pmma_ratio > PMMA_SCHWELLE:
            cluster_identitaeten[f"Cluster {i}"] = "PMMA-Rückstand"
            continue

        g_peak_intensity_max = (
            spectrum_graphen.spectral_data[g_peak_mask].max()
            if g_peak_mask.any()
            else 0
        )
        baseline_mask_links = (
            spectrum_graphen.spectral_axis >= G_BASELINE_FENSTER_LINKS[0]
        ) & (spectrum_graphen.spectral_axis <= G_BASELINE_FENSTER_LINKS[1])
        baseline_mask_rechts = (
            spectrum_graphen.spectral_axis >= G_BASELINE_FENSTER_RECHTS[0]
        ) & (spectrum_graphen.spectral_axis <= G_BASELINE_FENSTER_RECHTS[1])
        baseline_links_mean = (
            np.mean(spectrum_graphen.spectral_data[baseline_mask_links])
            if baseline_mask_links.any()
            else g_peak_intensity_max
        )
        baseline_rechts_mean = (
            np.mean(spectrum_graphen.spectral_data[baseline_mask_rechts])
            if baseline_mask_rechts.any()
            else g_peak_intensity_max
        )
        lokale_baseline = (baseline_links_mean + baseline_rechts_mean) / 2
        g_peak_prominenz = g_peak_intensity_max - lokale_baseline

        if g_peak_prominenz < G_PEAK_PROMINENZ_SCHWELLE:
            cluster_identitaeten[f"Cluster {i}"] = "Substrat"
            continue

        pos_g = G_PEAK_REF
        if g_peak_mask.any():
            x_g, y_g = (
                spectrum_graphen.spectral_axis[g_peak_mask],
                spectrum_graphen.spectral_data[g_peak_mask],
            )
            try:
                p0_g = [np.max(y_g) - np.min(y_g), x_g[np.argmax(y_g)], 15, np.min(y_g)]
                params_g, _ = curve_fit(lorentzian, x_g, y_g, p0=p0_g)
                pos_g = params_g[1]
            except RuntimeError:
                logger.info(
                    f"  WARNUNG: G-Peak-Fit für Cluster-Index {i} fehlgeschlagen."
                )

        fwhm_2d = 999
        pos_2d = TWOD_PEAK_REF
        x_peak, y_peak = spectrum_graphen.spectral_axis, spectrum_graphen.spectral_data
        mask_2d = (x_peak >= TWOD_PEAK_FENSTER[0]) & (x_peak <= TWOD_PEAK_FENSTER[1])
        if mask_2d.any():
            x_2d, y_2d = x_peak[mask_2d], y_peak[mask_2d]
            try:
                p0_2d = [
                    np.max(y_2d) - np.min(y_2d),
                    x_2d[np.argmax(y_2d)],
                    20,
                    np.min(y_2d),
                ]
                params_2d, _ = curve_fit(lorentzian, x_2d, y_2d, p0=p0_2d)
                pos_2d = params_2d[1]
                fwhm_2d = 2 * abs(params_2d[2])
            except RuntimeError:
                logger.info(
                    f"  WARNUNG: 2D-Peak-Fit für Cluster-Index {i} fehlgeschlagen."
                )
        logger.info(
            f"  -> Cluster {gefundene_cluster_ids[i]}: FWHM(2D)={fwhm_2d:.2f} cm-1, Pos(G)={pos_g:.2f} cm-1, Pos(2D)={pos_2d:.2f} cm-1"
        )

        twod_peak_intensity = (
            spectrum_graphen.spectral_data[mask_2d].max() if mask_2d.any() else 0
        )
        i2d_ig_ratio = (
            twod_peak_intensity / g_peak_intensity_max
            if g_peak_intensity_max > 0
            else 0
        )
        d_peak_mask = (spectrum_graphen.spectral_axis >= D_PEAK_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= D_PEAK_FENSTER[1]
        )
        d_peak_intensity = (
            spectrum_graphen.spectral_data[d_peak_mask].max()
            if d_peak_mask.any()
            else 0
        )
        id_ig_ratio = (
            d_peak_intensity / g_peak_intensity_max if g_peak_intensity_max > 0 else 0
        )

        schicht_label = ""

        is_fwhm_slg = fwhm_2d < FWHM_GRENZE_SLG
        is_ratio_slg = i2d_ig_ratio > I2D_IG_GRENZE_SLG

        if is_fwhm_slg and is_ratio_slg:
            schicht_label = "Monolage Graphen"

        elif fwhm_2d < FWHM_GRENZE_BLG:
            schicht_label = "Zweischichtiges Graphen"

        else:
            spectrum_silizium = mean_spectra_silizium[i]
            si_mask = (spectrum_silizium.spectral_axis >= SI_PEAK_FENSTER) & (
                spectrum_silizium.spectral_axis <= SI_PEAK_FENSTER[1]
            )
            si_verhaeltnis = (
                min(spectrum_silizium.spectral_data[si_mask].max(), 1.0)
                if si_mask.any()
                else 0
            )

            if si_verhaeltnis > 0.75:
                schicht_label = "Zweischichtiges Graphen"
            else:
                schicht_label = "Viele Lagen Graphen"

        qualitaet_label = ""
        if id_ig_ratio >= ID_IG_GRENZE_HOCH:
            qualitaet_label = " (stark defekt)"
        elif id_ig_ratio >= ID_IG_GRENZE_NIEDRIG:
            qualitaet_label = " (defekt)"
        else:
            qualitaet_label = " (hohe Qualität)"

        strain_doping_label = ""
        delta_pos_g = pos_g - G_PEAK_REF
        delta_pos_2d = pos_2d - TWOD_PEAK_REF

        if abs(delta_pos_g) > 1:
            shift_ratio = delta_pos_2d / delta_pos_g
        else:
            shift_ratio = 0

        if delta_pos_g > SHIFT_THRESHOLD and delta_pos_2d > SHIFT_THRESHOLD:
            if shift_ratio > 2.0:
                strain_doping_label = ", kompressiv verspannt"
            else:
                strain_doping_label = ", p-dotiert"
        elif delta_pos_g < -SHIFT_THRESHOLD and delta_pos_2d < -SHIFT_THRESHOLD:
            strain_doping_label = ", tensil verspannt"
        elif delta_pos_g > SHIFT_THRESHOLD and delta_pos_2d < -SHIFT_THRESHOLD:
            strain_doping_label = ", n-dotiert"

        cluster_identitaeten[f"Cluster {i}"] = (
            schicht_label + qualitaet_label + strain_doping_label
        )

    logger.info("Identifizierung abgeschlossen:", cluster_identitaeten)
    return cluster_identitaeten


def DBSCAN_Clustering(eps, min_samples, valid_mask_1d, scaled_scores, h, w):
    logger.info(
        f"Führe DBSCAN-Clustering mit eps={eps:.4f} und min_samples={min_samples} durch..."
    )

    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan_model.fit_predict(scaled_scores)
    return cluster_labels


def finde_besten_eps(scores, min_samples):
    logger.info("Bestimme optimalen eps-Wert automatisch...")

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(scores)
    distances, indices = neighbors_fit.kneighbors(scores)

    distances = np.sort(distances, axis=0)
    distances = distances[:, min_samples - 1]

    kneedle = KneeLocator(
        range(len(distances)), distances, S=1.0, curve="convex", direction="increasing"
    )

    optimal_eps = kneedle.knee_y
    logger.info(f"-> Optimaler eps-Wert gefunden: {optimal_eps:.4f}")
    return optimal_eps


def run_pca_k_mean_analysis(file_bytes):
    """
    Führt die PCA-K-Mean-Analyse aus und gibt Plot-Daten zurück.
    Akzeptiert file_bytes von st.file_uploader.
    """
    try:
        # Erstelle eine temporäre Datei, um die Bytes zu speichern
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        logger.info(f"Temporäre Datei erstellt: {temp_file_path}")

        # Schritt 1: Lade und verarbeite die Raman-Karte.
        valid_mask_1d, karte_silizium, karte_graphen, h, w = Laden_Vorverarbeitung(
            temp_file_path
        )

        # Schritt 2: Trenne Graphen-Spektren von Substrat-Spektren
        graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(
            karte_graphen, valid_mask_1d, h, w
        )

        # Schritt 3: PCA nur auf den Graphen-Daten durchführen
        scores, loadings, optimal_pcs_gefunden = PCA(
            graphen_mask_1d, karte_graphen, h, w
        )

        # Schritt 4: K-Means-Clustering nur auf den Graphen-Daten durchführen
        graphen_cluster_labels = K_Mean(graphen_mask_1d, scores, h, w)

        # Schritt 5: Erstelle die finale Cluster-Karte
        logger.info("Kombiniere Ergebnisse zu finaler Cluster-Karte...")
        final_cluster_map_1d = np.full(h * w, np.nan)
        SUBSTRAT_LABEL = 0
        final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
        final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels + 1
        final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

        # Schritt 6: Berechne mittlere Spektren
        logger.info("Berechne mittlere Spektren für jeden finalen Cluster...")
        unique_final_labels = sorted(
            [label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)]
        )

        mean_spectra_graphen = []
        mean_spectra_silizium = []
        gefundene_cluster_ids = []

        for label in unique_final_labels:
            cluster_mask_1d = final_cluster_map_1d == label
            cluster_mask_2d = cluster_mask_1d.reshape((h, w))

            if np.any(cluster_mask_2d):
                mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
                mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean)
                gefundene_cluster_ids.append(int(label))

        # Schritt 7: Cluster automatisch identifizieren
        neue_labels_map = identifiziere_cluster(
            mean_spectra_graphen,
            mean_spectra_silizium,
            gefundene_cluster_ids,
            substrat_label=SUBSTRAT_LABEL,
        )
        finale_plot_labels = [
            f"Cluster {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}"
            for i, original_id in enumerate(gefundene_cluster_ids)
        ]

        # Schritt 8: Plot-Parameter bestimmen
        global_max_intensity = 0
        for spectrum in mean_spectra_graphen:
            current_max = np.max(spectrum.spectral_data)
            if current_max > global_max_intensity:
                global_max_intensity = current_max

        plot_ylim = global_max_intensity * 1.1
        logger.info(f"  Setze einheitliches Y-Achsen-Limit auf: {plot_ylim:.4f} a.u.")

        # Schritt 9: Daten für Streamlit zurückgeben
        return {
            "success": True,
            "cluster_map": final_cluster_map_2d,
            "unique_labels": unique_final_labels,
            "mean_spectra": mean_spectra_graphen,
            "plot_labels": finale_plot_labels,
            "y_limit": plot_ylim,
            "map_title": "Finale Cluster-Karte (K-Mean auf Graphen)",
        }

    except Exception as e:
        logger.info(f"FEHLER während der K-Means-Analyse: {e}")
        import traceback

        logger.info(traceback.format_exc())
        return {"success": False, "error": str(e)}

    finally:
        # Bereinige die temporäre Datei
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporäre Datei gelöscht: {temp_file_path}")


def run_pca_dbscan_analysis(file_bytes):
    try:
        # Erstelle eine temporäre Datei, um die Bytes zu speichern
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        logger.info(f"Temporäre Datei erstellt: {temp_file_path}")

        # Schritt 1: Lade und verarbeite die Raman-Karte.
        valid_mask_1d, karte_silizium, karte_graphen, h, w = Laden_Vorverarbeitung(
            temp_file_path
        )

        # Schritt 2: Trenne Graphen-Spektren von Substrat-Spektren
        graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(
            karte_graphen, valid_mask_1d, h, w
        )

        # Schritt 3: PCA nur auf den Graphen-Daten
        scores, loadings, optimal_pcs_gefunden = PCA(
            graphen_mask_1d, karte_graphen, h, w
        )

        # Schritt 4: Skalierung
        logger.info("Skaliere PCA-Scores der Graphen-Region für die Cluster-Analyse...")
        scores_for_scaling = np.array(scores).T
        scaler = StandardScaler()
        scaled_scores = scaler.fit_transform(scores_for_scaling)

        # Schritt 5: DBSCAN-Parameter
        min_samples_auto = 2 * optimal_pcs_gefunden
        logger.info(f"Bestimme min_samples automatisch: {min_samples_auto}")
        eps_auto = finde_besten_eps(scaled_scores, min_samples_auto)

        # Schritt 6: DBSCAN-Clustering
        graphen_cluster_labels = DBSCAN_Clustering(
            eps_auto, min_samples_auto, graphen_mask_1d, scaled_scores, h, w
        )

        # Schritt 7: Finale Cluster-Karte
        logger.info("Kombiniere Ergebnisse zu finaler Cluster-Karte...")
        final_cluster_map_1d = np.full(h * w, np.nan)
        SUBSTRAT_LABEL = -2  # Fester Wert aus deinem Skript
        final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
        final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels
        final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

        # Schritt 8: Mittlere Spektren
        logger.info("Berechne mittlere Spektren für jeden finalen Cluster...")
        unique_final_labels = sorted(
            [label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)]
        )

        mean_spectra_graphen = []
        mean_spectra_silizium = []
        gefundene_cluster_ids = []

        for label in unique_final_labels:
            cluster_mask_1d = final_cluster_map_1d == label
            cluster_mask_2d = cluster_mask_1d.reshape((h, w))

            if np.any(cluster_mask_2d):
                mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
                mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean)
                gefundene_cluster_ids.append(int(label))

        # Schritt 9: Cluster identifizieren
        neue_labels_map = identifiziere_cluster(
            mean_spectra_graphen,
            mean_spectra_silizium,
            gefundene_cluster_ids,
            substrat_label=SUBSTRAT_LABEL,
        )
        finale_plot_labels = [
            f"Cluster {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}"
            for i, original_id in enumerate(gefundene_cluster_ids)
        ]

        # Schritt 10: Plot-Parameter
        global_max_intensity = 0
        for spectrum in mean_spectra_graphen:
            current_max = np.max(spectrum.spectral_data)
            if current_max > global_max_intensity:
                global_max_intensity = current_max

        plot_ylim = global_max_intensity * 1.1
        logger.info(f"  Setze einheitliches Y-Achsen-Limit auf: {plot_ylim:.4f} a.u.")

        # Schritt 11: Daten für Streamlit zurückgeben
        return {
            "success": True,
            "cluster_map": final_cluster_map_2d,
            "unique_labels": unique_final_labels,
            "mean_spectra": mean_spectra_graphen,
            "plot_labels": finale_plot_labels,
            "y_limit": plot_ylim,
            "map_title": "Finale Cluster-Karte (DBSCAN auf Graphen)",
        }

    except Exception as e:
        logger.info(f"FEHLER während der DBSCAN-Analyse: {e}")
        import traceback

        logger.info(traceback.format_exc())
        return {"success": False, "error": str(e)}

    finally:
        # Bereinige die temporäre Datei
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporäre Datei gelöscht: {temp_file_path}")
