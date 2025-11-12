import ramanspy as rp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from kneed import KneeLocator
import streamlit as st
import tempfile
import os
import sys
import io



def Laden_Vorverarbeitung(Dateipfad):
    print("Starte Vorverarbeitung...")
    raman_image = rp.load.witec(Dateipfad, laser_excitation=488.047)

    pipeline = rp.preprocessing.Pipeline([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.denoise.Gaussian(),
        rp.preprocessing.baseline.ASLS(),
    ])

    raman_image = pipeline.apply(raman_image)
    
    cropper_si = rp.preprocessing.misc.Cropper(region=(400, 700))
    karte_silizium = cropper_si.apply(raman_image)

    cropper_graphen = rp.preprocessing.misc.Cropper(region=(1200, 3500))
    karte_graphen = cropper_graphen.apply(raman_image)

    if karte_graphen.spectral_data.shape[-1] == 0:
        raise ValueError(f"Der Graphen-Bereich (1200-3500 cm⁻¹) enthält keine Datenpunkte.")
    
    
    si_referenz_intensitaet = karte_silizium.spectral_data.max()
    print(f"Interner Si-Standard (I_Si0) gefunden: {si_referenz_intensitaet:.2f} a.u.")

    if si_referenz_intensitaet > 0:
        karte_silizium.spectral_data /= si_referenz_intensitaet
        karte_graphen.spectral_data /= si_referenz_intensitaet
    
    h, w, _ = karte_graphen.spectral_data.shape
    flat_spectra = karte_graphen.spectral_data.reshape((h * w, -1))
    valid_mask_1d = np.sum(np.abs(flat_spectra), axis=1) > 1e-6 
    
    if np.sum(valid_mask_1d) == 0:
        raise ValueError("Keine gültigen Spektren nach der Vorverarbeitung gefunden.")
    print(f"Anzahl der gültigen Spektren gefunden: {np.sum(valid_mask_1d)}")

    return valid_mask_1d, karte_silizium, karte_graphen, h, w

def PCA(valid_mask_1d, preprocessed_image, h, w, variance_threshold=0.90):
    print("Starte PCA-Analyse...")
    valid_mask_2d = valid_mask_1d.reshape((h, w))
    
    image_for_pca = preprocessed_image[valid_mask_2d]
    data_for_pca = image_for_pca.spectral_data.reshape(-1, image_for_pca.spectral_data.shape[-1])

    pca_model = SklearnPCA(n_components=15) 
    
    scores_np = pca_model.fit_transform(data_for_pca)
    loadings = pca_model.components_
    plot_components = 4
    # --- NEU: PLOTTING DER ERSTEN 4 (oder 'plot_components') LOADINGS ---
    if plot_components > 0:
        
        # Stelle sicher, dass wir nicht mehr Komponenten plotten als vorhanden sind
        num_to_plot = min(45, loadings.shape[0])
        
        # Erstelle eine Abbildung mit 'num_to_plot' Unter-Plots (Subplots)
        fig, axes = plt.subplots(num_to_plot, 1, 
                                 figsize=(10, 2.5 * num_to_plot), 
                                 sharex=True)
        
        if num_to_plot == 1:
            axes = [axes] # Mache 'axes' iterierbar, falls nur 1 Plot
            
        fig.suptitle('PCA-Loadings (Eigenvektoren)', fontsize=16, y=1.02)
        
        for i in range(num_to_plot):
            axes[i].plot(preprocessed_image.spectral_axis, loadings[i, :])
            axes[i].set_title(f'PC {i+1} (Loading)')
            axes[i].set_ylabel('Gewichtung (a.u.)')
            axes[i].grid(linestyle=':', alpha=0.7)
            
        axes[-1].set_xlabel('Raman Shift ($cm^{-1}$)') # LaTeX für cm^-1
        plt.tight_layout()
        
        # HINWEIS: In Ihrer Streamlit-App (erwähnt in 3.2.1 )
        # würden Sie 'plt.show()' wahrscheinlich durch 'st.pyplot(fig)' ersetzen.
        plt.show()
    # --- ENDE DES NEUEN PLOTTING-TEILS ---
    
    print("PCA-Analyse abgeschlossen.")

    # --- ANALYSE DES ERKLÄRTEN VARIANZANTEILS ---
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # --- AUTOMATISCHE AUSWAHL DER PCs ---
    # Zuerst prüfen, ob der Schwellenwert überhaupt erreicht wird.
    if np.any(cumulative_variance >= variance_threshold):
        # np.argmax gibt den Index des ersten 'True'-Wertes zurück.
        # +1, da Indizes bei 0 beginnen und wir die Anzahl der Komponenten wollen.
        optimal_pcs = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"\nAUTOMATISCHE AUSWAHL: {optimal_pcs} Hauptkomponenten werden ausgewählt, um mindestens {variance_threshold:.0%} der Varianz zu erklären.")
    else:
        # Dieser Fall tritt ein, wenn der Schwellenwert nie erreicht wird.
        # Wir nehmen dann alle berechneten Komponenten.
        optimal_pcs = len(cumulative_variance) # z.B. 15
        print(f"\nWARNUNG: Der Varianz-Schwellenwert von {variance_threshold:.0%} wurde mit {len(cumulative_variance)} Komponenten nicht erreicht. Verwende alle {optimal_pcs} Komponenten.")

    # --- VISUALISIERUNG ---
    print("\n--- Analyse des erklärten Varianzanteils ---")
    for i, variance in enumerate(explained_variance):
        print(f"Hauptkomponente {i+1}: erklärt {variance:.2%} der Varianz")
    
    # plt.figure(figsize=(10, 6))
    # x_ticks = range(1, len(explained_variance) + 1)
    # plt.bar(x_ticks, explained_variance, alpha=0.6, align='center', label='Individueller erklärter Varianzanteil')
    # plt.step(x_ticks, cumulative_variance, where='mid', label='Kumulativer erklärter Varianzanteil', color='red')
    # plt.ylabel('Erklärter Varianzanteil')
    # plt.xlabel('Hauptkomponenten')
    # plt.title('Erklärter Varianzanteil pro Hauptkomponente (Scree Plot)')
    # plt.xticks(x_ticks)
    # plt.legend(loc='best')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.axhline(y=variance_threshold, color='g', linestyle='--', label=f'{variance_threshold:.0%} Schwellenwert')
    # plt.axvline(x=optimal_pcs, color='purple', linestyle=':', label=f'Ausgewählte PCs = {optimal_pcs}')
    # plt.ylim(0, 1.1)
    # plt.show()
        
    # Reduziere die Scores und Loadings auf die automatisch bestimmte, optimale Anzahl
    final_scores = scores_np[:, :optimal_pcs].T
    final_loadings = loadings[:optimal_pcs, :]
        
    return final_scores, final_loadings, optimal_pcs

def finde_optimales_k(daten, k_max=10):
    print("Suche nach optimaler Clusteranzahl (K) mittels Silhouetten-Analyse...")
    
    # --- KORREKTUR: Zugriff auf die Anzahl der Proben ---
    # daten.shape ist ein Tupel (z.B. (62500, 3)). Wir brauchen die Anzahl der Proben, also daten.shape.
    num_samples = daten.shape[0]
    
    # Berechne die Stichprobengröße: 10% der Daten, aber maximal 10.000
    sample_size = min(10000, num_samples // 10)
    
    # Wähle zufällige Indizes aus der Gesamtmenge der Proben.
    sample_indices = np.random.choice(num_samples, size=sample_size, replace=False)
    daten_sample = daten[sample_indices]
    print(f"  Analysiere eine Stichprobe von {sample_size} Spektren...")

    silhouette_scores =  [] # Initialisierung als leere Liste
    k_range = range(2, k_max + 1) # Silhouetten-Score ist für k=1 nicht definiert

    for k in k_range:
        # Führe K-Means für das aktuelle k durch
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans_model.fit_predict(daten_sample)
        
        # Berechne den Silhouetten-Score und speichere ihn
        score = silhouette_score(daten_sample, cluster_labels)
        silhouette_scores.append(score)
        print(f"  Silhouetten-Score für K={k}: {score:.4f}")


    # Finde das k mit dem höchsten Score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"-> Optimales K gefunden: {optimal_k} (höchster Silhouetten-Score)")

    # # Optional: Plot zur Visualisierung der Scores für die Bachelorarbeit
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_range, silhouette_scores, 'bo-')
    # plt.xlabel('Anzahl der Cluster (K)')
    # plt.ylabel('Durchschnittlicher Silhouetten-Score')
    # plt.title('Silhouetten-Analyse zur Bestimmung des optimalen K')
    # plt.grid(True)
    # plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimales K = {optimal_k}')
    # plt.legend()
    # plt.show()

    return optimal_k

def filtere_graphen_spektren(karte_graphen, valid_mask_1d, h, w):
    print("Filtere Spektren: Trenne Graphen von Substrat...")
    
    # Initialisiere die neue Maske
    graphen_mask_1d = np.copy(valid_mask_1d)
    
    # Fenster und Schwellenwerte
    G_PEAK_FENSTER = (1550, 1620)
    G_BASELINE_FENSTER_LINKS = (1450, 1550)
    G_BASELINE_FENSTER_RECHTS = (1620, 1720)
    G_PEAK_PROMINENZ_SCHWELLE = 0.01

    # Erstelle ein Array der validen Indizes
    valid_indices = np.where(valid_mask_1d)[0] # [0] um das Tupel zu entpacken
    
    # Wandle die validen 1D-Indizes in 2D-Koordinaten um
    y_coords, x_coords = np.unravel_index(valid_indices, (h, w))

    # Iteriere über die Koordinaten, um jedes valide Spektrum zu prüfen
    for i in range(len(valid_indices)):
        original_index = valid_indices[i]
        spectrum = karte_graphen[y_coords[i], x_coords[i]]
        
        # KORREKTUR: Greife auf die Tupel-Elemente mit [0] und [1] zu
        g_peak_mask = (spectrum.spectral_axis >= G_PEAK_FENSTER[0]) & (spectrum.spectral_axis <= G_PEAK_FENSTER[1])
        baseline_mask_links = (spectrum.spectral_axis >= G_BASELINE_FENSTER_LINKS[0]) & (spectrum.spectral_axis <= G_BASELINE_FENSTER_LINKS[1])
        baseline_mask_rechts = (spectrum.spectral_axis >= G_BASELINE_FENSTER_RECHTS[0]) & (spectrum.spectral_axis <= G_BASELINE_FENSTER_RECHTS[1])
        
        g_peak_intensity_max = spectrum.spectral_data[g_peak_mask].max() if g_peak_mask.any() else 0
        baseline_links_mean = np.mean(spectrum.spectral_data[baseline_mask_links]) if baseline_mask_links.any() else g_peak_intensity_max
        baseline_rechts_mean = np.mean(spectrum.spectral_data[baseline_mask_rechts]) if baseline_mask_rechts.any() else g_peak_intensity_max
        
        lokale_baseline = (baseline_links_mean + baseline_rechts_mean) / 2
        g_peak_prominenz = g_peak_intensity_max - lokale_baseline

        if g_peak_prominenz < G_PEAK_PROMINENZ_SCHWELLE:
            graphen_mask_1d[original_index] = False

    substrat_mask_1d = valid_mask_1d & ~graphen_mask_1d
    
    print(f"  {np.sum(graphen_mask_1d)} Graphen-Spektren und {np.sum(substrat_mask_1d)} Substrat-Spektren gefunden.")
    
    return graphen_mask_1d, substrat_mask_1d

def K_Mean(valid_mask_1d, scores, h, w):
    print("Starte K-Means-Clustering...")

    # Das.T ist notwendig, um die Daten-Form von (Komponenten, Spektren)
    # zu (Spektren, Komponenten) zu korrigieren.
    pca_scores_for_clustering = np.array(scores).T
    
    # Schritt 1: Bestimme automatisch die optimale Anzahl an Clustern
    # Der Parameter 'cluster' wird nicht mehr übergeben, sondern hier ermittelt.
    optimal_k = finde_optimales_k(pca_scores_for_clustering, k_max=10)

    # Schritt 2: Führe das finale K-Means-Clustering mit dem optimalen k durch
    print(f"Führe finales K-Means-Clustering mit K={optimal_k} durch...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto') 
    cluster_labels = kmeans_model.fit_predict(pca_scores_for_clustering)

    # Schritt 3: Visualisiere das Cluster-Ergebnis
    cluster_map_1d = np.full(h * w, np.nan)
    cluster_map_1d[valid_mask_1d] = cluster_labels
    cluster_map_2d = cluster_map_1d.reshape((h, w))

    # plt.figure(figsize=(10, 7))
    # plt.imshow(cluster_map_2d, cmap='viridis') 
    # plt.colorbar(label='Cluster-ID')
    # plt.title(f'K-Means Clustering (Automatisch bestimmtes K = {optimal_k})')
    
    return cluster_labels

def identifiziere_cluster(mean_spectra_graphen, mean_spectra_silizium, gefundene_cluster_ids, substrat_label=None):
    """
    Identifiziert Cluster basierend auf ihren mittleren Spektren, inklusive der Erkennung
    von PMMA-Rückständen und der Analyse von Verspannung/Dotierung.
    """
    print("Starte finale, hierarchische Identifizierung (inkl. PMMA/Strain/Doping-Check)...")

    # --- Hilfsfunktion für den Fit ---
    def lorentzian(x, amplitude, center, width, offset):
        return offset + (amplitude * (width**2 / ((x - center)**2 + width**2)))

    # --- Fenster, Schwellenwerte und Referenzpositionen ---
    SI_PEAK_FENSTER = (500, 540)
    D_PEAK_FENSTER = (1300, 1400)
    G_PEAK_FENSTER = (1550, 1620)
    TWOD_PEAK_FENSTER = (2600, 2800)
    G_BASELINE_FENSTER_LINKS = (1450, 1550)
    G_BASELINE_FENSTER_RECHTS = (1620, 1720)
    G_PEAK_PROMINENZ_SCHWELLE = 0.01
    FWHM_GRENZE_SLG = 38.0
    FWHM_GRENZE_BLG = 50
    I2D_IG_GRENZE_SLG = 1.5
    I2D_IG_GRENZE_BLG = 0.8
    ID_IG_GRENZE_NIEDRIG = 0.3
    ID_IG_GRENZE_HOCH = 1.0
    
    # Referenzwerte für Strain/Doping
    G_PEAK_REF = 1582
    TWOD_PEAK_REF = 2680
    SHIFT_THRESHOLD = 4

    # Fenster und Schwellenwert für PMMA-Erkennung
    PMMA_CH_FENSTER = (2800, 3100)
    PMMA_CO_FENSTER = (1720, 1750)
    PMMA_SCHWELLE = 3.0  # Heuristik: Wenn PMMA-Signale 3x stärker sind als G-Region, ist es PMMA

    cluster_identitaeten = {}

    for i, spectrum_graphen in enumerate(mean_spectra_graphen):
        # 1. Prüfe auf vorab identifiziertes Substrat
        if substrat_label is not None and gefundene_cluster_ids[i] == substrat_label:
            cluster_identitaeten[f"Cluster {i}"] = "Substrat"
            continue

        # 2. Prüfe auf dominante PMMA-Signale
        g_peak_mask = (spectrum_graphen.spectral_axis >= G_PEAK_FENSTER[0]) & (spectrum_graphen.spectral_axis <= G_PEAK_FENSTER[1])
        pmma_ch_mask = (spectrum_graphen.spectral_axis >= PMMA_CH_FENSTER[0]) & (spectrum_graphen.spectral_axis <= PMMA_CH_FENSTER[1])
        pmma_co_mask = (spectrum_graphen.spectral_axis >= PMMA_CO_FENSTER[0]) & (spectrum_graphen.spectral_axis <= PMMA_CO_FENSTER[1])

        # Berechne mittlere Intensitäten in den relevanten Bereichen
        g_region_intensity = np.mean(spectrum_graphen.spectral_data[g_peak_mask]) if g_peak_mask.any() else 1e-9
        pmma_ch_intensity = np.mean(spectrum_graphen.spectral_data[pmma_ch_mask]) if pmma_ch_mask.any() else 0
        pmma_co_intensity = np.mean(spectrum_graphen.spectral_data[pmma_co_mask]) if pmma_co_mask.any() else 0
        
        # Berechne das Verhältnis der PMMA-Signale zur G-Region
        pmma_ratio = (pmma_ch_intensity + pmma_co_intensity) / g_region_intensity

        print("Verhältnis von PMMA zu G-Peak ist: " + str(pmma_ratio))
        
        if pmma_ratio > PMMA_SCHWELLE:
            cluster_identitaeten[f"Cluster {i}"] = "PMMA-Rückstand"
            continue # Überspringe die restliche Analyse für dieses Cluster

        # 3. Prüfe auf Substrat als Fallback (G-Peak-Prominenz)
        g_peak_intensity_max = spectrum_graphen.spectral_data[g_peak_mask].max() if g_peak_mask.any() else 0
        baseline_mask_links = (spectrum_graphen.spectral_axis >= G_BASELINE_FENSTER_LINKS[0]) & (spectrum_graphen.spectral_axis <= G_BASELINE_FENSTER_LINKS[1])
        baseline_mask_rechts = (spectrum_graphen.spectral_axis >= G_BASELINE_FENSTER_RECHTS[0]) & (spectrum_graphen.spectral_axis <= G_BASELINE_FENSTER_RECHTS[1])
        baseline_links_mean = np.mean(spectrum_graphen.spectral_data[baseline_mask_links]) if baseline_mask_links.any() else g_peak_intensity_max
        baseline_rechts_mean = np.mean(spectrum_graphen.spectral_data[baseline_mask_rechts]) if baseline_mask_rechts.any() else g_peak_intensity_max
        lokale_baseline = (baseline_links_mean + baseline_rechts_mean) / 2
        g_peak_prominenz = g_peak_intensity_max - lokale_baseline

        if g_peak_prominenz < G_PEAK_PROMINENZ_SCHWELLE:
            cluster_identitaeten[f"Cluster {i}"] = "Substrat"
            continue

        # 4. Wenn es Graphen ist, führe die detaillierte Analyse durch
        
        # G-Peak-Position durch Fitting
        pos_g = G_PEAK_REF
        if g_peak_mask.any():
            x_g, y_g = spectrum_graphen.spectral_axis[g_peak_mask], spectrum_graphen.spectral_data[g_peak_mask]
            try:
                p0_g = [np.max(y_g) - np.min(y_g), x_g[np.argmax(y_g)], 15, np.min(y_g)]
                params_g, _ = curve_fit(lorentzian, x_g, y_g, p0=p0_g)
                pos_g = params_g[1]
            except RuntimeError:
                print(f"  WARNUNG: G-Peak-Fit für Cluster-Index {i} fehlgeschlagen.")

        # 2D-Peak-Analyse (FWHM und Position)
        fwhm_2d = 999
        pos_2d = TWOD_PEAK_REF
        x_peak, y_peak = spectrum_graphen.spectral_axis, spectrum_graphen.spectral_data
        mask_2d = (x_peak >= TWOD_PEAK_FENSTER[0]) & (x_peak <= TWOD_PEAK_FENSTER[1])
        if mask_2d.any():
            x_2d, y_2d = x_peak[mask_2d], y_peak[mask_2d]
            try:
                p0_2d = [np.max(y_2d) - np.min(y_2d), x_2d[np.argmax(y_2d)], 20, np.min(y_2d)]
                params_2d, _ = curve_fit(lorentzian, x_2d, y_2d, p0=p0_2d)
                pos_2d = params_2d[1]
                fwhm_2d = 2 * abs(params_2d[2])
            except RuntimeError:
                print(f"  WARNUNG: 2D-Peak-Fit für Cluster-Index {i} fehlgeschlagen.")
        print(f"  -> Cluster {gefundene_cluster_ids[i]}: FWHM(2D)={fwhm_2d:.2f} cm-1, Pos(G)={pos_g:.2f} cm-1, Pos(2D)={pos_2d:.2f} cm-1")

        # Berechnung der Verhältnisse
        twod_peak_intensity = spectrum_graphen.spectral_data[mask_2d].max() if mask_2d.any() else 0
        i2d_ig_ratio = twod_peak_intensity / g_peak_intensity_max if g_peak_intensity_max > 0 else 0
        d_peak_mask = (spectrum_graphen.spectral_axis >= D_PEAK_FENSTER[0]) & (spectrum_graphen.spectral_axis <= D_PEAK_FENSTER[1])
        d_peak_intensity = spectrum_graphen.spectral_data[d_peak_mask].max() if d_peak_mask.any() else 0
        id_ig_ratio = d_peak_intensity / g_peak_intensity_max if g_peak_intensity_max > 0 else 0

        # Hierarchisches Regelwerk für Schichtdicke
        schicht_label = ""
        if fwhm_2d < FWHM_GRENZE_SLG:
            if i2d_ig_ratio > I2D_IG_GRENZE_SLG:
                schicht_label = "Monolage Graphen"
            else:
                schicht_label = "Dotiertes SLG"
        elif fwhm_2d < FWHM_GRENZE_BLG:
            schicht_label = "Zweischichtiges Graphen"
        else:
            spectrum_silizium = mean_spectra_silizium[i]
            si_mask = (spectrum_silizium.spectral_axis >= SI_PEAK_FENSTER[0]) & (spectrum_silizium.spectral_axis <= SI_PEAK_FENSTER[1])
            si_verhaeltnis = min(spectrum_silizium.spectral_data[si_mask].max(), 1.0) if si_mask.any() else 0
            if si_verhaeltnis > 0.75:
                schicht_label = "Zweischichtiges Graphen"
            else:
                schicht_label = "Viele Lagen Graphen"
        
        # Regelwerk für Defektqualität
        qualitaet_label = ""
        if id_ig_ratio >= ID_IG_GRENZE_HOCH:
            qualitaet_label = " (stark defekt)"
        elif id_ig_ratio >= ID_IG_GRENZE_NIEDRIG:
            qualitaet_label = " (defekt)"
        else:
            qualitaet_label = " (hohe Qualität)"
            
        # Regelwerk für Strain/Doping
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
        
        # Finales Label zusammensetzen
        cluster_identitaeten[f"Cluster {i}"] = schicht_label + qualitaet_label + strain_doping_label

    print("Identifizierung abgeschlossen:", cluster_identitaeten)
    return cluster_identitaeten

def DBSCAN_Clustering(eps, min_samples, valid_mask_1d, scaled_scores, h, w):
    print(f"Führe DBSCAN-Clustering mit eps={eps:.4f} und min_samples={min_samples} durch...")



    # Initialisiere und führe das DBSCAN-Modell aus
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan_model.fit_predict(scaled_scores)

    # Visualisiere das Cluster-Ergebnis
    cluster_map_1d = np.full(h * w, np.nan)
    cluster_map_1d[valid_mask_1d] = cluster_labels
    cluster_map_2d = cluster_map_1d.reshape((h, w))

    # plt.figure(figsize=(10, 7))
    # plt.imshow(cluster_map_2d, cmap='viridis') 
    # plt.colorbar(label='Cluster-ID')
    # plt.title(f'DBSCAN Clustering (Automatisch: eps={eps:.4f}, min_samples={min_samples})')
    
    return cluster_labels

def finde_besten_eps(scores, min_samples):
    print("Bestimme optimalen eps-Wert automatisch...")
    
    
    # Berechne die Distanz für jeden Punkt zu seinen k-nächsten Nachbarn
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(scores)
    distances, indices = neighbors_fit.kneighbors(scores)
    
    # Nimm die Distanz zum k-ten Nachbarn und sortiere sie
    distances = np.sort(distances, axis=0)
    distances = distances[:, min_samples-1]
    
    # Finde den Ellenbogenpunkt programmatisch mit kneed
    # Die Kurve ist "konvex" und "ansteigend"
    kneedle = KneeLocator(range(len(distances)), distances, S=20, curve='convex', direction='increasing')
    
    # Der y-Wert des Ellenbogens ist unser optimales eps
    optimal_eps = kneedle.knee_y*0.6
    print(f"-> Optimaler eps-Wert gefunden: {optimal_eps:.4f}")

    # Erstelle den Plot zur visuellen Überprüfung (optional, aber gut für die Arbeit)
    # plt.figure(figsize=(10, 7))
    # kneedle.plot_knee()
    # plt.title("K-Distanz-Graph zur Bestimmung des optimalen eps")
    # plt.xlabel("Datenpunkte (sortiert nach Abstand)")
    # plt.ylabel(f"Abstand zum {min_samples}. nächsten Nachbarn (eps)")
    # plt.grid(True)
    # plt.show()
    
    return optimal_eps

def PCA_K_Mean(Pfad_Raman_Karte): 
    # Schritt 1: Lade und verarbeite die Raman-Karte.
    valid_mask_1d, karte_silizium, karte_graphen, h, w = Laden_Vorverarbeitung(Pfad_Raman_Karte)

    # Schritt 2: Trenne Graphen-Spektren von Substrat-Spektren
    graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(karte_graphen, valid_mask_1d, h, w)

    # Schritt 3: PCA nur auf den Graphen-Daten durchführen
    # Die PCA-Funktion muss nun die 'graphen_mask_1d' anstelle der 'valid_mask_1d' verwenden
    scores, loadings, optimal_pcs_gefunden = PCA(graphen_mask_1d, karte_graphen, h, w)

    # Schritt 4: K-Means-Clustering nur auf den Graphen-Daten durchführen
    # Die K_Mean-Funktion muss ebenfalls die 'graphen_mask_1d' verwenden
    graphen_cluster_labels = K_Mean(graphen_mask_1d, scores, h, w)
    
    # Schritt 5: Erstelle die finale Cluster-Karte und kombiniere die Ergebnisse
    print("Kombiniere Ergebnisse zu finaler Cluster-Karte...")
    final_cluster_map_1d = np.full(h * w, np.nan)
    
    # Weise dem Substrat einen festen Cluster-Label zu (z.B. 0)
    SUBSTRAT_LABEL = 0
    final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
    
    # Weise den Graphen-Clustern die gefundenen Labels zu, aber verschoben, um Konflikte zu vermeiden
    # (z.B. wenn K-Means auch ein Label 0 findet)
    final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels + 1 # Verschiebt Labels zu 1, 2, 3,...
    
    final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

    # Schritt 6: Berechne mittlere Spektren für jeden Cluster (inkl. Substrat)
    print("Berechne mittlere Spektren für jeden finalen Cluster...")
    unique_final_labels = sorted([label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)])
    
    mean_spectra_graphen = []
    mean_spectra_silizium = []
    gefundene_cluster_ids = []

    for label in unique_final_labels:
        cluster_mask_1d = (final_cluster_map_1d == label)
        cluster_mask_2d = cluster_mask_1d.reshape((h, w))
        
        if np.any(cluster_mask_2d):
            mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
            mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean) 
            gefundene_cluster_ids.append(int(label))

    # Schritt 7: Cluster automatisch identifizieren und Ergebnisse plotten
    neue_labels_map = identifiziere_cluster(mean_spectra_graphen, mean_spectra_silizium, gefundene_cluster_ids)
    
    finale_plot_labels = [f"Cluster {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}" for i, original_id in enumerate(gefundene_cluster_ids)]
    
    global_max_intensity = 0
    for spectrum in mean_spectra_graphen:
        # Finde das Maximum in jedem Spektrum und vergleiche es mit dem bisherigen globalen Maximum
        current_max = np.max(spectrum.spectral_data)
        if current_max > global_max_intensity:
            global_max_intensity = current_max
    
    # Setze das obere Limit für die Y-Achse auf das globale Maximum plus 10% Puffer
    plot_ylim = global_max_intensity * 1.1
    print(f"  Setze einheitliches Y-Achsen-Limit auf: {plot_ylim:.4f} a.u.")

    # --- Manuelles Plotten mit Subplots ---
    # Bestimme die Anzahl der benötigten Plots
    num_plots = len(mean_spectra_graphen)
    # Erstelle ein Grid für die Plots (z.B. 2 Spalten)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols # Berechnet die benötigte Anzahl an Zeilen
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows), squeeze=False)
    fig.suptitle('Mittlere Spektren der gefundenen Cluster', fontsize=16)

    # Iteriere durch die Spektren und die Achsen des Grids
    for i, (spectrum, label) in enumerate(zip(mean_spectra_graphen, finale_plot_labels)):
        ax = axes.flat[i] # Wähle die nächste freie Achse
        
        # Plotte das Spektrum auf dieser Achse
        ax.plot(spectrum.spectral_axis, spectrum.spectral_data, label=label)
        
        # Setze Titel, Labels und das Y-Limit
        ax.set_title(label)
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_ylim(0, plot_ylim)
        ax.grid(True, linestyle='--')
        ax.legend()

    # Verstecke leere, ungenutzte Plots im Grid
    for i in range(num_plots, len(axes.flat)):
        axes.flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Passt das Layout an den Haupttitel an

    # Die finale Cluster-Karte wird weiterhin in einem separaten Fenster angezeigt
    plt.figure(figsize=(10, 7))
    cmap = plt.cm.get_cmap('viridis', len(unique_final_labels))
    plt.imshow(final_cluster_map_2d, cmap=cmap) 
    cbar = plt.colorbar(ticks=unique_final_labels)
    cbar.set_label('Cluster-ID')
    plt.title(f'Finale Cluster-Karte (K-Mean auf Graphen)')

    plt.show()


def PCA_DBSCAN(Pfad_Raman_Karte):
    # Schritt 1: Lade und verarbeite die Raman-Karte.
    valid_mask_1d, karte_silizium, karte_graphen, h, w = Laden_Vorverarbeitung(Pfad_Raman_Karte)

    # Schritt 2: Trenne Graphen-Spektren von Substrat-Spektren
    graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(karte_graphen, valid_mask_1d, h, w)

    # --- FOKUSSIERTE ANALYSE NUR AUF GRAPHEN-DATEN ---
    
    # Schritt 3: PCA nur auf den Graphen-Daten durchführen
    scores, loadings, optimal_pcs_gefunden = PCA(graphen_mask_1d, karte_graphen, h, w)

    # Schritt 4: Skalierung der Graphen-Scores
    print("Skaliere PCA-Scores der Graphen-Region für die Cluster-Analyse...")
    scores_for_scaling = np.array(scores).T
    scaler = StandardScaler()
    scaled_scores = scaler.fit_transform(scores_for_scaling)
    

    # Schritt 5: Bestimme die DBSCAN-Parameter automatisch auf den skalierten Graphen-Daten
    min_samples_auto = 80
    print(f"Bestimme min_samples automatisch: {min_samples_auto}")
    eps_auto = finde_besten_eps(scaled_scores, min_samples_auto)
    
    # Schritt 6: DBSCAN-Clustering nur auf den Graphen-Daten durchführen
    graphen_cluster_labels = DBSCAN_Clustering(eps_auto, min_samples_auto, graphen_mask_1d, scaled_scores, h, w)
    
    # --- ZUSAMMENFÜHREN DER ERGEBNISSE ---

    # Schritt 7: Erstelle die finale Cluster-Karte und kombiniere die Ergebnisse
    print("Kombiniere Ergebnisse zu finaler Cluster-Karte...")
    final_cluster_map_1d = np.full(h * w, np.nan)
    
    # Weise dem Substrat einen festen Cluster-Label zu (z.B. -2, um von DBSCAN-Rauschen -1 unterscheidbar zu sein)
    SUBSTRAT_LABEL = -2
    final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
    
    # Weise den Graphen-Clustern die gefundenen Labels zu
    final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels
    
    final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

    # Schritt 8: Berechne mittlere Spektren für jeden finalen Cluster (inkl. Substrat)
    print("Berechne mittlere Spektren für jeden finalen Cluster...")
    unique_final_labels = sorted([label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)])
    
    mean_spectra_graphen = []
    mean_spectra_silizium = []
    gefundene_cluster_ids = []

    for label in unique_final_labels:
        cluster_mask_1d = (final_cluster_map_1d == label)
        cluster_mask_2d = cluster_mask_1d.reshape((h, w))
        
        if np.any(cluster_mask_2d):
            mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
            mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean) 
            gefundene_cluster_ids.append(int(label))

    # Schritt 9: Cluster automatisch identifizieren und Ergebnisse plotten
    # Wir müssen die identifiziere_cluster Funktion anpassen, um das Substrat-Label zu kennen
    neue_labels_map = identifiziere_cluster(mean_spectra_graphen, mean_spectra_silizium, gefundene_cluster_ids, substrat_label=SUBSTRAT_LABEL)
    
    finale_plot_labels = [f"Cluster {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}" for i, original_id in enumerate(gefundene_cluster_ids)]
    
    # Schritt 8: Ergebnisse plotten
    print("Plotte die finalen mittleren Spektren mit einheitlicher Y-Achse...")

    # --- NEUER TEIL: Globales Maximum finden ---
    global_max_intensity = 0
    for spectrum in mean_spectra_graphen:
        # Finde das Maximum in jedem Spektrum und vergleiche es mit dem bisherigen globalen Maximum
        current_max = np.max(spectrum.spectral_data)
        if current_max > global_max_intensity:
            global_max_intensity = current_max
    
    # Setze das obere Limit für die Y-Achse auf das globale Maximum plus 10% Puffer
    plot_ylim = global_max_intensity * 1.1
    print(f"  Setze einheitliches Y-Achsen-Limit auf: {plot_ylim:.4f} a.u.")

    # --- Manuelles Plotten mit Subplots ---
    # Bestimme die Anzahl der benötigten Plots
    num_plots = len(mean_spectra_graphen)
    # Erstelle ein Grid für die Plots (z.B. 2 Spalten)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols # Berechnet die benötigte Anzahl an Zeilen
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows), squeeze=False)
    fig.suptitle('Mittlere Spektren der gefundenen Cluster', fontsize=16)

    # Iteriere durch die Spektren und die Achsen des Grids
    for i, (spectrum, label) in enumerate(zip(mean_spectra_graphen, finale_plot_labels)):
        ax = axes.flat[i] # Wähle die nächste freie Achse
        
        # Plotte das Spektrum auf dieser Achse
        ax.plot(spectrum.spectral_axis, spectrum.spectral_data, label=label)
        
        # Setze Titel, Labels und das Y-Limit
        ax.set_title(label)
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_ylim(0, plot_ylim) # HIER WIRD DAS EINHEITLICHE LIMIT GESETZT
        ax.grid(True, linestyle='--')
        ax.legend()

    # Verstecke leere, ungenutzte Plots im Grid
    for i in range(num_plots, len(axes.flat)):
        axes.flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Passt das Layout an den Haupttitel an

    # Die finale Cluster-Karte wird weiterhin in einem separaten Fenster angezeigt
    plt.figure(figsize=(10, 7))
    cmap = plt.cm.get_cmap('viridis', len(unique_final_labels))
    plt.imshow(final_cluster_map_2d, cmap=cmap) 
    cbar = plt.colorbar(ticks=unique_final_labels)
    cbar.set_label('Cluster-ID')
    plt.title(f'Finale Cluster-Karte (DBSCAN auf Graphen)')

    plt.show()