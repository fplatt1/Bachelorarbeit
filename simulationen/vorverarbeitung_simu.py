import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp

# --- 1. FUNKTION ZUM PLOTTEN (Bleibt gleich für konsistentes Design) ---
def save_plot(x, y, title, filename, show_baseline=None, annotate_spike=False):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color='#1f77b4', linewidth=1.2, label='Spektrum')
    
    if show_baseline is not None:
        plt.plot(x, show_baseline, color='orange', linestyle='--', linewidth=1.5, label='Erkannte Baseline')
        plt.legend(loc='upper right')

    if annotate_spike:
        # Wir wissen, der Spike ist bei Index 300
        spike_idx = 300
        plt.annotate('Kosmische Strahlung', 
                     xy=(x[spike_idx], y[spike_idx]), 
                     xytext=(x[spike_idx]+400, y[spike_idx]+100),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                     color='red')

    plt.title(title, fontsize=12, fontweight='bold', loc='left')
    plt.xlabel('Wellenzahl ($\mathrm{cm}^{-1}$)', fontsize=11) # type: ignore
    plt.ylabel('Intensität (a.u.)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Gespeichert: {filename}")

# --- 2. DATEN SIMULIEREN (Physikalisch realistisch) ---
# Wir erzeugen Numpy-Arrays, wie sie aus dem Messgerät kommen würden
x_axis = np.linspace(400, 3200, 1000)

# Graphen-Signal (G Peak ~1580, 2D Peak ~2700)
peaks = 50 * np.exp(-((x_axis - 1580)**2) / (2 * 12**2)) + \
        90 * np.exp(-((x_axis - 2700)**2) / (2 * 25**2))

# Störungen
baseline_drift = 100 + 0.08 * x_axis + 40 * np.sin(x_axis / 600)
noise = np.random.normal(0, 4, len(x_axis))
spikes = np.zeros_like(x_axis)
spikes[300] = 150 # Spike bei Index 300

# Das "schmutzige" Rohsignal als Numpy Array
y_raw_data = peaks + baseline_drift + noise + spikes


# --- 3. RAMANSPY PIPELINE ANWENDEN ---

# WICHTIG: Wir wandeln die Numpy-Daten in ein Ramanspy-Objekt um
# Damit funktionieren deine Filter genau wie in der Bachelorarbeit
spectrum_obj = rp.Spectrum(y_raw_data, x_axis)

# -- STUFE 1: Rohdaten speichern --
save_plot(spectrum_obj.spectral_axis, spectrum_obj.spectral_data, 
          "Rohdaten", "1_Rohdaten.png", annotate_spike=True)


# -- STUFE 2: Whitaker-Hayes --
# Wir wenden den Filter an und holen uns das Ergebnis als neues Objekt
despiker = rp.preprocessing.despike.WhitakerHayes()
spectrum_despiked = despiker.apply(spectrum_obj)

save_plot(spectrum_despiked.spectral_axis, spectrum_despiked.spectral_data,  # type: ignore
          "Nach Whitaker-Hayes (Despiking)", "2_Despiked.png")


# -- STUFE 3: Gauß-Filter --
denoiser = rp.preprocessing.denoise.Gaussian() # Ggf. Parameter anpassen: std=...
spectrum_denoised = denoiser.apply(spectrum_despiked)

# Für Plot 3 brauchen wir die Baseline separat, um sie anzuzeigen.
# Wir "schummeln" kurz und berechnen die Baseline auf den geglätteten Daten, 
# ohne sie abzuziehen, nur zum Anzeigen.
# Ramanspy zieht sie bei .apply() direkt ab, daher nutzen wir hier kurz 
# die interne Logik oder den ASLS Filter, um die Differenz zu sehen.
asls_calculator = rp.preprocessing.baseline.ASLS()
spectrum_final = asls_calculator.apply(spectrum_denoised)

# Die Baseline ist die Differenz aus "Denoised" und "Final"
calculated_baseline = spectrum_denoised.spectral_data - spectrum_final.spectral_data # type: ignore

save_plot(spectrum_denoised.spectral_axis, spectrum_denoised.spectral_data,  # type: ignore
          "Nach Gauß-Filter & Baseline-Erkennung", "3_Denoised.png",
          show_baseline=calculated_baseline)


# -- STUFE 4: Finales Spektrum --
save_plot(spectrum_final.spectral_axis, spectrum_final.spectral_data,  # type: ignore
          "Finales Spektrum (Baseline korrigiert)", "4_Final.png")

print("Fertig! Die Ramanspy-Pipeline wurde erfolgreich auf die Testdaten angewendet.")
