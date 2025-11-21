import matplotlib.pyplot as plt
import numpy as np

def lorentzian(x, x0, gamma, A):
    """Erzeugt eine Lorentz-Kurve (typisch für Raman-Peaks)."""
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

# 1. Synthetische Daten erzeugen (Simuliertes Spektrum mit PMMA-Resten)
x = np.linspace(1200, 3200, 2000)
baseline = 5 + 0.002 * x  # Leichte Steigung der Baseline

# Peaks definieren (Graphen + Defekt + PMMA)
# D-Peak (Defekt)
y_d = lorentzian(x, 1350, 15, 80)
# G-Peak (Referenz)
y_g = lorentzian(x, 1580, 10, 200)
# 2D-Peak (Monolage)
y_2d = lorentzian(x, 2700, 25, 350)
# PMMA (Breiter Buckel um 2950 + kleiner bei 1730)
y_pmma = lorentzian(x, 2950, 80, 50) + lorentzian(x, 1730, 15, 30)

# Gesamtspektrum + Rauschen
y = baseline + y_d + y_g + y_2d + y_pmma + np.random.normal(0, 2, len(x))

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y, color='black', linewidth=1.2, label='Raman-Spektrum')

# Farben für die Markierungen
color_g = '#1f77b4'   # Blau
color_2d = '#d62728'  # Rot
color_d = '#ff7f0e'   # Orange
color_pmma = 'gray'   # Grau

# 2. Merkmale markieren

# A) G-Peak (Position & Intensität)
g_idx = np.abs(x - 1580).argmin()
ax.vlines(x[g_idx], baseline[g_idx], y[g_idx], color=color_g, linestyle='--', alpha=0.7)
ax.annotate(r'$I(G)$', xy=(1580, y[g_idx]), xytext=(1520, y[g_idx]), 
            color=color_g, fontsize=12, arrowprops=dict(arrowstyle='->', color=color_g))
ax.text(1580, baseline[g_idx]-15, 'Pos(G)\n~1580 $cm^{-1}$', color=color_g, ha='center', va='top')

# B) 2D-Peak (Position, Intensität, FWHM)
twod_idx = np.abs(x - 2700).argmin()
ax.vlines(x[twod_idx], baseline[twod_idx], y[twod_idx], color=color_2d, linestyle='--', alpha=0.7)
ax.annotate(r'$I(2D)$', xy=(2700, y[twod_idx]), xytext=(2600, y[twod_idx]), 
            color=color_2d, fontsize=12, arrowprops=dict(arrowstyle='->', color=color_2d))
ax.text(2700, baseline[twod_idx]-15, 'Pos(2D)\n~2700 $cm^{-1}$', color=color_2d, ha='center', va='top')

# FWHM einzeichnen (Full Width at Half Maximum)
half_max = baseline[twod_idx] + (y[twod_idx] - baseline[twod_idx]) / 2
# Finde Indizes links und rechts vom Peak auf halber Höhe
idx_left = np.abs(x[:twod_idx] - (x[twod_idx]-25)).argmin() # Näherung für Demo
idx_right = np.abs(x[twod_idx:] - (x[twod_idx]+25)).argmin() + twod_idx
ax.hlines(half_max, x[idx_left], x[idx_right], color=color_2d, linewidth=2)
ax.text(2700, half_max + 10, 'FWHM(2D)', color=color_2d, ha='center', fontsize=10, fontweight='bold')

# C) D-Peak (Defekt)
d_idx = np.abs(x - 1350).argmin()
ax.vlines(x[d_idx], baseline[d_idx], y[d_idx], color=color_d, linestyle='--', alpha=0.7)
ax.text(1350, y[d_idx]+20, r'$I(D)$ (Defekt)', color=color_d, ha='center')

# D) PMMA Bereiche (Schattierung)
# Bereich 1: C=O (~1730)
ax.axvspan(1700, 1760, color=color_pmma, alpha=0.2)
ax.text(1730, y.max()*0.2, 'PMMA\n(C=O)', color='black', alpha=0.7, ha='center', fontsize=9)

# Bereich 2: C-H (~2800-3100)
ax.axvspan(2800, 3100, color=color_pmma, alpha=0.2)
ax.text(2950, y.max()*0.25, 'PMMA-Hintergrund\n(C-H Streckung)', color='black', alpha=0.7, ha='center', fontsize=9)

# 3. Erklär-Box für Ratios (Verhältnisse)
textstr = '\n'.join((
    r'$\bf{Extrahierte\ Merkmale:}$',
    r'1. $I(D)/I(G)$ (Defektdichte)',
    r'2. FWHM (2D) (Lagenzahl)',
    r'3. $I(2D)/I(G)$ (Qualität)',
    r'4. Pos(G) (Strain/Doping)',
    r'5. Pos(2D) (Strain/Doping)',
    r'6. PMMA-Ratio (Sauberkeit)'
))
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# Layout-Optimierung
ax.set_xlabel('Wellenzahl ($cm^{-1}$)', fontsize=12)
ax.set_ylabel('Intensität (a.u.)', fontsize=12)
ax.set_title('Visualisierung der Merkmalsextraktion am Raman-Spektrum', fontsize=14, fontweight='bold')
ax.set_xlim(1200, 3200)
ax.set_ylim(0, y.max()*1.2)
ax.grid(True, linestyle=':', alpha=0.6)

# Speichern und Anzeigen
plt.tight_layout()
plt.savefig('Feature_Extraction_Visualization.png', dpi=300)
plt.show()