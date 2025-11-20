import numpy as np
import matplotlib.pyplot as plt

# 1. EINSTELLUNGEN & DATEN
# ------------------------
np.random.seed(42) # Für Reproduzierbarkeit

# Wir erzeugen zwei "Cluster" (ähnlich wie deine Graphen-Daten im Feature-Space)
# Das symbolisiert z.B. "Monolage" und "Substrat"
data_cluster_1 = np.random.normal(loc=[0.3, 0.7], scale=0.08, size=(200, 2))
data_cluster_2 = np.random.normal(loc=[0.7, 0.3], scale=0.08, size=(200, 2))
data = np.vstack([data_cluster_1, data_cluster_2])

# SOM Parameter
grid_h, grid_w = 10, 10  # 10x10 Neuronen Gitter
input_dim = 2            # 2D Daten (für die Visualisierung)
epochs = 1000            # Anzahl der Trainingsschritte
learning_rate_start = 0.5
radius_start = 3.0       # Radius der Gauß-Glocke (Kooperation)

# Initialisierung der Gewichte (Zufällig in der Mitte startend)
# Form: (Höhe, Breite, Input-Dimension)
weights = np.random.rand(grid_h, grid_w, input_dim) * 0.5 + 0.25
weights_initial = weights.copy() # Speichern für den "Vorher"-Plot

# 2. DER SOM-ALGORITHMUS (Iteratives Training)
# --------------------------------------------
# Dies implementiert exakt deine Beschreibung: Wettbewerb -> Kooperation -> Anpassung

for epoch in range(epochs):
    # Zerfallsrate für Lernrate und Radius (werden kleiner mit der Zeit)
    decay = 1.0 - (epoch / epochs)
    learning_rate = learning_rate_start * decay
    radius = radius_start * decay
    
    # Ein zufälliges Datenpunkt auswählen (Raman-Spektrum)
    idx = np.random.randint(0, data.shape[0])
    sample = data[idx]
    
    # SCHRITT 1: WETTBEWERB (Finde BMU)
    # Euklidische Distanz berechnen
    diff = weights - sample
    dists = np.sum(diff**2, axis=-1) # Quadratische Distanz
    bmu_idx = np.unravel_index(np.argmin(dists), (grid_h, grid_w))
    
    # SCHRITT 2 & 3: KOOPERATION & ANPASSUNG
    # Wir aktualisieren alle Gewichte basierend auf Abstand zur BMU
    for i in range(grid_h):
        for j in range(grid_w):
            # Distanz im Gitter (Topologische Distanz)
            grid_dist = np.sqrt((i - bmu_idx[0])**2 + (j - bmu_idx[1])**2)
            
            # Gaußsche Nachbarschaftsfunktion (Kooperation)
            if grid_dist <= radius:
                influence = np.exp(-(grid_dist**2) / (2 * (radius**2)))
                
                # Gewichts-Update (Anpassung)
                weights[i, j] += learning_rate * influence * (sample - weights[i, j])

# 3. VISUALISIERUNG
# -----------------
def plot_som_grid(ax, w, d, title):
    # Datenpunkte plotten
    ax.scatter(d[:, 0], d[:, 1], c='lightgray', alpha=0.7, label='Datenpunkte', s=10)
    
    # SOM Gitter zeichnen (Das "Netz")
    # Horizontale Linien
    for i in range(w.shape[0]):
        ax.plot(w[i, :, 0], w[i, :, 1], 'k-', lw=1, alpha=0.8, marker='o', markersize=3)
    # Vertikale Linien
    for j in range(w.shape[1]):
        ax.plot(w[:, j, 0], w[:, j, 1], 'k-', lw=1, alpha=0.8)
        
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_som_grid(axes[0], weights_initial, data, "(a) Initialisierung (Zufall)")
plot_som_grid(axes[1], weights, data, "(b) Nach 1000 Iterationen (Topologie gelernt)")

plt.tight_layout()
plt.show()
# plt.savefig("som_visualisierung.png", dpi=300) # Zum Speichern einkommentieren