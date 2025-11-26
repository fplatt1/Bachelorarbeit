import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_blobs

# --- 1. Daten Generierung ---
# Wir erhöhen die Standardabweichung (cluster_std) deutlich auf 1.8.
# Wir nutzen 4 Cluster für mehr Komplexität.
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=1.8, random_state=42) # type: ignore

# --- 2. K-Means Generator ---
def kmeans_generator(X, n_clusters):
    # Zufällige Initialisierung im Datenbereich
    rng = np.random.RandomState() 
    min_x, min_y = X.min(axis=0)
    max_x, max_y = X.max(axis=0)
    
    # Start-Zentren zufällig wählen
    centers = rng.rand(n_clusters, 2)
    centers[:, 0] = centers[:, 0] * (max_x - min_x) + min_x
    centers[:, 1] = centers[:, 1] * (max_y - min_y) + min_y
    
    labels = np.zeros(len(X))
    
    # Yield Initial State
    yield centers, labels, "Start: Zufällige Zentren"

    iteration = 0
    while True:
        iteration += 1
        
        # 1. Zuordnung (Expectation)
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        # Zeige den Schritt der Zuordnung
        yield centers, new_labels, f"{iteration} Punkte zuordnen"
        
        # Check auf Veränderung der Labels
        if np.array_equal(labels, new_labels) and iteration > 1:
            yield centers, new_labels, "Konvergiert! (Fertig)"
            break
        labels = new_labels

        # 2. Update Zentren (Maximization)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centers[i] 
                                for i in range(n_clusters)])
        
        # Zeige den Schritt der Zentren-Verschiebung
        yield new_centers, labels, f"{iteration} Zentren bewegen"
        
        # Check auf Konvergenz der Position
        if np.allclose(centers, new_centers, atol=1e-4):
            yield new_centers, labels, "Konvergiert! (Fertig)"
            break
            
        centers = new_centers

# --- 3. Animation Setup ---
n_clusters = 4
gen = kmeans_generator(X, n_clusters)

fig, ax = plt.subplots(figsize=(8, 6))
# Zoom etwas raus, um alles zu sehen
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

# Scatterplot Daten
scat = ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.6)
# Scatterplot Zentren
centers_scat = ax.scatter([], [], c='red', s=200, marker='X', edgecolors='black', linewidths=2, zorder=10)

def animate(data):
    centers, labels, text = data
    
    if "Start" in text:
        scat.set_color('lightgray')
    else:
        scat.set_array(labels)
        # Tab10 ist gut für unterscheidbare Farben
        scat.set_cmap('tab10')
        scat.set_clim(-0.5, 9.5) 
    
    centers_scat.set_offsets(centers)
    ax.set_title(f'K-Means: {text}', fontsize=14)
    return scat, centers_scat

# Intervall erhöht für bessere Sichtbarkeit
ani = animation.FuncAnimation(fig, animate, frames=gen, interval=800, blit=False, save_count=50, repeat=False)

ani.save('kmeans_training.gif', writer='pillow', fps=1)
plt.tight_layout()
plt.show()