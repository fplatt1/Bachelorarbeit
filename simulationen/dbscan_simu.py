import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons

# --- 1. Daten Generierung (Moons + Extra Rauschen) ---
# Basis-Cluster (Monde)
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Wir fügen explizite Rauschpunkte (Outliers) hinzu, die zufällig verteilt sind
n_noise = 20
np.random.seed(42)
x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5

# Zufällige Punkte im gesamten Bereich
X_noise = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_noise, 2))

# Alles zusammenfügen
X = np.vstack([X_moons, X_noise])

# --- 2. DBSCAN Generator-Logik ---
def dbscan_generator(X, eps, min_samples):
    # Status-Codes für Visualisierung:
    # -2 = Unbesucht
    # -1 = Rauschen (Noise)
    # >= 0 = Cluster ID
    labels = np.full(len(X), -2) 
    visited = np.full(len(X), False)
    cluster_id = 0
    
    # Hilfsfunktion für Distanzberechnung
    def get_neighbors(p_idx):
        dists = np.linalg.norm(X - X[p_idx], axis=1)
        return np.where(dists <= eps)[0]

    for i in range(len(X)):
        if visited[i]:
            continue
            
        visited[i] = True
        neighbors = get_neighbors(i)
        
        # Visueller Schritt: Wir prüfen Punkt i (Gelber Fokus)
        yield labels.copy(), i, "Prüfe Punkt..."
        
        if len(neighbors) < min_samples:
            # Als Rauschen markieren
            labels[i] = -1 
            yield labels.copy(), i, "Rauschen (Noise) entdeckt!"
        else:
            # Neuen Cluster starten
            labels[i] = cluster_id
            seeds = list(neighbors)
            yield labels.copy(), i, f"Neuer Cluster {cluster_id} gefunden"
            
            # Cluster expandieren (Region Growing)
            idx = 0
            while idx < len(seeds):
                current_point = seeds[idx]
                idx += 1
                
                if not visited[current_point]:
                    visited[current_point] = True
                    current_neighbors = get_neighbors(current_point)
                    
                    if len(current_neighbors) >= min_samples:
                        seeds.extend(current_neighbors)
                
                # Wenn Punkt noch keinem Cluster angehört (war unbesucht oder Rauschen)
                if labels[current_point] < 0:
                    labels[current_point] = cluster_id
                    # Wir zeigen das Wachstum nur alle paar Schritte, damit die Animation flüssig bleibt
                    if idx % 5 == 0:
                         yield labels.copy(), current_point, f"Cluster {cluster_id} wächst..."
            
            cluster_id += 1
            
    yield labels.copy(), -1, "Fertig!"

# --- 3. Animation Setup ---
eps = 0.2
min_samples = 5
gen = dbscan_generator(X, eps, min_samples)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

# Basis-Scatterplot
scat = ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, edgecolor='k', alpha=0.6)

# Such-Radius (Epsilon)
circle = plt.Circle((0,0), eps, color='green', fill=False, alpha=0.0, linewidth=2) # type: ignore
ax.add_patch(circle)

# Highlight für den aktuellen Punkt
current_marker, = ax.plot([], [], 'o', color='yellow', markeredgecolor='black', markersize=10, alpha=0)

def animate(data):
    current_labels, current_idx, status_text = data
    
    # Farben update
    colors = []
    for lbl in current_labels:
        if lbl == -2:
            colors.append('lightgray') # Unbesucht
        elif lbl == -1:
            colors.append('black')     # Rauschen
        else:
            # Cluster-Farben (zyklisch durch Tab10 Colormap)
            colors.append(plt.cm.tab10(lbl % 10)) # type: ignore
            
    scat.set_color(colors)
    
    # Fokus-Punkt und Kreis update
    if current_idx != -1:
        pos = X[current_idx]
        circle.center = pos
        circle.set_alpha(0.8)
        current_marker.set_data([pos[0]], [pos[1]])
        current_marker.set_alpha(1.0)
    else:
        circle.set_alpha(0.0)
        current_marker.set_alpha(0.0)
        
    ax.set_title(f'DBSCAN: {status_text}', fontsize=14)
    return scat, circle, current_marker

# Interval=20ms sorgt für eine zügige Animation, da DBSCAN viele Schritte hat
ani = animation.FuncAnimation(fig, animate, frames=gen, interval=20, blit=False, save_count=3000, repeat=False)

ani.save('dbscan_training.gif', writer='pillow', fps=5)
plt.tight_layout()
plt.show()