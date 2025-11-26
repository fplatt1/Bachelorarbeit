import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. SOM Klasse (Minimal-Implementierung) ---
class SimpleSOM:
    def __init__(self, height, width, input_dim):
        self.height = height
        self.width = width
        # Initialisierung der Gewichte (zufällig im Bereich der Daten)
        # Wir starten zentraler, damit man das "Entfalten" besser sieht
        self.weights = np.random.rand(height, width, input_dim) * 0.5 + 0.25
        
    def get_best_matching_unit(self, sample):
        """Findet das Neuron mit dem geringsten Abstand zum Input-Sample"""
        dist = np.linalg.norm(self.weights - sample, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dist), (self.height, self.width))
        return bmu_idx

    def train_step(self, sample, learning_rate, radius):
        """Ein Trainingsschritt: Update des BMU und der Nachbarn"""
        bmu_idx = self.get_best_matching_unit(sample)
        
        # Grid-Koordinaten für alle Neuronen erstellen
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        # Euklidischer Abstand aller Neuronen zum BMU im Gitter
        dist_to_bmu = np.sqrt((x - bmu_idx[1])**2 + (y - bmu_idx[0])**2)
        
        # Nachbarschaftsfunktion (Gauß)
        influence = np.exp(-(dist_to_bmu**2) / (2 * (radius**2)))
        
        # Gewichte anpassen: W_neu = W_alt + LR * Influence * (Input - W_alt)
        # Wir müssen dimensionsgerecht "broadcasten"
        influence = influence[:, :, np.newaxis]
        self.weights += learning_rate * influence * (sample - self.weights)

# --- 2. Daten Generierung (Zwei Blobs wie in deinem Beispiel) ---
np.random.seed(42)
# Cluster 1
data1 = np.random.normal(loc=[0.2, 0.2], scale=0.1, size=(200, 2))
# Cluster 2
data2 = np.random.normal(loc=[0.8, 0.8], scale=0.1, size=(200, 2))
# Noise/Rauschen dazwischen
noise = np.random.rand(50, 2)

data = np.vstack([data1, data2, noise])
np.random.shuffle(data)

# --- 3. Animation Setup ---

# Parameter
grid_h, grid_w = 10, 10  # 10x10 Neuronen
n_iterations = 200      # Anzahl der Frames in der Animation
batch_size_per_frame = 5 # Wieviele Trainingsschritte pro Frame
init_lr = 0.5
init_radius = max(grid_h, grid_w) / 2

som = SimpleSOM(grid_h, grid_w, 2)

fig, ax = plt.subplots(figsize=(8, 8))
fig.suptitle('Self-Organizing Map (SOM) Training', fontsize=16)

# Statische Elemente (Datenpunkte)
ax.scatter(data[:, 0], data[:, 1], c='lightgray', alpha=0.7, s=20, label='Daten')

# Dynamische Elemente (Das Gitter)
# Wir nutzen LineCollection nicht, sondern plotten einfach neu (einfacher zu verstehen)
lines = []
dots, = ax.plot([], [], 'o', color='black', markersize=4, zorder=5) # Die Neuronen

def animate(i):
    # 1. Training durchführen (mehrere Schritte pro Frame für Geschwindigkeit)
    # Decay von Lernrate und Radius über die Zeit
    progress = i / n_iterations
    current_lr = init_lr * (1 - progress)
    current_radius = init_radius * (1 - progress)
    # Radius darf nicht 0 werden
    current_radius = max(current_radius, 0.5)
    
    for _ in range(batch_size_per_frame):
        sample = data[np.random.randint(0, len(data))]
        som.train_step(sample, current_lr, current_radius)
    
    # 2. Visualisierung updaten
    # Alte Linien entfernen
    global lines
    for line in lines:
        line.remove()
    lines = []
    
    w = som.weights
    
    # Horizontale Linien zeichnen
    for r in range(grid_h):
        l, = ax.plot(w[r, :, 0], w[r, :, 1], 'b-', alpha=0.6, linewidth=1)
        lines.append(l)
        
    # Vertikale Linien zeichnen
    for c in range(grid_w):
        l, = ax.plot(w[:, c, 0], w[:, c, 1], 'b-', alpha=0.6, linewidth=1)
        lines.append(l)
    
    # Die Punkte (Knoten) updaten
    dots.set_data(w[:, :, 0].flatten(), w[:, :, 1].flatten())
    
    ax.set_title(f'Iteration: {i * batch_size_per_frame} | Radius: {current_radius:.2f}')
    return lines + [dots]

# Animation erstellen
ani = animation.FuncAnimation(fig, animate, frames=n_iterations, interval=50, blit=False)

# Optional: Achsen fixieren
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.legend()

print("Animation wird generiert... Bitte warten.")
# Um es als GIF zu speichern (erfordert imagemagick oder pillow):
ani.save('som_training.gif', writer='pillow', fps=20)

plt.show()