import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def draw_vector(v0, v1, ax=None):
    """Hilfsfunktion zum Zeichnen der Pfeile (Eigenvektoren)"""
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0,
                    color='red')
    ax.annotate('', xy=v1, xytext=v0, arrowprops=arrowprops)

# 1. Daten generieren (stark korrelierte Daten, damit PCA Sinn ergibt)
rng = np.random.RandomState(42)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

# 2. PCA durchführen
pca = PCA(n_components=2)
pca.fit(X)

# 3. Plotten
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], alpha=0.4, label='Datenpunkte')

# Mittelpunkt der Daten
mean = pca.mean_

# Zeichnen der Hauptkomponenten als Vektoren
# Die Länge der Pfeile entspricht hier der Varianz (Eigenvalues)
for length, vector, i in zip(pca.explained_variance_, pca.components_, [1, 2]):
    v = vector * 3 * np.sqrt(length) # Skalierung für Sichtbarkeit (3 Sigma)
    draw_vector(mean, mean + v, ax=ax)
    # Beschriftung der Pfeile
    ax.text(*(mean + v + 0.1), f'PC{i}', color='red', fontsize=12, weight='bold') # type: ignore

# Styling für die Präsentation
ax.set_title('PCA: Finden eines neuen Koordinatensystems', fontsize=14)
ax.set_xlabel('Ursprüngliches Feature X1')
ax.set_ylabel('Ursprüngliches Feature X2')
ax.grid(True, linestyle='--', alpha=0.6)
ax.axis('equal') # Wichtig, damit die Orthogonalität sichtbar ist!
plt.legend()

plt.tight_layout()
plt.show()

# 1. Höherdimensionale Daten generieren (z.B. 15 Dimensionen)
# Wir erzeugen Daten, bei denen die ersten paar Dimensionen wichtig sind, der Rest Rauschen
np.random.seed(42)
n_samples = 500
n_features = 15
# Eigenwerte fallen exponentiell ab -> typisch für echte Daten
eigenvalues = np.exp(-np.arange(n_features) * 0.5) 
cov_matrix = np.diag(eigenvalues)
X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=cov_matrix, size=n_samples)

# 2. PCA fitten
pca = PCA().fit(X)

# 3. Kumulative Varianz berechnen
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Finde den Punkt, wo 95% erreicht werden
threshold = 0.95
n_components_95 = np.argmax(cumulative_variance >= threshold) + 1

# 4. Plotten
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_features + 1), cumulative_variance, marker='o', linestyle='--', color='b')

# Linie für 95%
plt.axhline(y=threshold, color='r', linestyle='-', label=f'{threshold*100}% Schwelle')
plt.axvline(x=n_components_95, color='r', linestyle='-.') # type: ignore

# Highlighten des Schnittpunkts
plt.plot(n_components_95, cumulative_variance[n_components_95-1], 'ro', markersize=10)
plt.text(n_components_95 + 0.5, threshold - 0.05, 
         f'{n_components_95} Komponenten\nerklären >95% Varianz', 
         color='red', fontsize=12)

plt.title('Erklärte Varianz vs. Anzahl der Komponenten', fontsize=14)
plt.xlabel('Anzahl der Hauptkomponenten')
plt.ylabel('Kumulative erklärte Varianz (0 bis 1)')
plt.grid(True)
plt.legend()
plt.xticks(range(1, n_features + 1))

plt.tight_layout()
plt.show()