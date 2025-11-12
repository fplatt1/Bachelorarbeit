import numpy as np
import matplotlib.pyplot as plt

def lorentzian(x, x0, fwhm, amplitude):
    """ Erzeugt eine Lorentz-Funktion. """
    gamma = fwhm / 2
    return amplitude * (gamma**2 / ((x - x0)**2 + gamma**2))

# x-Achse für den Raman-Shift
x = np.linspace(400, 3000, 2000)

# --- Parameter für Spektrum (a): Ideales Graphen ---
params_si_ideal =   [520, 8, 0.8]
params_g_ideal =    [1582, 15, 1.0]
params_2d_ideal =   [2680, 28, 2.5] # I(2D)/I(G) > 2

# --- Parameter für Spektrum (b): Defektes Graphen ---
params_si_defekt =  [520, 8, 0.8]
params_d_defekt =   [1350, 40, 0.7] # I(D)/I(G) ~ 0.7
params_g_defekt =   [1582, 20, 1.0] # G-Peak oft etwas breiter bei Defekten
params_2d_defekt =  [2685, 45, 1.0] # I(2D)/I(G) ~ 1.0, breiter

# --- Spektren generieren ---
# Ideales Spektrum
y_ideal = (lorentzian(x, *params_si_ideal) +
           lorentzian(x, *params_g_ideal) +
           lorentzian(x, *params_2d_ideal))
y_ideal += np.random.normal(0, 0.02, x.shape) # Rauschen

# Defektes Spektrum
y_defekt = (lorentzian(x, *params_si_defekt) +
            lorentzian(x, *params_d_defekt) +
            lorentzian(x, *params_g_defekt) +
            lorentzian(x, *params_2d_defekt))
y_defekt += np.random.normal(0, 0.02, x.shape) # Rauschen


# --- Plotten ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot (a): Ideales Spektrum
ax1.plot(x, y_ideal, color='blue', linewidth=1.5)
ax1.set_title('(a) Ideales Raman-Spektrum einer Graphen-Monolage', fontsize=14)
ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
ax1.text(520, params_si_ideal[1] + 0.1, 'Si', ha='center', fontsize=11)
ax1.text(1582, params_g_ideal[1] + 0.1, 'G', ha='center', fontsize=11)
ax1.text(2680, params_2d_ideal[1] + 0.1, '2D', ha='center', fontsize=11)
ax1.grid(linestyle='--', alpha=0.6)
ax1.set_ylim(-0.1, 3.0)

# Plot (b): Defektes Spektrum
ax2.plot(x, y_defekt, color='red', linewidth=1.5)
ax2.set_title('(b) Typisches Raman-Spektrum von defektreichem Graphen', fontsize=14)
ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
ax2.text(520, params_si_defekt[1] + 0.1, 'Si', ha='center', fontsize=11)
ax2.text(1350, params_d_defekt[1] + 0.1, 'D', ha='center', fontsize=11, color='red', weight='bold')
ax2.text(1582, params_g_defekt[1] + 0.1, 'G', ha='center', fontsize=11)
ax2.text(2685, params_2d_defekt[1] + 0.1, '2D', ha='center', fontsize=11)
ax2.grid(linestyle='--', alpha=0.6)
ax2.set_ylim(-0.1, 2.0)


plt.xlim(400, 3000)
plt.tight_layout()
plt.savefig('vergleich_graphen_spektren.png', dpi=300)
plt.show()