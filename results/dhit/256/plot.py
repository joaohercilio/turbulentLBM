import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

N = 256

L     = 2 * np.pi
delta = L / N
t0    = 318.457776566172

BLACK, RED, BLUE = "#000000", "#C1272D", "#267BCF"

mpl.rcParams.update({
    "figure.figsize":       (3.9, 2.6),
    "figure.dpi":           600,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "font.family":          "STIXGeneral",
    "mathtext.fontset":     "stix",
    "font.size":            8,
    "axes.labelsize":       9,
    "legend.fontsize":      7.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "axes.linewidth":       0.6,
    "lines.linewidth":      1.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.45,
    "ytick.minor.width":    0.45,
    "xtick.major.size":     3.2,
    "ytick.major.size":     3.2,
    "xtick.minor.size":     1.8,
    "ytick.minor.size":     1.8,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "legend.frameon":       False,
    "legend.handlelength":  1.7,
    "legend.handletextpad": 0.5,
    "legend.labelspacing":  0.35,
    "legend.borderaxespad": 0.0,
})

t_spectral, energy_spectral = np.loadtxt("energySpectral.out", skiprows=1, unpack=True)
k0 = energy_spectral[0]
t_spectral = t_spectral / t0
k_spectral = energy_spectral / k0
dissipation_spectral = -np.gradient(k_spectral, t_spectral)

step, energy, _ = np.loadtxt("statistics_d3q19.dat", unpack=True)
t_d3q19 = step * delta / t0
k_d3q19 = energy / k0
dissipation_d3q19 = -np.gradient(k_d3q19, t_d3q19)

step, energy, _ = np.loadtxt("statistics_d3q27.dat", unpack=True)
t_d3q27 = step * delta / t0
k_d3q27 = energy / k0
dissipation_d3q27 = -np.gradient(k_d3q27, t_d3q27)

fig, ax = plt.subplots()

ax.loglog(t_spectral, dissipation_spectral, color=BLACK, lw=1.1, zorder=2, label="Pseudo-spectral")
ax.loglog(t_d3q19, dissipation_d3q19, color=RED, lw=0.0, marker="o", markersize=2.6, markevery=4, markerfacecolor="none", markeredgewidth=0.7, zorder=3, label="D3Q19")
ax.loglog(t_d3q27, dissipation_d3q27, color=BLUE, lw=0.0, marker="x", markersize=2.8, markevery=4, markeredgewidth=0.7, zorder=4, label="D3Q27")

ax.set_xlabel(r"$t/t_0$")
ax.set_ylabel(r"$\langle\varepsilon\rangle/\langle\varepsilon_0\rangle$")
ax.set_xlim(1e-1, 1e1)
ax.set_ylim(1e-5, 3)
ax.legend(loc="lower left", bbox_to_anchor=(0.06, 0.06))

fig.savefig("dissipation.pdf")
