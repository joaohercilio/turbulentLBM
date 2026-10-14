import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

N, RE = 512+128, 1600

L     = N / (2 * np.pi)        
u0    = 1 / L                  
nu    = 1 / RE
nodes = N ** 3

BLACK, RED, BLUE, GREEN = "#000000", "#C1272D", "#267BCF", "#45CF26" 

mpl.rcParams.update({
    "figure.figsize":       (3.9, 2.85),
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


import re
t_dns, E_dns = [], []
with open("spectral.dat") as f:
    for line in f:
        m = re.search(r"t=([-\d.eE+]+)\s+E=([-\d.eE+]+)", line)
        if m:
            t_dns.append(float(m.group(1)))
            E_dns.append(float(m.group(2)))
t_dns = np.array(t_dns)
E_dns = np.array(E_dns)
dissipation_dns = -np.gradient(E_dns, t_dns)

t_dns_wang, _, dissipation_dns_wang, _ = np.loadtxt("DNS.dat", unpack=True)


fig, ax = plt.subplots()

ax.plot(t_dns, dissipation_dns, color=BLACK, lw=1.1, zorder=2, label="Pseudo-spectral Joao")
ax.plot(t_dns_wang, dissipation_dns_wang, color=GREEN, lw=1.1, zorder=2, label="Pseudo-spectral Wang")
ax.set_xlabel(r"$t^{*}$")
ax.set_ylabel(r"$-\,\mathrm{d}k/\mathrm{d}t$")
ax.set_xlim(0, 20)
ax.set_ylim(0, 0.0150)
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99))

axi = ax.inset_axes([0.665, 0.520, 0.320, 0.420])
axi.plot(t_dns, dissipation_dns, color=BLACK, lw=0.9, zorder=2)
axi.plot(t_dns_wang, dissipation_dns_wang, color=GREEN, lw=0.9, zorder=2)
axi.set_xlim(8, 10)
axi.set_ylim(0.0104, 0.0134)
axi.set_xticks([8, 9, 10])
axi.set_yticks([0.011, 0.012, 0.013])
axi.tick_params(labelsize=6, width=0.45, length=2.0, pad=1.5)
axi.tick_params(which="minor", length=1.1)
for spine in axi.spines.values():
    spine.set_linewidth(0.45)
fig.savefig("dissipation.pdf")




