import os
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import glob
import h5py
import numpy as np
import tables as T

BLACK, RED, BLUE, GREEN = "#000000", "#C1272D", "#267BCF", "#2E8B57"

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

STYLE = {
    "d3q15": dict(color=GREEN, marker="s", markersize=2.4, markerfacecolor="none", markeredgewidth=0.7),
    "d3q19": dict(color=RED, marker="o", markersize=2.6, markerfacecolor="none", markeredgewidth=0.7),
    "d3q27": dict(color=BLUE, marker="x", markersize=2.8, markeredgewidth=0.7),
}
FIG = "results/figures"
C = T.C
E_MIN = 1e-14
MARKS = lambda n: np.unique(np.geomspace(1, n, 25).astype(int)) - 1
MA = np.mean([T.read_params(p)["Ma_rms"] for p in glob.glob("initial/initial_*.txt")])
ma = lambda b: f"Ma = {b*MA:.3f}"


def load(path):
    return np.loadtxt(f"{path}/statistics_mean.dat", ndmin=2)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def series(name, ylabel, f, log=False, hline=None):
    s = load("results/spectral")
    for o in T.ORDERS:
        for b in T.SCALES:
            cases = [(l, f"results/lbm/{l}_{o}/Ma_x{b}") for l in T.LATTICES]
            cases = [(l, p) for l, p in cases if os.path.exists(f"{p}/statistics_mean.dat")]
            if not cases:
                continue
            fig, ax = plt.subplots()
            ax.plot(s[:, 1], f(s), color=BLACK, lw=1.1, zorder=2, label="Pseudo-spectral")
            for l, p in cases:
                m = load(p)
                ax.plot(m[:, 1], f(m), lw=0, label=l.upper(), zorder=3, **STYLE[l])
            if hline is not None:
                ax.axhline(hline, color=BLACK, lw=0.5, ls=":", zorder=0)
            if log:
                ax.set_yscale("log")
            ax.set_xlabel(r"$t/t_e$")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 5)
            ax.set_title(f"{o} order, {ma(b)}")
            ax.legend(loc="best")
            save(fig, f"{FIG}/{o}_Ma_x{b}/{name}.pdf")




ax3 = lambda q: (lambda a: np.mean([a[:, C[q + d]] for d in T.AX], axis=0))
an = lambda d: (lambda a: a[:, C["ratio" + d]] - ax3("ratio")(a))
col = lambda c: (lambda a: a[:, C[c]])


def band(cfg, f, rel=True):
    """erro pareado LBM - espectral (relativo em % se rel): t, media e desvio padrao entre realizacoes"""
    root = cfg.split("lbm/")[0]
    P = []
    for r in T.realizations(cfg):
        if T.post.ok(f"{root}spectral/realization{r}"):
            a, s = T.load(f"{cfg}/realization{r}"), T.load(f"{root}spectral/realization{r}")
            fs = f(s[[np.argmin(np.abs(s[:, 1] - t)) for t in a[:, 1]]])
            P.append(100*(f(a) - fs)/fs if rel else f(a) - fs)
    n = min(len(p) for p in P)
    P = np.array([p[:n] for p in P])
    return a[:n, 1], P.mean(axis=0), P.std(axis=0, ddof=1)

K0 = np.loadtxt("results/spectral/statistics_mean.dat", ndmin=2)[0]
series("energy", r"$K/K_0$", lambda a: a[:, C["K"]]/K0[C["K"]], log=True)
series("dissipation", r"$\varepsilon/\varepsilon_0$", lambda a: a[:, C["eps"]]/K0[C["eps"]], log=True)
series("reynolds", r"$Re_\lambda$", lambda a: a[:, C["ReL"]])
series("skewness", r"$S$", ax3("S"))
series("flatness", r"$F$", ax3("F"))
series("ratio", r"$I_r$", ax3("ratio"), hline=2)
series("anisotropy110", r"$I_{r,110} - I_r$", an("110"), hline=0)
series("anisotropy111", r"$I_{r,111} - I_r$", an("111"), hline=0)

zs = np.load("results/spectral/spectrum.npz")
for o in T.ORDERS:
    for b in T.SCALES:
        cases = [(l, f"results/lbm/{l}_{o}/Ma_x{b}/spectrum.npz") for l in T.LATTICES]
        cases = [(l, p) for l, p in cases if os.path.exists(p)]
        if not cases:
            continue
        for ts in (0.4, 2.0, 4.8):
            fig, ax = plt.subplots()
            i = np.argmin(np.abs(zs["t_te"] - ts))
            k = zs["kc"][1:]
            ax.loglog(k, zs["E"][i, 1:], color=BLACK, lw=1.1, zorder=2, label="Pseudo-spectral")
            for l, p in cases:
                z = np.load(p)
                j = np.argmin(np.abs(z["t_te"] - ts))
                ax.loglog(k, z["E"][j, 1:], lw=0, markevery=MARKS(len(k)), label=l.upper(), zorder=3, **STYLE[l])
            kr = np.array([2.0, 30.0])
            Er = 10*zs["E"][i, 2]*(kr/kr[0])**(-5/3)
            ax.loglog(kr, Er, color=BLACK, lw=0.6, ls="--", zorder=1)
            ax.text(kr[1]*1.1, Er[1], r"$\kappa^{-5/3}$", va="center")
            ax.axvline(zs["N"]/3, color=BLACK, lw=0.6, ls=":", zorder=0)
            ax.text(0.97*zs["N"]/3, 1e-8, r"$\kappa = N/3$", rotation=90, ha="right", va="center", fontsize=7)
            ax.set_xlabel(r"$\kappa$")
            ax.set_ylabel(r"$E(\kappa)$")
            ax.set_xlim(1, zs["kc"][-1])
            ax.set_ylim(E_MIN, 2*zs["E"][:, 1:].max())
            ax.set_title(rf"{o} order, {ma(b)}, $t/t_e \approx {ts:g}$")
            ax.legend(loc="upper right")
            axin = ax.inset_axes([0.1, 0.08, 0.4, 0.4])
            kz = (k >= 60) & (k <= zs["N"]/3)
            axin.loglog(k[kz], zs["E"][i, 1:][kz], color=BLACK, lw=1.1)
            for l, p in cases:
                z = np.load(p)
                j = np.argmin(np.abs(z["t_te"] - ts))
                axin.loglog(k[kz], z["E"][j, 1:][kz], lw=0, markevery=10, **STYLE[l])
            axin.tick_params(labelsize=6)
            axin.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            save(fig, f"{FIG}/{o}_Ma_x{b}/spectrum_t{ts:g}.pdf")

            fig, ax = plt.subplots()
            eps = np.loadtxt("results/spectral/statistics_mean.dat", ndmin=2)[i, C["eps"]]
            ax.semilogx(k, zs["Pi"][i, 1:]/eps, color=BLACK, lw=1.1, zorder=2, label="Pseudo-spectral")
            for l, p in cases:
                z = np.load(p)
                j = np.argmin(np.abs(z["t_te"] - ts))
                ax.semilogx(k, z["Pi"][j, 1:]/eps, lw=0, markevery=MARKS(len(k)), label=l.upper(), zorder=3, **STYLE[l])
            ax.axhline(1, color=BLACK, lw=0.5, ls=":", zorder=0)
            ax.set_xlabel(r"$\kappa$")
            ax.set_ylabel(r"$\Pi(\kappa)/\varepsilon$")
            ax.set_xlim(1, zs["N"]/3)
            ax.set_title(rf"{o} order, {ma(b)}, $t/t_e \approx {ts:g}$")
            ax.legend(loc="best")
            save(fig, f"{FIG}/{o}_Ma_x{b}/flux_t{ts:g}.pdf")

            for j3, dn in ((3, "110"), (4, "111")):
                fig, ax = plt.subplots()
                eta = np.loadtxt("results/spectral/statistics_mean.dat", ndmin=2)[i, C["eta"]]
                D = zs["DLL"][i]
                ax.semilogx(zs["r"][1:]/eta, D[j3, 1:]/D[:3, 1:].mean(axis=0), color=BLACK, lw=1.1, label="Pseudo-spectral")
                for l, p in cases:
                    z = np.load(p)
                    j = np.argmin(np.abs(z["t_te"] - ts))
                    D = z["DLL"][j]
                    ax.semilogx(z["r"][1:]/eta, D[j3, 1:]/D[:3, 1:].mean(axis=0), lw=0, markevery=MARKS(len(z["r"]) - 1), label=l.upper(), **STYLE[l])
                ax.axhline(1, color=BLACK, lw=0.5, ls=":")
                ax.set_xlabel(r"$r/\eta$")
                ax.set_ylabel(rf"$D_{{LL}}^{{({dn})}}/D_{{LL}}$")
                ax.set_title(rf"{o} order, {ma(b)}, $t/t_e \approx {ts:g}$")
                ax.legend(loc="best")
                save(fig, f"{FIG}/{o}_Ma_x{b}/dll{dn}_t{ts:g}.pdf")

for q, sym, name in (("S", "S", "S"), ("F", "F", "F"), ("ratio", "I_r", "I_r")):
    for o in T.ORDERS:
        for b in T.SCALES:
            fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.4), sharey=True)
            for ax, l in zip(axs, T.LATTICES):
                cfg = f"lbm/{l}_{o}/Ma_x{b}"
                if T.realizations(cfg):
                    for f, ls, lab in ((ax3(q), "-", rf"${sym}$"), (col(q + "110"), "--", rf"${sym}^{{(110)}}$"),
                                       (col(q + "111"), ":", rf"${sym}^{{(111)}}$")):
                        t, m, e = band(cfg, f)
                        ax.plot(t, m, color=STYLE[l]["color"], ls=ls, label=lab)
                        ax.fill_between(t, m - e, m + e, color=STYLE[l]["color"], alpha=0.15, lw=0)
                ax.axhline(0, color=BLACK, lw=0.5, ls=":", zorder=0)
                ax.set_xlim(0, 5)
                ax.set_xlabel(r"$t/t_e$")
                ax.set_title(l.upper())
                ax.legend(loc="best")
            axs[0].set_ylabel(r"$\overline{e}$ [%]")
            fig.suptitle(f"{o} order, {ma(b)}", y=1.02)
            save(fig, f"{FIG}/{o}_Ma_x{b}/{name}_directions.pdf")

for o in T.ORDERS:
    for b in T.SCALES:
        fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.6))
        for ax, q, sym in zip(axs, ("K", "eps"), ("K", r"\varepsilon")):
            for l in T.LATTICES:
                cfg = f"lbm/{l}_{o}/Ma_x{b}"
                if T.realizations(cfg):
                    t, m, e = band(cfg, T.Q[q][1])
                    ax.plot(t, m, color=STYLE[l]["color"], label=l.upper())
                    ax.fill_between(t, m - e, m + e, color=STYLE[l]["color"], alpha=0.15, lw=0)
            ax.axhline(0, color=BLACK, lw=0.5, ls=":", zorder=0)
            ax.set_xlim(0, 5)
            ax.set_xlabel(r"$t/t_e$")
            ax.set_ylabel(r"$\overline{e}$ [%]")
            ax.set_title(rf"${sym}$")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        fig.suptitle(f"{o} order, {ma(b)}", y=1.12)
        save(fig, f"{FIG}/{o}_Ma_x{b}/energy_error.pdf")

spec = {int(d.split("realization")[1]): d for d in __import__("glob").glob("spectral/realization*") if T.post.ok(d)}
fig, axs = plt.subplots(2, 2, figsize=(7.8, 4.8), sharex=True)
for ax, q, sym in zip(axs.flat, ("eps", "F", "S", "ratio111"), (r"\varepsilon", "F", "S", r"I_r^{(111)}")):
    for l in T.LATTICES:
        for o, ls in (("2nd", "-"), ("3rd", "--")):
            pts = [(b, *(lambda d: (100*np.mean(d), 100*np.std(d, ddof=1)))(T.relative(f"lbm/{l}_{o}/Ma_x{b}", spec, T.Q[q][1])))
                   for b in T.SCALES if T.realizations(f"lbm/{l}_{o}/Ma_x{b}")]
            if pts:
                b, m, e = np.array(pts).T
                ax.errorbar(b*MA, m, yerr=e, ls=ls, lw=0.8, elinewidth=0.6, capsize=1.5, label=f"{l.upper()} {o}", **STYLE[l])
    ax.axhline(0, color=BLACK, lw=0.5, ls=":", zorder=0)
    ax.set_title(rf"${sym}$")
    ax.set_xticks([b*MA for b in T.SCALES], [f"{b*MA:.3f}" for b in T.SCALES])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
for ax in axs[1]:
    ax.set_xlabel(r"Ma $= u_{\mathrm{rms}}/c_s$")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\overline{e}$ [%]")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
save(fig, f"{FIG}/summary/mach_order.pdf")

for q, label in (("aniso111", r"$\Delta(I_{r,111} - I_r)$"),
                 ("aniso110", r"$\Delta(I_{r,110} - I_r)$"),
                 ("S", r"$\Delta S$"), ("S110", r"$\Delta S_{110}$"), ("S111", r"$\Delta S_{111}$"),
                 ("F", r"$\Delta F$"), ("F110", r"$\Delta F_{110}$"), ("F111", r"$\Delta F_{111}$"),
                 ("K", r"$\Delta K/K$"), ("eps", r"$\Delta\varepsilon/\varepsilon$")):
    fig, ax = plt.subplots()
    for l in T.LATTICES:
        for o, ls in (("2nd", "-"), ("3rd", "--")):
            pts = [(b,) + T.paired(f"lbm/{l}_{o}/Ma_x{b}", spec, q, (0, 2.5)) for b in T.SCALES
                   if os.path.isdir(f"lbm/{l}_{o}/Ma_x{b}")]
            pts = [p for p in pts if p[1] is not None]
            if pts:
                b, m, e, n = np.array(pts, dtype=float).T
                ax.errorbar(b + (0.04 if o == "3rd" else -0.04), m, yerr=e*np.sqrt(n), ls=ls, lw=0.7,
                            elinewidth=0.6, capsize=1.5, label=f"{l.upper()} {o}", **STYLE[l])
    ax.axhline(0, color=BLACK, lw=0.5, ls=":")
    ax.set_xticks(T.SCALES, [f"{b*MA:.3f}" for b in T.SCALES])
    ax.set_xlabel(r"Ma $= u_{\mathrm{rms}}/c_s$")
    ax.set_ylabel(label + r", $0 < t/t_e < 2.5$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=6)
    save(fig, f"{FIG}/summary/paired_{q}.pdf")

for q, label in (("aniso111", r"I_r^{(111)} - I_r"), ("S", "S"), ("F", "F")):
    f = T.Q[q][1]
    rs = sorted(spec)
    ps = [T.window(T.load(spec[r])[:, 1], f(T.load(spec[r])), (0, 2.5)) for r in rs]
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.6))
    axs[0].plot(rs, ps, lw=0, marker="o", markersize=3.5, color=BLACK, label="Pseudo-spectral")
    for l in T.LATTICES:
        cfg = f"lbm/{l}_{T.ORDERS[0]}/Ma_x1"
        lb = [T.window(T.load(f"{cfg}/realization{r}")[:, 1], f(T.load(f"{cfg}/realization{r}")), (0, 2.5)) for r in rs]
        axs[0].plot(rs, lb, lw=0, label=l.upper(), **STYLE[l])
        axs[1].plot(rs, np.array(lb) - ps, lw=0, **STYLE[l])
    axs[1].axhline(0, color=BLACK, lw=0.5, ls=":", zorder=0)
    for ax in axs:
        ax.set_xticks(rs)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel("realization")
    axs[0].set_ylabel(rf"${label}$")
    axs[1].set_ylabel(rf"$\Delta {label}$")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.suptitle(rf"{T.ORDERS[0]} order, {ma(1)}, mean over $0 < t/t_e < 2.5$", y=1.12)
    save(fig, f"{FIG}/summary/realizations_{q}.pdf")

fig, ax = plt.subplots(figsize=(3.9, 2.8))
for root, ls, tag in (("../HIT512_eta08/", ":", r"$\eta^+ = 0.8$"), ("../HIT512/", "--", r"$\eta^+ = 1.1$"),
                      ("../HIT512_eta2/", "-", r"$\eta^+ = 2$")):
    zs = np.load(f"{root}results/spectral/spectrum.npz")
    i = np.argmin(np.abs(zs["t_te"] - 2.0))
    n = int(zs["N"])//3
    for l in T.LATTICES:
        z = np.load(f"{root}results/lbm/{l}_3rd/Ma_x1/spectrum.npz")
        j = np.argmin(np.abs(z["t_te"] - 2.0))
        ax.plot(zs["kc"][1:n], z["E"][j, 1:n]/zs["E"][i, 1:n], color=STYLE[l]["color"], ls=ls)
ax.axhline(1, color=BLACK, lw=0.5, ls=":", zorder=0)
ax.axvline(zs["N"]/3, color=BLACK, lw=0.6, ls=":", zorder=0)
ax.text(0.98*zs["N"]/3, 0.75, r"$\kappa = N/3$", rotation=90, ha="right", va="center", fontsize=7)
ax.set_xlim(20, 185)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$E_{\mathrm{LBM}}(\kappa)/E_{\mathrm{PS}}(\kappa)$")
ax.set_title(r"3rd order, Ma $\approx$ 0.029, $t/t_e \approx 2$")
line = mpl.lines.Line2D
lat = ax.legend([line([], [], color=STYLE[l]["color"]) for l in T.LATTICES], [l.upper() for l in T.LATTICES], loc="lower left")
ax.add_artist(lat)
ax.legend([line([], [], color=BLACK, ls=ls) for ls in (":", "--", "-")], [r"$\eta^+ = 0.8$", r"$\eta^+ = 1.1$", r"$\eta^+ = 2$"],
          loc="lower left", bbox_to_anchor=(0.25, 0))
save(fig, f"{FIG}/resolution.pdf")

fig, ax = plt.subplots()
for root, ls, tag in (("../HIT512_eta08/", ":", r"$\eta^+ = 0.8$"), ("../HIT512/", "--", r"$\eta^+ = 1.1$"),
                      ("../HIT512_eta2/", "-", r"$\eta^+ = 2$")):
    zs = np.load(f"{root}results/spectral/spectrum.npz")
    z = np.load(f"{root}results/lbm/d3q27_3rd/Ma_x1/spectrum.npz")
    i, j = np.argmin(np.abs(zs["t_te"] - 2.0)), np.argmin(np.abs(z["t_te"] - 2.0))
    k = zs["kc"][1:]
    ax.loglog(k, zs["E"][i, 1:], color=BLACK, lw=1.0, ls=ls, label=f"PS, {tag}")
    ax.loglog(k, z["E"][j, 1:], lw=0, markevery=MARKS(len(k)), zorder=3, **STYLE["d3q27"])
ax.loglog([], [], lw=0, label="D3Q27", **STYLE["d3q27"])
ax.axvline(zs["N"]/3, color=BLACK, lw=0.6, ls=":", zorder=0)
ax.text(0.97*zs["N"]/3, 1e-8, r"$\kappa = N/3$", rotation=90, ha="right", va="center", fontsize=7)
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$E(\kappa)$")
ax.set_xlim(1, zs["kc"][-1])
ax.set_ylim(E_MIN, 2*zs["E"][:, 1:].max())
ax.set_title(r"3rd order, Ma $\approx$ 0.029, $t/t_e \approx 2$")
ax.legend(loc="lower left")
save(fig, f"{FIG}/resolution_spectrum.pdf")
