import glob
import os
import re
import numpy as np
import h5py
from scipy import stats

HEADER = open(sorted(glob.glob("spectral/realization*/statistics.dat"))[0]).readline().lstrip("# ").strip()
C = {n: i for i, n in enumerate(HEADER.split())}


def number(d):
    return int(d.split("realization")[1])


def ok(d):
    if not (os.path.exists(f"{d}/done") and os.path.exists(f"{d}/spectrum.npz")):
        return False
    return np.isfinite(np.loadtxt(f"{d}/statistics.dat", ndmin=2)).all()


def realizations(cfg):
    return sorted(number(d) for d in glob.glob(f"{cfg}/realization*") if ok(d))


def failures(cfg):
    return sorted(number(d) for d in glob.glob(f"{cfg}/realization*")
                  if os.path.exists(f"{d}/failed") or (os.path.exists(f"{d}/done") and not ok(d)))


def spectral():
    return {r: f"spectral/realization{r}" for r in realizations("spectral")}


def err(a):
    return a.std(axis=0, ddof=1)/np.sqrt(len(a)) if len(a) > 1 else np.full(a.shape[1:], np.nan)


def savetxt(path, a):
    np.savetxt(path, a, header=HEADER, comments="# ")


def ensemble(dirs, out):
    os.makedirs(out, exist_ok=True)
    S = [np.loadtxt(f"{d}/statistics.dat", ndmin=2) for d in dirs]
    n = min(len(x) for x in S)
    S = np.array([x[:n] for x in S])
    savetxt(f"{out}/statistics_mean.dat", S.mean(axis=0))
    savetxt(f"{out}/statistics_err.dat", err(S))
    Z = [np.load(f"{d}/spectrum.npz") for d in dirs]
    res = {k: Z[0][k] for k in ("kc", "r", "nu", "N")}
    res["t"], res["t_te"] = Z[0]["t"][:n], Z[0]["t_te"][:n]
    for k in ("E", "Pi", "DLL"):
        a = np.array([z[k][:n] for z in Z])
        res[k], res[k + "_err"] = a.mean(axis=0), err(a)
    np.savez(f"{out}/spectrum.npz", **res)
    return S


def averages():
    spec = spectral()
    ensemble(spec.values(), "results/spectral")
    for cfg in sorted(glob.glob("lbm/*/Ma_x*")):
        reals = [r for r in realizations(cfg) if r in spec]
        out = cfg.replace("lbm/", "results/lbm/", 1)
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/realizations.txt", "w") as g:
            g.write(f"usadas: {reals}\nfalhas: {failures(cfg)}\n")
        if not reals:
            continue
        P = []
        for r, l in zip(reals, ensemble([f"{cfg}/realization{r}" for r in reals], out)):
            s = np.loadtxt(f"{spec[r]}/statistics.dat", ndmin=2)
            match = np.array([np.argmin(np.abs(s[:, 1] - t)) for t in l[:, 1]])
            assert np.abs(s[match, 1] - l[:, 1]).max() < 2e-3, f"{cfg} r{r}: tempos nao batem"
            d = l - s[match]
            d[:, :2] = l[:, :2]
            P.append(d)
        P = np.array(P)
        savetxt(f"{out}/paired_mean.dat", P.mean(axis=0))
        savetxt(f"{out}/paired_err.dat", err(P))
    return spec


OUT = "results/tables"
WINDOWS = [(0, 1), (1, 2.5), (2.5, 5)]
LATTICES = ["d3q15", "d3q19", "d3q27"]
ORDERS = ["2nd", "3rd"]
SCALES = [1, 2, 3]
AX = ("100", "010", "001")

axes = lambda q: (lambda s: np.mean([s[:, C[q + a]] for a in AX], axis=0))
col = lambda c: (lambda s: s[:, C[c]])
aniso = lambda d: (lambda s: s[:, C["ratio" + d]] - axes("ratio")(s))

Q = {
    "K":        ("K (LBM-PS)/PS [%]",   col("K"), True),
    "eps":      ("eps (LBM-PS)/PS [%]", col("eps"), True),
    "S":        ("S eixos",             axes("S"), False),
    "F":        ("F eixos",             axes("F"), False),
    "S110":     ("S110",                col("S110"), False),
    "S111":     ("S111",                col("S111"), False),
    "F110":     ("F110",                col("F110"), False),
    "F111":     ("F111",                col("F111"), False),
    "ratio":    ("I_r eixos",           axes("ratio"), False),
    "ratio110": ("I_r,110",             col("ratio110"), False),
    "ratio111": ("I_r,111",             col("ratio111"), False),
    "aniso110": ("I_r,110 - I_r eixos", aniso("110"), False),
    "aniso111": ("I_r,111 - I_r eixos", aniso("111"), False),
}


def load(d):
    return np.loadtxt(f"{d}/statistics.dat", ndmin=2)


def fmt(m, e, scale=1, digits=4):
    if m is None or not np.isfinite(m):
        return "-"
    s = f"{scale*m:+.{digits}f}"
    return s + (f" ± {scale*e:.{digits}f}" if np.isfinite(e) else "")


def fmt_q(m, e, q):
    return fmt(m, e, 100, 3) if Q[q][2] else fmt(m, e)


def stat(a):
    a = np.asarray(a)
    return (a.mean(), a.std(ddof=1)/np.sqrt(len(a)) if len(a) > 1 else np.nan) if len(a) else (None, None)


def sig(m, e, n):
    return "*" if m is not None and n > 1 and abs(m) > stats.t.ppf(0.975, n - 1)*e else ""


def window(t, y, w):
    m = (t > w[0] + 1e-9) & (t <= w[1] + 1e-9)
    return y[m].mean()


def mean_in(d, f, w):
    s = load(d)
    return window(s[:, 1], f(s), w)


def matched(a, b):
    return b[[np.argmin(np.abs(b[:, 1] - t)) for t in a[:, 1]]]


def write(name, title, head, rows, note=""):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.csv", "w") as g:
        g.write(";".join(head) + "\n")
        for r in rows:
            g.write(";".join(str(x) for x in r) + "\n")
    md = [f"## {title}", ""] + ([note, ""] if note else [])
    md += ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    with open(f"{OUT}/{name}.md", "w") as g:
        g.write("\n".join(md) + "\n")
    return "\n".join(md)


def paired(cfg, spec, q, w, other=None):
    vals = []
    _, f, rel = Q[q]
    for r in realizations(cfg):
        if other is None:
            if r not in spec:
                continue
            a = load(f"{cfg}/realization{r}")
            b = matched(a, load(spec[r]))
        else:
            if not ok(f"{other}/realization{r}"):
                continue
            a, b = load(f"{cfg}/realization{r}"), load(f"{other}/realization{r}")
        ya, yb = f(a), f(b)
        vals.append(window(a[:, 1], (ya - yb)/yb if rel else ya - yb, w))
    return stat(vals) + (len(vals),)


def configs():
    return [(l, o, b, f"lbm/{l}_{o}/Ma_x{b}") for b in SCALES for o in ORDERS for l in LATTICES
            if os.path.isdir(f"lbm/{l}_{o}/Ma_x{b}")]


def accuracy_tables(spec):
    out = []
    head = ["lattice", "ordem", "Mach", "n"] + [f"{a}-{b} te" for a, b in WINDOWS]
    for q, (label, _, _) in Q.items():
        rows = []
        for l, o, b, cfg in configs():
            res = [paired(cfg, spec, q, w) for w in WINDOWS]
            n = max(r[2] for r in res)
            if n:
                rows.append([l, o, f"x{b}", n] + [fmt_q(m, e, q) + sig(m, e, k) for m, e, k in res])
        out.append(write(f"accuracy_{q}", f"Erro pareado LBM - espectral: {label}", head, rows))
    return out


def anisotropy_table(spec):
    rows = []
    for q in ("aniso111", "aniso110"):
        f = Q[q][1]
        rows.append(["espectral", "-", "-", q, len(spec)]
                    + [fmt(*stat([mean_in(spec[r], f, w) for r in sorted(spec)])) for w in WINDOWS])
        for l, o, b, cfg in configs():
            rs = realizations(cfg)
            if rs:
                rows.append([l, o, f"x{b}", q, len(rs)]
                            + [fmt(*stat([mean_in(f"{cfg}/realization{r}", f, w) for r in rs])) for w in WINDOWS])
    head = ["caso", "ordem", "Mach", "assinatura", "n"] + [f"{a}-{b} te" for a, b in WINDOWS]
    return write("anisotropy", "Assinatura de anisotropia absoluta (isotropico = 0)", head, rows,
                 "O valor do espectral e o nivel de ruido amostral de uma realizacao finita; o efeito da rede e a diferenca pareada (accuracy_aniso*).")


def spectra_table(spec):
    bands = [(1, 10), (10, 50), (50, 100), (100, 170)]
    rows = []
    for l, o, b, cfg in configs():
        for ts in (0.4, 2.0, 4.8):
            vals = {bd: [] for bd in bands}
            for r in realizations(cfg):
                if r not in spec:
                    continue
                zl, zs = np.load(f"{cfg}/realization{r}/spectrum.npz"), np.load(f"{spec[r]}/spectrum.npz")
                rel = zl["E"][np.argmin(np.abs(zl["t_te"] - ts))]/zs["E"][np.argmin(np.abs(zs["t_te"] - ts))] - 1
                for bd in bands:
                    vals[bd].append(rel[bd[0]:bd[1]].mean())
            if vals[bands[0]]:
                rows.append([l, o, f"x{b}", f"{ts:g}", len(vals[bands[0]])] + [fmt(*stat(vals[bd]), 100, 2) for bd in bands])
    head = ["lattice", "ordem", "Mach", "t/te", "n"] + [f"kappa {a}-{b} [%]" for a, b in bands]
    return write("spectra", "Erro relativo pareado de E(kappa) por banda, (LBM-PS)/PS [%]", head, rows,
                 "Acima de kappa = N/3 = 170 o espectral e truncado (dealiasing); bandas so ate 170.")


GROUPS = {"energy": ("K", "eps"), "skewness": ("S", "S110", "S111"),
          "flatness": ("F", "F110", "F111"), "ratio": ("ratio", "ratio110", "ratio111")}
TEX = {"K": "$K$", "eps": r"$\varepsilon$", "S": "$S$", "S110": "$S^{(110)}$", "S111": "$S^{(111)}$",
       "F": "$F$", "F110": "$F^{(110)}$", "F111": "$F^{(111)}$",
       "ratio": "$I_r$", "ratio110": "$I_r^{(110)}$", "ratio111": "$I_r^{(111)}$"}


def relative(cfg, spec, f):
    d = []
    for r in realizations(cfg):
        if r in spec:
            a = load(f"{cfg}/realization{r}")
            p = matched(a, load(spec[r]))
            d.append(window(a[:, 1], (f(a) - f(p))/f(p), (0, 2.5)))
    return d


def relative_table(spec, o, b, group):
    rows, tex = [], []
    for q in GROUPS[group]:
        f = Q[q][1]
        ps = np.array([mean_in(spec[r], f, (0, 2.5)) for r in sorted(spec)])
        err = [(100*np.mean(d), 100*np.std(d, ddof=1)) for d in (relative(f"lbm/{l}_{o}/Ma_x{b}", spec, f) for l in LATTICES)]
        best = np.argmin([abs(m) for m, _ in err])
        sampling = f"{100*ps.std(ddof=1)/abs(ps.mean()):.2f}"
        cells = [f"{m:+.2f} ± {e:.2f}" for m, e in err]
        rows.append([Q[q][0], f"{ps.mean():.4g}", sampling] + [f"**{c}**" if i == best else c for i, c in enumerate(cells)])
        cells = [f"${m:+.2f} \\pm {e:.2f}$" for m, e in err]
        tex.append(" & ".join([TEX[q], sampling] + [f"$\\mathbf{{{c[1:-1]}}}$" if i == best else c for i, c in enumerate(cells)]) + r" \\")
    name = f"relative_{group}_{o}_Ma_x{b}"
    with open(f"{OUT}/{name}.tex", "w") as g:
        g.write("\n".join([r"\begin{tabular}{@{}ccccc@{}}", r"\toprule",
                           r" & & \multicolumn{3}{c}{$\overline{e} \pm \operatorname{std}(e_k)$ [\%]} \\", r"\cmidrule(l){3-5}",
                           r"$q$ & $\sigma_s$ [\%] & D3Q15 & D3Q19 & D3Q27 \\",
                           r"\midrule"] + tex + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    head = ["quantidade", "PS", "desvio entre realizacoes [%]"] + [f"{l.upper()} [%]" for l in LATTICES]
    return write(name, f"Erro relativo (LBM - PS)/PS, {group}, {o} ordem, Ma x{b}, media em 0-2.5 te", head, rows,
                 f"n = {len(spec)} realizacoes. PS: media entre realizacoes. Desvio: desvio padrao entre realizacoes / |media| (amostragem). "
                 "Redes: media ± desvio padrao entre realizacoes do erro relativo pareado. Negrito: menor |erro|.")


def effect_table(name, title, pairs, cols=("lattice", "ordem/Mach", "comparacao")):
    qs = ("K", "eps", "S", "F", "aniso110", "aniso111")
    rows = []
    for label, cfg, ref in pairs:
        if not (os.path.isdir(cfg) and os.path.isdir(ref)):
            continue
        res = [paired(cfg, None, q, (0, 2.5), other=ref) for q in qs]
        if res[0][2]:
            rows.append(label + [res[0][2]] + [fmt_q(m, e, q) + sig(m, e, n) for (m, e, n), q in zip(res, qs)])
    head = list(cols) + ["n", "K rel [%]", "eps rel [%]", "S", "F", "aniso110", "aniso111"]
    return write(name, title, head, rows, "Media em 0-2.5 te, diferenca pareada realizacao a realizacao. "
                 "* = |media| > t(97.5%, n-1) x erro: diferenca significativa a 95%.")


def performance_table(spec):
    ma = np.sqrt(3)*np.mean([load(d)[0, C["urms"]] for d in spec.values()])
    rows = []
    for l, o, b, cfg in configs():
        ml = [float(m.group(1)) for r in realizations(cfg)
              if (m := re.search(r"MLUPS: ([\d.e+]+)", open(f"{cfg}/realization{r}/log.txt").read()))]
        if ml:
            rows.append([l, o, f"x{b}", f"{b*ma:.4f}", len(ml), f"{np.mean(ml):.0f}"])
    return write("performance", "Desempenho por configuracao", ["lattice", "ordem", "Mach", "Ma_rms", "n", "MLUPS (com saidas)"], rows,
                 "Ma_rms medio das ICs; MLUPS inclui as 25 escritas de campo.")


def initial_conditions_table(spec):
    rows = [[f"initial_{r - 1}"] + [f"{load(d)[0, C[q]]:{f}}" for q, f in
             (("ReL", ".1f"), ("eta_delta", ".3f"), ("urms", ".5f"), ("L11", ".3f"), ("te", ".2f"))]
            + [f"{np.sqrt(3)*load(d)[0, C['urms']]:.4f}"] for r, d in spec.items()]
    md = write("initial_conditions", "Condicoes iniciais", ["IC", "Re_lambda", "eta/delta", "urms", "L11", "te", "Ma_rms"], rows)
    files = sorted(glob.glob("initial/initial_*.vtkhdf"), key=lambda p: int(re.findall(r"\d+", p)[-1]))
    if len(files) < 2:
        return md
    U = []
    for p in files:
        with h5py.File(p, "r") as f:
            N = f["VTKHDF/PointData/ux"].shape[0]
            low = np.abs(np.fft.fftfreq(N, 1/N)) <= 3
            U.append(np.array([(np.fft.rfftn(f["VTKHDF/PointData/" + c][:].T)/N**3)[np.ix_(low, low, np.arange(4))]
                               for c in ("ux", "uy", "uz")]))
    wz = np.where(np.arange(4) == 0, 1.0, 2.0)
    dot = lambda a, b: float(np.real(np.sum(a*np.conj(b)*wz)))
    names = [re.findall(r"\d+", p)[-1] for p in files]
    rows = [[f"initial_{a}"] + [f"{dot(x, y)/np.sqrt(dot(x, x)*dot(y, y)):+.3f}" for y in U] for a, x in zip(names, U)]
    return md + "\n\n" + write("ic_correlation", "Correlacao das grandes escalas (|kappa_i| <= 3) entre as ICs", ["IC"] + names, rows,
                                 "Proxima de 0 fora da diagonal = realizacoes independentes.")


def failures_table():
    rows = []
    for l, o, b, cfg in configs():
        fail = failures(cfg)
        rows.append([l, o, f"x{b}", len(realizations(cfg)), len(fail), str(fail) if fail else "-"])
    return write("runs", "Corridas concluidas e falhas (instabilidade / NaN)",
                 ["lattice", "ordem", "Mach", "ok", "falhas", "realizacoes com falha"], rows)


if __name__ == "__main__":
    spec = averages()
    parts = [failures_table(), initial_conditions_table(spec), performance_table(spec)]
    parts += accuracy_tables(spec)
    parts.append(anisotropy_table(spec))
    parts += [relative_table(spec, o, b, g) for o in ORDERS for b in SCALES for g in GROUPS]
    parts.append(spectra_table(spec))
    parts.append(effect_table("order_effect", "Efeito da ordem: 3a - 2a (pareado)",
                              [([l, f"Ma_x{b}", "3rd - 2nd"], f"lbm/{l}_3rd/Ma_x{b}", f"lbm/{l}_2nd/Ma_x{b}") for b in SCALES for l in LATTICES]))
    parts.append(effect_table("mach_effect", "Efeito do Mach: x2 - x1 e x3 - x1 (pareado)",
                              [([l, o, f"x{b} - x1"], f"lbm/{l}_{o}/Ma_x{b}", f"lbm/{l}_{o}/Ma_x1") for o in ORDERS for l in LATTICES for b in (2, 3)]))
    parts.append(effect_table("lattice_effect", "Diferenca entre redes (pareado)",
                              [([o, f"Ma_x{b}", f"{a} - {c}"], f"lbm/{a}_{o}/Ma_x{b}", f"lbm/{c}_{o}/Ma_x{b}")
                               for b in SCALES for o in ORDERS for a, c in (("d3q19", "d3q15"), ("d3q27", "d3q15"), ("d3q27", "d3q19"))],
                              ("ordem", "Mach", "comparacao")))
    with open(f"{OUT}/summary.md", "w") as g:
        g.write(f"# Campanha HIT512: resumo\n\nEspectral: {len(spec)} realizacoes ({sorted(spec)}).\n\n" + "\n\n".join(parts) + "\n")
    print(open(f"{OUT}/summary.md").read()[:3000])
