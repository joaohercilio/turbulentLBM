"""Tabelas comparativas da campanha HIT512 (results/tables/*.csv e *.md).

   Toda media em janela de tempo e feita por realizacao e so depois entre
   realizacoes: o erro (desvio/sqrt(n)) respeita a correlacao temporal.
   Diferencas LBM - espectral sao pareadas: mesma IC, mesmos tempos.
"""

import glob
import os
import re
import numpy as np
import h5py
from scipy import stats
import post

def read_params(path):
    """{nome: valor} de um arquivo "nome valor" por linha (o mesmo de hitgpu.read_params)"""
    P = {}
    for line in open(path):
        line = line.split("#")[0].split(None, 1)
        if len(line) == 2:
            v = line[1].strip()
            for conv in (int, float, str):
                try:
                    P[line[0]] = conv(v)
                    break
                except ValueError:
                    pass
    return P


OUT = "results/tables"
WINDOWS = [(0, 1), (1, 2.5), (2.5, 5)]
LATTICES = ["d3q15", "d3q19", "d3q27"]
ORDERS = ["3rd"]
SCALES = [1, 2, 3]
C = post.COL
AX = ("100", "010", "001")

Q = {
    "K":         ("K (LBM-PS)/PS [%]",     lambda s: s[:, C["K"]], True),
    "eps":       ("eps (LBM-PS)/PS [%]",   lambda s: s[:, C["eps"]], True),
    "S":         ("S eixos",               lambda s: np.mean([s[:, C["S" + a]] for a in AX], axis=0), False),
    "F":         ("F eixos",               lambda s: np.mean([s[:, C["F" + a]] for a in AX], axis=0), False),
    "S110":      ("S110",                  lambda s: s[:, C["S110"]], False),
    "S111":      ("S111",                  lambda s: s[:, C["S111"]], False),
    "F110":      ("F110",                  lambda s: s[:, C["F110"]], False),
    "F111":      ("F111",                  lambda s: s[:, C["F111"]], False),
    "ratio":     ("I_r eixos",             lambda s: np.mean([s[:, C["ratio" + a]] for a in AX], axis=0), False),
    "ratio110":  ("I_r,110",               lambda s: s[:, C["ratio110"]], False),
    "ratio111":  ("I_r,111",               lambda s: s[:, C["ratio111"]], False),
    "aniso110":  ("I_r,110 - I_r eixos",   lambda s: s[:, C["ratio110"]] - np.mean([s[:, C["ratio" + a]] for a in AX], axis=0), False),
    "aniso111":  ("I_r,111 - I_r eixos",   lambda s: s[:, C["ratio111"]] - np.mean([s[:, C["ratio" + a]] for a in AX], axis=0), False),
}


def load(d):
    return np.loadtxt(f"{d}/statistics.dat", ndmin=2)


def fmt(m, e, scale=1, digits=4):
    if m is None or not np.isfinite(m):
        return "-"
    s = f"{scale*m:+.{digits}f}"
    return s + (f" ± {scale*e:.{digits}f}" if np.isfinite(e) else "")


def stat(a):
    a = np.asarray(a)
    return (a.mean(), a.std(ddof=1)/np.sqrt(len(a)) if len(a) > 1 else np.nan) if len(a) else (None, None)


def sig(m, e, n):
    return "*" if m is not None and n > 1 and abs(m) > stats.t.ppf(0.975, n - 1)*e else ""


def window(t, y, w):
    m = (t > w[0] + 1e-9) & (t <= w[1] + 1e-9)
    return y[m].mean()


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


def realizations(cfg):
    return [int(d.split("realization")[1]) for d in sorted(glob.glob(f"{cfg}/realization*")) if post.ok(d)]


def paired(cfg, spec, q, w, other=None):
    """media em janela de (cfg - other) por realizacao; other = espectral se None"""
    vals = []
    _, f, rel = Q[q]
    for r in realizations(cfg):
        if other is None:
            if r not in spec:
                continue
            a, b = load(f"{cfg}/realization{r}"), load(spec[r])
            b = b[[np.argmin(np.abs(b[:, 1] - t)) for t in a[:, 1]]]
        else:
            if not post.ok(f"{other}/realization{r}"):
                continue
            a, b = load(f"{cfg}/realization{r}"), load(f"{other}/realization{r}")
        ya, yb = f(a), f(b)
        d = (ya - yb)/yb if rel else ya - yb
        vals.append(window(a[:, 1], d, w))
    return stat(vals) + (len(vals),)


def configs():
    return [(l, o, b, f"lbm/{l}_{o}/Ma_x{b}") for b in SCALES for o in ORDERS for l in LATTICES
            if os.path.isdir(f"lbm/{l}_{o}/Ma_x{b}")]


def accuracy_tables(spec):
    out = []
    for q, (label, _, rel) in Q.items():
        rows = []
        for l, o, b, cfg in configs():
            res = [paired(cfg, spec, q, w) for w in WINDOWS]
            n = max(r[2] for r in res)
            if n == 0:
                continue
            rows.append([l, o, f"x{b}", n] + [fmt(m, e, 100 if rel else 1, 3 if rel else 4) + sig(m, e, k) for m, e, k in res])
        head = ["lattice", "ordem", "Mach", "n"] + [f"{a}-{b} te" for a, b in WINDOWS]
        out.append(write(f"accuracy_{q}", f"Erro pareado LBM - espectral: {label}", head, rows))
    return out


def anisotropy_table(spec):
    rows = []
    for q in ("aniso111", "aniso110"):
        f = Q[q][1]
        vals = [[window(load(spec[r])[:, 1], f(load(spec[r])), w) for r in sorted(spec)] for w in WINDOWS]
        rows.append(["espectral", "-", "-", q, len(spec)] + [fmt(*stat(v)) for v in vals])
        for l, o, b, cfg in configs():
            rs = realizations(cfg)
            if not rs:
                continue
            vals = [[window(load(f"{cfg}/realization{r}")[:, 1], f(load(f"{cfg}/realization{r}")), w) for r in rs] for w in WINDOWS]
            rows.append([l, o, f"x{b}", q, len(rs)] + [fmt(*stat(v)) for v in vals])
    head = ["caso", "ordem", "Mach", "assinatura", "n"] + [f"{a}-{b} te" for a, b in WINDOWS]
    return write("anisotropy", "Assinatura de anisotropia absoluta (isotropico = 0)", head, rows,
                 "O valor do espectral e o nivel de ruido amostral de uma realizacao finita; o efeito da rede e a diferenca pareada (accuracy_aniso*).")


def spectra_table(spec):
    bands = [(1, 10), (10, 50), (50, 100), (100, 170)]
    times = [0.4, 2.0, 4.8]
    rows = []
    for l, o, b, cfg in configs():
        for ts in times:
            vals = {bd: [] for bd in bands}
            for r in realizations(cfg):
                if r not in spec:
                    continue
                zl, zs = np.load(f"{cfg}/realization{r}/spectrum.npz"), np.load(f"{spec[r]}/spectrum.npz")
                il, i_s = np.argmin(np.abs(zl["t_te"] - ts)), np.argmin(np.abs(zs["t_te"] - ts))
                rel = zl["E"][il]/zs["E"][i_s] - 1
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
    """erro relativo pareado (LBM - PS)/PS, media em 0-2.5 te, um valor por realizacao"""
    d = []
    for r in realizations(cfg):
        if r in spec:
            a, p = load(f"{cfg}/realization{r}"), load(spec[r])
            p = p[[np.argmin(np.abs(p[:, 1] - t)) for t in a[:, 1]]]
            d.append(window(a[:, 1], (f(a) - f(p))/f(p), (0, 2.5)))
    return d


def relative_table(spec, o, b, group):
    rs = sorted(spec)
    rows, tex = [], []
    for q in GROUPS[group]:
        f = Q[q][1]
        ps = np.array([window(load(spec[r])[:, 1], f(load(spec[r])), (0, 2.5)) for r in rs])
        err = []
        for l in LATTICES:
            cfg = f"lbm/{l}_{o}/Ma_x{b}"
            d = relative(cfg, spec, f)
            err.append((100*np.mean(d), 100*np.std(d, ddof=1)))
        best = np.argmin([abs(m) for m, _ in err])
        cells = [f"{m:+.2f} ± {e:.2f}" for m, e in err]
        sampling = f"{100*ps.std(ddof=1)/abs(ps.mean()):.2f}"
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
                 f"n = {len(rs)} realizacoes. PS: media entre realizacoes. Desvio: desvio padrao entre realizacoes / |media| (amostragem). "
                 "Redes: media ± desvio padrao entre realizacoes do erro relativo pareado. Negrito: menor |erro|.")


def effect_table(name, title, pairs, cols=("lattice", "ordem/Mach", "comparacao")):
    qs = ("K", "eps", "S", "F", "aniso110", "aniso111")
    rows = []
    for label, cfg, ref in pairs:
        if not (os.path.isdir(cfg) and os.path.isdir(ref)):
            continue
        res = [paired(cfg, None, q, (0, 2.5), other=ref) for q in qs]
        if res[0][2] == 0:
            continue
        rows.append(label + [res[0][2]] + [fmt(m, e, 100 if Q[q][2] else 1, 3 if Q[q][2] else 4) + sig(m, e, n)
                                           for (m, e, n), q in zip(res, qs)])
    head = list(cols) + ["n", "K rel [%]", "eps rel [%]", "S", "F", "aniso110", "aniso111"]
    return write(name, title, head, rows, "Media em 0-2.5 te, diferenca pareada realizacao a realizacao. "
                 "* = |media| > t(97.5%, n-1) x erro: diferenca significativa a 95%.")


def performance_table():
    rows = []
    for l, o, b, cfg in configs():
        ml, steps = [], None
        for r in realizations(cfg):
            log = open(f"{cfg}/realization{r}/log.txt").read()
            m = re.search(r"MLUPS: ([\d.e+]+)", log)
            if m:
                ml.append(float(m.group(1)))
            steps = len(np.loadtxt(f"{cfg}/realization{r}/statistics_lbm.dat", ndmin=2))
        P = read_params("initial/initial_0.txt")
        tau = 0.5 + b*(P["tau"] - 0.5)
        ma, mamax = b*P["Ma_rms"], b*P["Ma_max"]
        nst = int(np.ceil(5*P["te_lbm"]/b))
        if ml:
            rows.append([l, o, f"x{b}", f"{tau:.5f}", f"{ma:.4f}", f"{mamax:.3f}", nst, len(ml), f"{np.mean(ml):.0f}"])
    head = ["lattice", "ordem", "Mach", "tau", "Ma_rms", "Ma_max", "passos (5 te)", "n", "MLUPS (com saidas)"]
    return write("performance", "Parametros e desempenho por configuracao", head, rows,
                 "Ma_rms e Ma_max da IC de referencia (initial_0); MLUPS inclui as 25 escritas de campo.")


def initial_conditions_table():
    files = sorted(glob.glob("initial/initial_*.vtkhdf"), key=lambda p: int(re.findall(r"\d+", p)[-1]))
    rows = []
    for p in files:
        a = read_params(os.path.splitext(p)[0] + ".txt")
        rows.append([os.path.basename(p), f"{a['t']:.1f}",
                     str(a.get("source", "spin-up original")), f"{a.get('perturbation', 0):g}",
                     f"{a['ReL']:.1f}", f"{a['eta_delta']:.3f}", f"{a['urms']:.5f}", f"{a['L11']:.3f}",
                     f"{a['te']:.2f}", f"{a['Ma_rms']:.4f}", f"{a['Ma_max']:.4f}"])
    head = ["IC", "t", "fonte", "perturbacao", "Re_lambda", "eta/delta", "urms", "L11", "te", "Ma_rms", "Ma_max"]
    md = write("initial_conditions", "Condicoes iniciais (initial_*.txt)", head, rows)
    cache = "results/ic_correlation.npz"
    if len(files) > 1 and (not os.path.exists(cache) or len(np.load(cache)["names"]) != len(files)):
        U = []
        for p in files:
            with h5py.File(p, "r") as f:
                N = int(read_params(os.path.splitext(p)[0] + ".txt")["N"])
                k = np.fft.fftfreq(N, 1/N)
                low = (np.abs(k) <= 3)
                comp = []
                for nm in ("ux", "uy", "uz"):
                    uh = np.fft.rfftn(f["VTKHDF/PointData/" + nm][:].T)/N**3
                    comp.append(uh[np.ix_(low, low, np.arange(4))].copy())
                U.append(np.array(comp))
        kz = np.arange(4); wz = np.where(kz == 0, 1.0, 2.0)
        dot = lambda a, b: float(np.real(np.sum(a*np.conj(b)*wz)))
        R = np.array([[dot(a, b)/np.sqrt(dot(a, a)*dot(b, b)) for b in U] for a in U])
        np.savez(cache, R=R, names=[os.path.basename(p) for p in files])
    if os.path.exists(cache):
        z = np.load(cache)
        names = [n.replace("initial_", "").replace(".vtkhdf", "").replace(".h5", "") for n in z["names"]]
        rows = [[f"initial_{a}"] + [f"{x:+.3f}" for x in row] for a, row in zip(names, z["R"])]
        md += "\n\n" + write("ic_correlation", "Correlacao das grandes escalas (|kappa_i| <= 3) entre as ICs", ["IC"] + names, rows,
                             "Proxima de 0 fora da diagonal = realizacoes independentes.")
    return md


def failures_table():
    rows = []
    for l, o, b, cfg in configs():
        n_ok = len(realizations(cfg))
        fail = sorted(int(d.split("realization")[1]) for d in glob.glob(f"{cfg}/realization*")
                      if os.path.exists(f"{d}/failed") or (os.path.exists(f"{d}/done") and not post.ok(d)))
        rows.append([l, o, f"x{b}", n_ok, len(fail), str(fail) if fail else "-"])
    return write("runs", "Corridas concluidas e falhas (instabilidade / NaN)", ["lattice", "ordem", "Mach", "ok", "falhas", "realizacoes com falha"], rows)


if __name__ == "__main__":
    spec, _ = post.run()
    parts = [failures_table(), initial_conditions_table(), performance_table()]
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
