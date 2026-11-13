"""Medias de ensemble e diferencas pareadas da campanha HIT512.

   results/spectral/            statistics_mean.dat, statistics_err.dat, spectrum.npz
   results/lbm/<ns>/Ma_x<b>/    idem + paired_mean.dat, paired_err.dat (LBM - espectral,
                                realizacao a realizacao, nos mesmos tempos) e realizations.txt
   Erro = desvio padrao entre realizacoes / sqrt(n). So entram realizacoes com
   espectral e LBM concluidos e finitos.
"""

import glob
import os
import numpy as np

with open("spectral/realization1/statistics.dat") as f:
    HEADER = f.readline().lstrip("# ").strip()
COL = {n: i for i, n in enumerate(HEADER.split())}


def ok(d):
    if not (os.path.exists(f"{d}/done") and os.path.exists(f"{d}/spectrum.npz")):
        return False
    return np.isfinite(np.loadtxt(f"{d}/statistics.dat", ndmin=2)).all()


def err(a):
    return a.std(axis=0, ddof=1)/np.sqrt(len(a)) if len(a) > 1 else np.full(a.shape[1:], np.nan)


def ensemble(dirs, out):
    os.makedirs(out, exist_ok=True)
    S = [np.loadtxt(f"{d}/statistics.dat", ndmin=2) for d in dirs]
    n = min(len(x) for x in S)
    S = np.array([x[:n] for x in S])
    np.savetxt(f"{out}/statistics_mean.dat", S.mean(axis=0), header=HEADER, comments="# ")
    np.savetxt(f"{out}/statistics_err.dat", err(S), header=HEADER, comments="# ")
    Z = [np.load(f"{d}/spectrum.npz") for d in dirs]
    res = {k: Z[0][k] for k in ("kc", "r", "n_dir", "m_dir", "nu", "N")}
    res["t"], res["t_te"] = Z[0]["t"][:n], Z[0]["t_te"][:n]
    for k in ("E", "Pi", "Ep", "DLL"):
        a = np.array([z[k][:n] for z in Z])
        res[k], res[k + "_err"] = a.mean(axis=0), err(a)
    np.savez(f"{out}/spectrum.npz", **res)
    return S


def run():
    spec = {int(d.split("realization")[1]): d for d in glob.glob("spectral/realization*") if ok(d)}
    ensemble([spec[r] for r in sorted(spec)], "results/spectral")
    summary = []
    for cfg in sorted(glob.glob("lbm/*/Ma_x*")):
        reals = sorted(int(d.split("realization")[1]) for d in glob.glob(f"{cfg}/realization*") if ok(d))
        failed = sorted(int(d.split("realization")[1]) for d in glob.glob(f"{cfg}/realization*")
                        if os.path.exists(f"{d}/failed") or (os.path.exists(f"{d}/done") and not ok(d)))
        reals = [r for r in reals if r in spec]
        out = cfg.replace("lbm/", "results/lbm/", 1)
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/realizations.txt", "w") as g:
            g.write(f"usadas: {reals}\nfalhas: {failed}\n")
        summary.append((cfg, reals, failed))
        if not reals:
            continue
        L = ensemble([f"{cfg}/realization{r}" for r in reals], out)
        P = []
        for r, l in zip(reals, L):
            s = np.loadtxt(f"{spec[r]}/statistics.dat", ndmin=2)
            match = np.array([np.argmin(np.abs(s[:, 1] - t)) for t in l[:, 1]])
            assert np.abs(s[match, 1] - l[:, 1]).max() < 2e-3, f"{cfg} r{r}: tempos nao batem"
            d = l - s[match]
            d[:, :2] = l[:, :2]
            P.append(d)
        P = np.array(P)
        np.savetxt(f"{out}/paired_mean.dat", P.mean(axis=0), header=HEADER, comments="# ")
        np.savetxt(f"{out}/paired_err.dat", err(P), header=HEADER, comments="# ")
    return spec, summary


if __name__ == "__main__":
    spec, summary = run()
    print(f"espectral: {len(spec)} realizacoes {sorted(spec)}")
    for cfg, reals, failed in summary:
        print(f"{cfg:24} n={len(reals)} {reals}" + (f"  falhas {failed}" if failed else ""))
