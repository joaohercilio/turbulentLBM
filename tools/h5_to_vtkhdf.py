"""Converte condicoes iniciais do formato antigo (.h5) para VTKHDF (ImageData).

   Antigo: datasets drho, ux, uy, uz, sxx, sxy, sxz, syy, syz, szz na raiz, indice [x, y, z].
   Novo:   arquivo.vtkhdf so com os 10 campos em VTKHDF/PointData/<nome>, shape [z, y, x] (x varia
           mais rapido); os atributos da raiz vao para arquivo.txt, "nome valor" por linha.
           Spacing = 2 pi/N no HIT (atributo N), 1 caso contrario.
   Um campo por vez na memoria. O .h5 original nao e apagado.
   Uso: python tools/h5_to_vtkhdf.py arquivo.h5 [...]   (grava arquivo.vtkhdf e arquivo.txt ao lado)
"""

import os
import sys
import numpy as np
import h5py

NAMES = ("drho", "ux", "uy", "uz", "sxx", "sxy", "sxz", "syy", "syz", "szz")

for src in sys.argv[1:]:
    base = os.path.splitext(src)[0]
    dst = base + ".vtkhdf"
    with h5py.File(src, "r") as fin, h5py.File(dst + ".part", "w") as fout:
        NX, NY, NZ = fin["ux"].shape
        with open(base + ".txt", "w") as g:
            for key, val in fin.attrs.items():
                val = val.decode() if isinstance(val, bytes) else val.item() if hasattr(val, "item") else val
                g.write(f"{key} {val!r}\n" if isinstance(val, float) else f"{key} {val}\n")
        g = fout.create_group("VTKHDF")
        g.attrs["Version"] = np.array([1, 0], np.int32)
        g.attrs.create("Type", np.bytes_("ImageData"))
        g.attrs["WholeExtent"] = np.array([0, NX - 1, 0, NY - 1, 0, NZ - 1], np.int64)
        g.attrs["Origin"] = np.zeros(3)
        g.attrs["Spacing"] = np.full(3, 2*np.pi/int(fin.attrs["N"]) if "N" in fin.attrs else 1.0)
        g.attrs["Direction"] = np.eye(3).ravel()
        point = g.create_group("PointData")
        for name in NAMES:
            point.create_dataset(name, data=np.ascontiguousarray(fin[name][:].transpose(2, 1, 0)))
    os.replace(dst + ".part", dst)
    print(f"{src} -> {dst} + {base}.txt", flush=True)
